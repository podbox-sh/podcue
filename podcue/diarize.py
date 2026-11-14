"""Speaker turns from pyannote segmentation-3.0 plus WeSpeaker ECAPA embeddings.

Ten-second windows every second go through the powerset segmentation model. Each local
speaker in each window gets one ECAPA embedding. Unit-normalized embeddings are clustered
with centroid linkage at pyannote 3.1's threshold. A small cluster is folded into the nearest
large one only when it is within that same distance; a farther cluster stays its own speaker.
Every 270-sample frame takes the cluster with the most votes.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np

RATE = 16_000
CHUNK = 160_000
HOP = 16_000
FRAME_STEP = 270
FRAME = 400
SHIFT = 160
FFT = 512
BINS = 80
EMBEDDING_SIZE = 192
EPS = float(np.finfo(np.float32).eps)
DIARIZATION_31_CENTROID_DISTANCE = 0.7045654963945799
MIN_CLUSTER_SIZE = 12
POWERSET = (
    frozenset(),
    frozenset({0}),
    frozenset({1}),
    frozenset({2}),
    frozenset({0, 1}),
    frozenset({0, 2}),
    frozenset({1, 2}),
)


class Turn(NamedTuple):
    start_ms: int
    end_ms: int
    speaker: int


class Tracks(NamedTuple):
    turns: list[Turn]
    centroids: list[list[float]]


def _mel(freq):
    return 1127.0 * np.log(1.0 + freq / 700.0)


def _mel_banks():
    low = _mel(20.0)
    delta = (_mel(RATE / 2.0) - low) / (BINS + 1)
    bins = _mel(np.arange(FFT // 2) * (RATE / FFT))[None, :]
    left = low + np.arange(BINS)[:, None] * delta
    center = left + delta
    right = center + delta
    banks = np.maximum(0.0, np.minimum((bins - left) / delta, (right - bins) / delta))
    return np.pad(banks, ((0, 0), (0, 1)))


BANKS = _mel_banks()
WINDOW = 0.54 - 0.46 * np.cos(2.0 * np.pi * np.arange(FRAME) / (FRAME - 1))


def fbank(samples: np.ndarray) -> np.ndarray:
    """Kaldi fbank as torchaudio computes it: 80 bins, hamming, 25 ms / 10 ms, dither 0."""
    scaled = samples.astype(np.float64) * 32768.0
    if len(scaled) < FRAME:
        raise ValueError("speaker span is shorter than one 25 ms frame")
    count = 1 + (len(scaled) - FRAME) // SHIFT
    index = np.arange(FRAME)[None, :] + SHIFT * np.arange(count)[:, None]
    frames = scaled[index]
    frames = frames - frames.mean(axis=1, keepdims=True)
    frames = frames - 0.97 * np.concatenate([frames[:, :1], frames[:, :-1]], axis=1)
    power = np.abs(np.fft.rfft(frames * WINDOW, n=FFT)) ** 2
    return np.log(np.maximum(power @ BANKS.T, EPS))


def chunk_audio(audio: np.ndarray, start: int) -> np.ndarray:
    piece = audio[start : start + CHUNK]
    if len(piece) == CHUNK:
        return piece
    padded = np.zeros(CHUNK, dtype=np.float32)
    padded[: len(piece)] = piece
    return padded


def take_mask(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Boolean-index axis 0. A mask that is not the same length is empty, not an error.

    The same shape mismatch crashes faster-whisper when an alignment has no frames:
    a length-1 jump mask applied to a length-0 time array.
    """
    if mask.ndim != 1 or values.shape[:1] != mask.shape:
        return np.empty((0, *values.shape[1:]), dtype=values.dtype)
    return values[mask]


def speaker_samples(chunk: np.ndarray, classes: np.ndarray, speaker: int) -> np.ndarray:
    """Samples where one local speaker talks, alone when there is enough of that."""
    active = np.zeros(CHUNK, dtype=bool)
    clean = np.zeros(CHUNK, dtype=bool)
    for frame, label in enumerate(classes):
        members = POWERSET[int(label)]
        if speaker not in members:
            continue
        begin = frame * FRAME_STEP
        end = min(CHUNK, begin + FRAME_STEP)
        active[begin:end] = True
        if len(members) == 1:
            clean[begin:end] = True
    chosen = clean if int(clean.sum()) > 400 else active
    if int(chosen.sum()) <= 400:
        return np.empty(0, dtype=np.float32)
    return take_mask(chunk, chosen)


def centroid_merges(unit: np.ndarray) -> list[tuple[int, int, float]]:
    """Centroid-linkage merges in scipy's (left, right, distance) order, nearest-neighbor chain."""
    n = unit.shape[0]
    dist2 = np.sum(unit * unit, axis=1)[:, None] + np.sum(unit * unit, axis=1)[None, :] - 2.0 * (unit @ unit.T)
    np.fill_diagonal(dist2, np.inf)
    dist2[dist2 < 0] = 0
    sizes = np.ones(n, dtype=np.int64)
    active = np.ones(n, dtype=bool)
    node_of = np.arange(n, dtype=np.int64)
    next_id = n
    nn = np.argmin(dist2, axis=1)
    nnd = dist2[np.arange(n), nn]
    merges = []
    for _ in range(n - 1):
        active_idx = np.flatnonzero(active)
        a = int(active_idx[np.argmin(nnd[active_idx])])
        b = int(nn[a])
        d_ab = float(dist2[a, b])
        left = int(node_of[a])
        right = int(node_of[b])
        if left > right:
            left, right = right, left
        merges.append((left, right, float(np.sqrt(max(d_ab, 0.0)))))
        na = float(sizes[a])
        nb = float(sizes[b])
        nt = na + nb
        row = (na / nt) * dist2[a] + (nb / nt) * dist2[b] + (-(na * nb) / (nt * nt)) * d_ab
        active[b] = False
        row[~active] = np.inf
        row[a] = np.inf
        row[row < 0] = 0
        dist2[a, :] = row
        dist2[:, a] = row
        dist2[b, :] = np.inf
        dist2[:, b] = np.inf
        sizes[a] = int(na + nb)
        node_of[a] = next_id
        next_id += 1
        invalid = ((nn == a) | (nn == b) | (np.arange(n) == a)) & active
        rows = np.flatnonzero(invalid)
        if len(rows) > 0:
            sub = dist2[rows]
            chosen = np.argmin(sub, axis=1)
            nn[rows] = chosen
            nnd[rows] = sub[np.arange(len(rows)), chosen]
        valid = active & ~invalid
        closer = valid & (dist2[:, a] < nnd)
        nn[closer] = a
        nnd[closer] = dist2[closer, a]
    return merges


def flat_clusters(merges: list[tuple[int, int, float]], n: int, threshold: float) -> np.ndarray:
    """scipy fcluster(criterion='distance') without scipy. Unclustered leaves get negative ids."""
    if n == 1:
        return np.array([0], dtype=np.int32)
    labels = np.full(n, -1, dtype=np.int32)
    visited = np.zeros(2 * n - 1, dtype=bool)
    stack_node = np.zeros(n, dtype=np.int32)
    k = 0
    stack_node[0] = 2 * n - 2
    leader = -1
    while k >= 0:
        node = int(stack_node[k])
        left, right, dist = merges[node - n]
        if leader == -1 and dist <= threshold:
            leader = node
        if left >= n and not visited[left]:
            visited[left] = True
            k += 1
            stack_node[k] = left
            continue
        if right >= n and not visited[right]:
            visited[right] = True
            k += 1
            stack_node[k] = right
            continue
        if left < n:
            labels[left] = leader
        if right < n:
            labels[right] = leader
        if leader == node:
            leader = -1
        k -= 1
    singleton = -2
    for index in range(n):
        if labels[index] < 0:
            labels[index] = singleton
            singleton -= 1
    return labels


def cluster_unit(unit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    count = unit.shape[0]
    if count == 1:
        return np.array([0], dtype=np.int32), unit.copy()
    labels = flat_clusters(centroid_merges(unit), count, DIARIZATION_31_CENTROID_DISTANCE)
    ids, counts = np.unique(labels, return_counts=True)
    min_size = min(MIN_CLUSTER_SIZE, max(1, round(0.1 * count)))
    large = ids[counts >= min_size]
    if len(large) == 0:
        return np.zeros(count, dtype=np.int32), unit.mean(axis=0, keepdims=True)
    large_centroids = np.vstack([unit[labels == kid].mean(axis=0) for kid in large])
    for kid in ids[counts < min_size]:
        members = labels == kid
        centroid = unit[members].mean(axis=0)
        distances = np.linalg.norm(large_centroids - centroid, axis=1)
        nearest = int(np.argmin(distances))
        # A small cluster far from every large voice is a different speaker, not a fold-in.
        if float(distances[nearest]) <= DIARIZATION_31_CENTROID_DISTANCE:
            labels[members] = large[nearest]
    _, compact = np.unique(labels, return_inverse=True)
    compact = compact.astype(np.int32)
    speakers = int(compact.max()) + 1
    centroids = np.vstack([unit[compact == kid].mean(axis=0) for kid in range(speakers)])
    return compact, centroids


def cluster_windows(vectors: np.ndarray) -> tuple[np.ndarray, list[list[float]]]:
    flat = vectors.reshape(-1, EMBEDDING_SIZE)
    finite = np.isfinite(flat).all(axis=1)
    labels = np.full(flat.shape[0], -2, dtype=np.int32)
    points = take_mask(flat, finite)
    if len(points) == 0:
        return labels.reshape(vectors.shape[:2]), []
    norms = np.maximum(np.linalg.norm(points, axis=1, keepdims=True), 1e-12)
    compact, centroids = cluster_unit(points / norms)
    if finite.shape == labels.shape and int(finite.sum()) == len(compact):
        labels[finite] = compact
    return labels.reshape(vectors.shape[:2]), [[float(value) for value in row] for row in centroids]


def stitch(audio_len: int, classes: np.ndarray, hard: np.ndarray) -> list[Turn]:
    """Vote each global frame to one cluster, then run-length encode the winners into turns."""
    n_speakers = int(hard.max()) + 1 if np.any(hard >= 0) else 0
    if n_speakers <= 0:
        return []
    n_global = (audio_len + FRAME_STEP - 1) // FRAME_STEP
    votes = np.zeros((n_global, n_speakers), dtype=np.uint16)
    for index, start in enumerate(range(0, audio_len, HOP)):
        row = classes[index]
        ids = hard[index]
        for frame, label in enumerate(row):
            members = POWERSET[int(label)]
            if not members:
                continue
            global_frame = (start + frame * FRAME_STEP) // FRAME_STEP
            if global_frame >= n_global:
                break
            for speaker in members:
                cluster_id = int(ids[speaker])
                if cluster_id < 0:
                    continue
                votes[global_frame, cluster_id] = min(65535, int(votes[global_frame, cluster_id]) + 1)
    winner = votes.argmax(axis=1)
    winner[votes.max(axis=1) == 0] = -1
    turns = []
    frame = 0
    while frame < n_global:
        speaker = int(winner[frame])
        end = frame + 1
        while end < n_global and int(winner[end]) == speaker:
            end += 1
        if speaker >= 0:
            start_ms = (frame * FRAME_STEP * 1000) // RATE
            end_ms = (min(end * FRAME_STEP, audio_len) * 1000) // RATE
            if end_ms > start_ms:
                turns.append(Turn(int(start_ms), int(end_ms), speaker))
        frame = end
    return turns


class Diarizer:
    def __init__(self, ecapa_path: Path, segmentation_path: Path):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.log_severity_level = 3
        providers = ["CPUExecutionProvider"]
        self._ecapa = ort.InferenceSession(str(ecapa_path), options, providers=providers)
        self._seg = ort.InferenceSession(str(segmentation_path), options, providers=providers)
        self._ecapa_feed = self._ecapa.get_inputs()[0].name
        self._seg_feed = self._seg.get_inputs()[0].name

    def _segment_all(self, audio: np.ndarray) -> np.ndarray:
        starts = list(range(0, len(audio), HOP))
        classes = np.empty((len(starts), 589), dtype=np.uint8)
        batch = 16
        for offset in range(0, len(starts), batch):
            group = starts[offset : offset + batch]
            wavs = np.stack([chunk_audio(audio, start) for start in group])[:, None, :]
            logits = self._seg.run(None, {self._seg_feed: wavs})[0]
            classes[offset : offset + len(group)] = np.argmax(logits, axis=-1).astype(np.uint8)
        return classes

    def _embed_one(self, samples: np.ndarray) -> np.ndarray:
        feats = fbank(samples)
        feats = feats - feats.mean(axis=0, keepdims=True)
        output = self._ecapa.run(None, {self._ecapa_feed: feats[None, :, :].astype(np.float32)})[0]
        return np.asarray(output, dtype=np.float32).reshape(-1)

    def _embed_all(self, audio: np.ndarray, classes: np.ndarray) -> np.ndarray:
        vectors = np.full((classes.shape[0], 3, EMBEDDING_SIZE), np.nan, dtype=np.float32)
        for index, start in enumerate(range(0, len(audio), HOP)):
            chunk = chunk_audio(audio, start)
            for speaker in range(3):
                samples = speaker_samples(chunk, classes[index], speaker)
                if len(samples) == 0:
                    continue
                try:
                    vectors[index, speaker] = self._embed_one(samples)
                except ValueError:
                    pass
        return vectors

    def diarize(self, audio: np.ndarray) -> Tracks:
        if len(audio) == 0:
            return Tracks([], [])
        classes = self._segment_all(audio)
        hard, centroids = cluster_windows(self._embed_all(audio, classes))
        return Tracks(stitch(len(audio), classes, hard), centroids)
