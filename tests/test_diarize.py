import numpy as np

from podcue.diarize import (
    CHUNK,
    EMBEDDING_SIZE,
    FRAME_STEP,
    HOP,
    Turn,
    centroid_merges,
    cluster_unit,
    cluster_windows,
    fbank,
    flat_clusters,
    speaker_samples,
    stitch,
)


def unit_rows(rows):
    rows = np.asarray(rows, dtype=np.float64)
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


def test_centroid_merges_join_nearest_pairs_first():
    points = unit_rows([[1, 0], [0.99, 0.01], [0, 1], [0.01, 0.99]])
    merges = centroid_merges(points)
    assert len(merges) == 3
    assert {merges[0][:2], merges[1][:2]} == {(0, 1), (2, 3)}
    assert merges[2][:2] == (4, 5)
    assert merges[2][2] > merges[1][2]


def test_flat_clusters_cut_at_the_threshold():
    points = unit_rows([[1, 0], [0.99, 0.01], [0, 1], [0.01, 0.99]])
    labels = flat_clusters(centroid_merges(points), 4, 0.5)
    assert labels[0] == labels[1] != labels[2] == labels[3]


def test_cluster_unit_folds_small_clusters_into_the_nearest_large_one():
    rng = np.random.default_rng(0)
    a = unit_rows(np.array([1.0, 0, 0]) + rng.normal(0, 0.01, (20, 3)))
    b = unit_rows(np.array([0, 1.0, 0]) + rng.normal(0, 0.01, (20, 3)))
    stray = unit_rows([[0.75, 0.66, 0.0]])
    labels, centroids = cluster_unit(np.vstack([a, b, stray]))
    assert len(centroids) == 2
    assert len(set(labels[:20])) == 1 and len(set(labels[20:40])) == 1
    assert labels[0] != labels[20]
    assert labels[40] in (labels[0], labels[20])


def test_cluster_windows_ignores_windows_without_embeddings():
    vectors = np.full((2, 3, EMBEDDING_SIZE), np.nan, dtype=np.float32)
    vectors[0, 0] = 1.0
    labels, centroids = cluster_windows(vectors)
    assert labels.tolist() == [[0, -2, -2], [-2, -2, -2]]
    assert len(centroids) == 1 and len(centroids[0]) == EMBEDDING_SIZE


def test_stitch_votes_frames_into_turns():
    audio_len = HOP * 2
    classes = np.zeros((2, 589), dtype=np.uint8)
    classes[0, :50] = 1  # local speaker 0 alone
    classes[0, 50:100] = 2  # local speaker 1 alone
    hard = np.array([[0, 1, -2], [-2, -2, -2]], dtype=np.int32)
    turns = stitch(audio_len, classes, hard)
    assert turns == [
        Turn(0, 50 * FRAME_STEP * 1000 // 16_000, 0),
        Turn(50 * FRAME_STEP * 1000 // 16_000, 100 * FRAME_STEP * 1000 // 16_000, 1),
    ]


def test_stitch_without_clusters_has_no_turns():
    classes = np.zeros((1, 589), dtype=np.uint8)
    assert stitch(HOP, classes, np.full((1, 3), -2, dtype=np.int32)) == []


def test_speaker_samples_prefer_clean_frames():
    chunk = np.arange(CHUNK, dtype=np.float32)
    classes = np.zeros(589, dtype=np.uint8)
    classes[0:10] = 1  # speaker 0 alone
    classes[10:20] = 4  # speakers 0 and 1 together
    samples = speaker_samples(chunk, classes, 0)
    assert len(samples) == 10 * FRAME_STEP
    assert len(speaker_samples(chunk, classes, 2)) == 0


def test_fbank_has_eighty_bins_per_ten_ms():
    feats = fbank(np.zeros(16_000, dtype=np.float32))
    assert feats.shape == (1 + (16_000 - 400) // 160, 80)
    assert np.isfinite(feats).all()
