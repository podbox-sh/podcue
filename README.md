# podcue

Turn podcast audio into timed, speaker-grouped cues that an agent can read.

podcue is a local-only command. It runs in the foreground, writes two files, and exits. Agents can use it optionally alongside the [PodBox skills](https://github.com/podbox-sh/skills). It is a simplified port of the engine in PodDrill.

## Requirements

- Python 3.12, 3.13 or 3.14 (onnxruntime 1.31.0 has no wheels for other versions). With [uv](https://docs.astral.sh/uv/) you don't need to install anything else in Python.
- `ffmpeg` on `PATH`.
- Network access on the first run, to download model weights.

## Usage

```sh
uvx podcue episode.mp3                       # writes ./episode.cues.json and ./episode.vtt
uvx podcue https://example.com/ep12.mp3 -o out/
uvx --from 'podcue[moonshine]' podcue ep.m4a --language ja
```

`podcue <audio file or URL> [-o OUT_DIR] [--language CODE]`

- The source can be any file ffmpeg can decode, or a direct `http(s)` link to one. podcue does not resolve web pages to audio.
- `--language` is one of `en de fr es it nl pl` (cactus-needle) or `ja ko zh` (Moonshine, from the `moonshine` extra). The default is `en`.
- The output names come from the source file name: `<name>.cues.json` and `<name>.vtt`. podcue prints both paths on stdout, one per line. Progress goes to stderr.

The pipeline converts the audio to 16 kHz mono WAV, diarizes the whole episode, joins each speaker's turns into regions, transcribes the regions in slices of up to 28 seconds, and groups the words into cues. A cue ends at sentence punctuation, after 12 seconds, or where a speaker region ends. Each cue gets the speaker whose diarization turns overlap it most.

### `<name>.cues.json`

```json
{
  "version": 1,
  "source": "episode.mp3",
  "durationMs": 3541200,
  "language": "en",
  "speakers": [
    { "id": "S1", "speakingMs": 1820400, "cueCount": 412, "nearestSpeaker": "S2", "nearestCosine": 0.3121 },
    { "id": "S2", "speakingMs": 1504300, "cueCount": 371, "nearestSpeaker": "S1", "nearestCosine": 0.3121 }
  ],
  "cues": [
    { "id": "c0001", "text": "Welcome back to the show.", "startMs": 80, "endMs": 1280, "speaker": "S1" }
  ]
}
```

| Field | Meaning |
| --- | --- |
| `version` | Schema version, `1`. |
| `source` | The path or URL exactly as given on the command line. |
| `durationMs` | Length of the decoded audio. |
| `language` | The `--language` the episode was transcribed with. |
| `speakers[].id` | `S1`, `S2`, … numbered in order of each speaker's first cue. A speaker is one voice cluster, not a known person. Two ids can turn out to be the same person. |
| `speakers[].speakingMs` | Total diarized speaking time of that voice. |
| `speakers[].cueCount` | Number of cues assigned to that voice. |
| `speakers[].nearestSpeaker`, `nearestCosine` | The other voice with the most similar ECAPA centroid, and that cosine similarity. A high value hints that two ids may be one person. Both are `null` when there is only one speaker. |
| `cues[].id` | `c0001`, `c0002`, … in time order. Summaries cite these. |
| `cues[].text` | The words of the cue. |
| `cues[].startMs`, `endMs` | Start of the first word and end of the last word, from the start of the episode. |
| `cues[].speaker` | A `speakers[].id`, or `null` when no diarization turn overlaps the cue. |

### `<name>.vtt`

WebVTT with the cue id as the cue identifier and a voice tag per speaker:

```
WEBVTT

c0001
00:00:00.080 --> 00:00:01.280
<v S1>Welcome back to the show.
```

## Checking a summary

```sh
uvx podcue check summary.yaml --cues episode.cues.json
```

`podcue check` validates a summary an agent wrote from the cues. The file can be YAML or JSON (by extension: `.json` is JSON, anything else is read as YAML). It prints each problem on stderr and exits 1 if there are any. Otherwise it prints one `ok` line and exits 0.

```yaml
summary: The guest says the city allows too few new homes.   # optional
speakers:                     # human label per speaker id
  S1: Ada Lovelace
  S2: Grace Hopper
paragraphs:
  - text: The guest says the city allows too few new homes near jobs.
    cues: [c0002, c0005]      # cue ids this paragraph is about
    startMs: 41000            # startMs of the earliest cited cue
    endMs: 88000              # endMs of the latest cited cue
    speakers: [S2]            # optional: extra speakers the paragraph refers to
```

`speakers` can also be a list: `[{ id: S1, label: Ada Lovelace }]`. Other keys (for example `version`, `index`, or ids) are ignored.

The check fails when:

- `paragraphs` is missing or empty, a paragraph has no `text`, or `cues` is not a non-empty list of ids.
- A cited cue id is not in the cues file, or a paragraph lists one cue twice.
- `startMs` or `endMs` is missing, or does not equal the start of the earliest cited cue or the end of the latest one (earliest and latest in transcript order).
- A speaker is referenced but has no human label. A paragraph references the speakers of its cited cues plus any it lists in `speakers`. A label is missing if it is absent, blank, `undetermined`, or just the speaker id. Labels such as `quote` or `ending` count as set.
- `speakers` names an id that is not in the cues file, or `summary` is present but empty.

## Weights

On first use podcue downloads two pinned ONNX files into the user cache directory (`~/Library/Caches/podcue` on macOS, `~/.cache/podcue` on Linux). Each file is checked against its pinned size and SHA-256 before use. A file that fails the check is downloaded again.

| File | Source (pinned revision) | Bytes | SHA-256 | License |
| --- | --- | --- | --- | --- |
| `voxceleb_ECAPA512.onnx` | [Wespeaker/wespeaker-voxceleb-ecapa-tdnn512](https://huggingface.co/Wespeaker/wespeaker-voxceleb-ecapa-tdnn512) @ `89426e1` | 24,861,931 | `f901ae3571912bad4e11e157744cb8b7d77a0b46923b83c410bdbb6c77d69736` | CC BY 4.0 |
| `segmentation.onnx` | [onnx-community/pyannote-segmentation-3.0](https://huggingface.co/onnx-community/pyannote-segmentation-3.0) @ `733a93b` | 5,986,908 | `057ee564753071c0b09b5b611648b50ac188d50846bff5f01e9f7bbf1591ea25` | MIT |

- **ECAPA:** WeSpeaker ECAPA-TDNN512 pretrained on VoxCeleb. VoxCeleb is used under CC BY 4.0. The model card has no license field. WeSpeaker says pretrained models follow the license of their training data, so the CC BY 4.0 attribution travels with the file.
- **Segmentation:** pyannote segmentation-3.0 powerset model, MIT license.
- **Speech-to-text:** cactus-needle (pinned to 3.2.0) downloads its own Whistle weights on first use. cactus-needle sends anonymous usage counts by default. Set `NEEDLE_TELEMETRY=0` to turn that off.
- **Moonshine** (`ja`, `ko`, `zh` only): moonshine-voice (pinned to 0.1.5) downloads the model into `<cache>/moonshine` and verifies it against its own CRC32C catalog. The non-English Moonshine models are under the non-commercial [Moonshine Community License](https://www.moonshine.ai/license).

The weights are not part of this package and are not covered by its MIT license. See [NOTICE](NOTICE).

## Development

```sh
uv run pytest                                  # unit tests + smoke test
uv run pytest --ignore tests/test_smoke.py     # unit tests only
uv run --extra moonshine pytest tests/test_smoke.py   # adds the Japanese Moonshine smoke case
```

The smoke test synthesizes a short two-voice clip, runs the full pipeline, and checks a summary against the result. It uses macOS `say` (and expects two speakers) or `espeak-ng` (whose voices usually cluster as one). It skips with the reason when ffmpeg or a synthesizer is missing, or when the weights cannot be downloaded.

## License

MIT License. Copyright 2026 Jianyu Zhao. Model weights are separate; see [Weights](#weights) and [NOTICE](NOTICE).
