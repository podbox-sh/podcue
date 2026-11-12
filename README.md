# podcue

Turn podcast audio into timed, speaker-grouped cues, label the voices, and store summaries whose paragraphs cite those cues. The optional `recap` extra downloads an episode and stitches a narrated clip recap.

podcue is a local-only command line tool. Every command runs in the foreground, prints one JSON line, and exits. It runs no daemon, queue, or server. All state lives in files in one folder per episode. Agents can use it alongside the [PodBox skills](https://github.com/podbox-sh/skills), but they don't have to.

## Requirements

- Python 3.12, 3.13 or 3.14 (onnxruntime 1.31.0 has no wheels for other versions). With [uv](https://docs.astral.sh/uv/) you don't need to install anything else in Python.
- `ffmpeg` on `PATH`.
- macOS or Linux (the episode lock uses `flock`).
- Network access on the first `start`, to download the diarization weights and, with the `asr` extra, the pinned speech model. The first narration downloads Kokoro weights and, if it is not already installed, the pinned spaCy `en_core_web_sm` 3.8.0 wheel.
- The `asr` extra (faster-whisper) for built-in speech-to-text: Whisper `small.en` for `en`, and multilingual Whisper `small` for `de fr es it nl pl`. Without it, pass `--transcript`. `ja ko zh` still need the `moonshine` extra.
- The `recap` extra for `fetch` of a page and for `build`: `kokoro`, `soundfile`, and `yt-dlp`, plus faster-whisper so a recap install can still `start`. Kokoro 0.9.4 narrates on Python 3.12.

## Agent workflow

For an hour-long episode with ~700 cues, this is about 13 tool calls:

1. `podcue start <file or URL>` returns when the cues and groups are ready. Keep `episode.episodeId`.
2. `podcue cues <episodeId> --limit 200` reads the transcript. Pass `--after <last cue id>` for the next page (~4 pages).
3. `podcue cues <episodeId> --group-id <groupId>` gathers evidence for one voice (~3 calls).
4. `podcue label-groups <episodeId> '[{"groupId":…,"label":…}]'` names the voices. Stop when `labeledShare` is about 0.95.
5. `podcue start-summary <episodeId>` returns a `summaryId`.
6. `podcue post-paragraph <episodeId> <summaryId> '[{…},{…}]'` appends all paragraphs in one call. podcue fills the times.
7. `podcue set-summary <episodeId> <summaryId> '{"summary":…}'` writes the summary.
8. `podcue summary <episodeId> <summaryId>` reads the result back.

Each write is validated when it's made, so there is no fix loop. `podcue check` exists for summary files written by hand.

## Recap

`fetch` and `build` need the `recap` extra. `start`, `cues`, and `label-groups` are unchanged; installing the extra is enough to run the whole recap, because the extra is a superset of the base package.

```sh
uvx --python 3.12 --from "podcue[recap]==0.2.4" podcue fetch https://example.com/ep.mp3 --out ./ep
uvx --python 3.12 --from "podcue[recap]==0.2.4" podcue start ./ep/audio.wav
uvx --python 3.12 --from "podcue[recap]==0.2.4" podcue cues ep_3f9a1c2b7d4e5f60 --limit 200
uvx --python 3.12 --from "podcue[recap]==0.2.4" podcue label-groups ep_3f9a1c2b7d4e5f60 '[{"groupId":"gc0001","label":"Ada Lovelace"}]'
uvx --python 3.12 --from "podcue[recap]==0.2.4" podcue build ep_3f9a1c2b7d4e5f60 script.json --out recap.mp3
```

Pass `--python 3.12`. Kokoro 0.9.4 declares `Requires-Python: <3.13`. A direct media URL downloads without yt-dlp; a page URL needs it, from the same extra. Plain `uvx podcue` does not install the extra, so transcription stays light.

## Output

- stdout gets exactly one JSON line per command. Progress goes to stderr.
- A non-zero exit always prints `{"kind":"error","message":…}`. Some errors add a `reason` (`not-found`, `not-audio`, `unknown-language`, `running`, `limit`) or a `problems` list (`check`).
- An id the episode doesn't have prints `{"state":"unknown","episodeId":…}` and exits 0. A `groupId`, `cueId` or `summaryId` field names the unknown id.
- A command that needs cues the episode doesn't have yet (the start is still running, or it failed) prints `{"state":"not-ready","episodeId":…,"on":"cues"}` and exits 0. `podcue episode` shows the reason.
- `--help` and `--version` print plain text.

The episode root is `--root DIR` (accepted by every command), else `$PODCUE_HOME`, else `~/podcue`.

The examples below come from a short two-voice clip. The ids, paths and cosines are illustrative.

## Commands

### `podcue start <file or URL> [--language CODE] [--transcript FILE]`

```sh
uvx --from 'podcue[asr]' podcue start ep12.mp3
uvx --from 'podcue[asr]' podcue start https://example.com/ep12.mp3
uvx --from 'podcue[moonshine]' podcue start ep12.m4a --language ja
uvx podcue start ep12.mp3 --transcript transcript.json
```

```json
{"kind": "started", "created": true, "episode": {"state": "present", "episodeId": "ep_3f9a1c2b7d4e5f60", "source": "/Users/me/ep12.mp3", "durationMs": 17664, "language": "en", "createdAt": "2026-11-05T10:12:03Z", "dir": "/Users/me/podcue/episodes/ep_3f9a1c2b7d4e5f60", "artifacts": [{"artifact": "transcript", "state": "ready", "progress": 100, "path": "/Users/me/podcue/episodes/ep_3f9a1c2b7d4e5f60/transcript.vtt"}, {"artifact": "cues", "state": "ready", "progress": 100, "path": "/Users/me/podcue/episodes/ep_3f9a1c2b7d4e5f60/cues.json"}], "groups": [{"groupId": "gc0001", "speakingMs": 9760, "cueCount": 4, "nearestGroupId": "gc0003", "nearestCosine": 0.1873, "label": null}, {"groupId": "gc0003", "speakingMs": 5840, "cueCount": 3, "nearestGroupId": "gc0001", "nearestCosine": 0.1873, "label": null}], "labeledShare": 0.0, "undeterminedShare": 0.0, "summaries": []}}
```

- The source is any file ffmpeg can decode, or a direct `http(s)` link to one. podcue doesn't resolve web pages to audio. A local path is checked before any model loads or any download starts. A local file is stored in `episode.json` as its absolute path. A later `start` of the same audio, after the file has moved, updates that path and does not reload the models.
- The episode id is `ep_` plus the first 16 hex characters of the sha256 of the converted 16 kHz mono WAV. The same audio always maps to the same folder.
- Idempotent: when the folder already has `cues.json`, `start` returns it with `"created": false` and loads no model. `--language` is then ignored, and the episode keeps its first language.
- `--fresh` recomputes that episode's transcript, cues, and automatic groups, and replaces those files in place. It does not delete other episodes, `labels.json`, summaries, or the shared model-weight cache. `--root <new dir>` is a separate tree: nothing in the old root is read, which is the other way to get a cold run.
- A start that failed or was interrupted is rerun by the next `start` of the same audio.
- A second `start` of the same episode while one is running exits 1 with `"reason": "running"`.
- `--language`, for built-in ASR, is `en` (Whisper `small.en`), `de fr es it nl pl` (Whisper `small`), from the `asr` extra, or `ja ko zh` (Moonshine, from the `moonshine` extra). The default is `en`. **`ja`, `ko` and `zh` are experimental.** In testing, Moonshine dropped the opening greeting 「こんにちは。」 of a Japanese clip and returned the rest as one cue without word times.
- `--transcript FILE` skips built-in ASR. Diarization and cue building still run. The schema is below.

#### `--transcript` file

```json
{
  "language": "en",
  "segments": [
    {
      "startMs": 80,
      "endMs": 2400,
      "text": "Welcome back to the show.",
      "words": [
        {"startMs": 80, "endMs": 480, "text": "Welcome"},
        {"startMs": 500, "endMs": 900, "text": "back"},
        {"startMs": 920, "endMs": 1100, "text": "to"},
        {"startMs": 1120, "endMs": 1400, "text": "the"},
        {"startMs": 1420, "endMs": 2400, "text": "show."}
      ]
    }
  ]
}
```

`segments` is required and non-empty. Each segment has integer `startMs`, `endMs` (greater than `startMs`), and non-empty `text`. `words` is optional. A word has the same three fields and must sit inside its segment, in time order. Segments are in time order. With no `words`, the text is split on whitespace and spread across the segment by character length. Cue text then comes from those words. `language` is required in the file unless you pass `--language`. Any code is accepted, not only the built-in list, and it is stored on the episode (compared case-insensitively: `EN` and `en` match, `en` and `en-US` do not). If the flag and the file both set a language and they differ, start exits 1 with `{"kind":"error"}`. That language is what closes a cue: `ja`, `zh`, and `ko` also close on `。！？`; every other code closes on `.`, `!`, and `?`. A word that starts outside a detected speech region is left out.

How it works: podcue converts the audio to 16 kHz mono WAV and diarizes the whole episode (pyannote segmentation plus ECAPA embeddings). A small embedding cluster joins a large one only when its centroid is within the same distance the diarizer already uses to link voices; a farther cluster stays its own speaker. It joins each speaker's turns into regions, transcribes the regions in slices of up to 28 seconds (or places your transcript words into those slices), and groups the words into cues. A cue ends at sentence punctuation for the episode language, after 12 seconds, or where a speaker region ends. **Speech the diarizer misses (outside the detected speech regions) is not transcribed,** and a transcript word that starts there is left out. Each cue gets the group whose diarization turns overlap it most, or `ungrouped` when none does. A cue under 1.5 seconds that sits between two cues of one speaker, while another speaker's turn is also inside it, is moved to the `uncertain` group instead of joining that neighbor. Built-in ASR is not imported for `--transcript`, nor for `fetch`, `cues`, `label-groups`, `build`, or `--help`. No Hugging Face token is required: the model files are public pinned commits, and `start` quiets the hub's missing-token warning.

### `podcue episode <episodeId>`

This has the same shape as `start`'s `episode`. While a start is running, both artifacts show `"state": "running"` with the stage (`weights`, `diarize`, `transcribe`, `write`) and progress:

```json
{"state": "present", "episodeId": "ep_3f9a1c2b7d4e5f60", "source": "/Users/me/ep12.mp3", "durationMs": 17664, "language": "en", "createdAt": "2026-11-05T10:12:03Z", "dir": "/Users/me/podcue/episodes/ep_3f9a1c2b7d4e5f60", "artifacts": [{"artifact": "transcript", "state": "running", "stage": "transcribe", "progress": 40}, {"artifact": "cues", "state": "running", "stage": "transcribe", "progress": 40}], "summaries": []}
```

A failed start shows `{"state": "failed", "message": …}`. A start that was killed shows the message `start was interrupted; run start again`. `groups`, `labeledShare`, and `undeterminedShare` appear once the cues are ready. `createdAt` is the UTC time of the start that wrote the cues. A later start of audio that is already ready leaves that stamp alone. A retry after a failed or interrupted start stamps the retry, not the earlier attempt.

### `podcue cues <episodeId> [--limit N] [--after CUE_ID] [--group-id GROUP_ID] [--hosts]`

```json
{"state": "present", "episodeId": "ep_3f9a1c2b7d4e5f60", "cues": [{"id": "c0003", "startMs": 5821, "endMs": 7021, "text": "Thanks for having me.", "groupId": "gc0003", "speaker": "gc0003", "nearestGroupId": "gc0001", "nearestCosine": 0.1873}], "hostNames": ["Ada Lovelace", "Grace Hopper"]}
```

- `--limit` is 1 through 200. The default is 50. A page shorter than the limit is the last one.
- `--after` starts after that cue. With `--group-id`, the cue must belong to that group.
- `--group-id` returns only that group's cues. `ungrouped` is always a valid id.
- `--hosts` adds `hostNames`: the person names in `labels.json`, leaving out `quote`, `ending` and `undetermined`.

### `podcue label-groups <episodeId> '<json>'`

```sh
podcue label-groups ep_3f9a1c2b7d4e5f60 '[{"groupId":"gc0001","label":"Ada Lovelace"},{"groupId":"gc0003","label":"Grace Hopper"}]'
```

```json
{"state": "present", "episodeId": "ep_3f9a1c2b7d4e5f60", "labels": [{"groupId": "gc0001", "label": "Ada Lovelace"}, {"groupId": "gc0003", "label": "Grace Hopper"}], "labeledShare": 1.0, "undeterminedShare": 0.0}
```

- The body is a JSON array. Each entry is `{groupId, label}` or `{fromCue, toCue, label}`. A label is a person's name, `undetermined`, `quote` (a played clip) or `ending` (a sign-off or song). Send only the groups you are changing. They are merged into `labels.json`, and the last entry for a group wins.
- A range labels every group whose cues all lie inside `fromCue` through `toCue`, inclusive, in cue order. A group with any cue outside that window is left as it is, which is how one entry can name a run of tiny quote groups without painting the host who speaks before and after them. `uncertain` and `mixed` are not selected by a range. An unknown cue id writes nothing. If `fromCue` is after `toCue`, the command exits 1 and writes nothing.
- After a successful `label-groups`, each cue's `speaker` is its group's label. `groupId` stays the group id. A cue whose `speaker` is `uncertain` or `mixed` keeps that, even when the surrounding group is labeled. Before any label, `speaker` is the group id.
- If any `groupId` is unknown, nothing is written.
- `labeledShare` is the share of cue time whose group has a person's name. `undetermined`, `quote`, and `ending` do not count. It is the sum of `endMs - startMs` over those cues, divided by the same sum over all cues, rounded to 4 places. It's the same measure as `groups[].speakingMs`. Stop labeling at about 0.95.
- `undeterminedShare` is that same share for groups labeled `undetermined` only. `quote` and `ending` are in neither field.

### `podcue start-summary <episodeId>`

```json
{"state": "present", "episodeId": "ep_3f9a1c2b7d4e5f60", "version": 1, "summaryId": "sm_8c41d07e2a9b3f15", "summary": "", "paragraphs": []}
```

This creates `summaries/<summaryId>.yaml`. Each call starts a new summary.

### `podcue post-paragraph <episodeId> <summaryId> '<json>'`

```sh
podcue post-paragraph ep_3f9a1c2b7d4e5f60 sm_8c41d07e2a9b3f15 '[
  {"text": "Ada asks why cities build so few homes near jobs.", "cues": ["c0002"], "part": "discussion"},
  {"text": "Grace blames zoning.", "cues": ["c0005", "c0004"], "part": "claim"}]'
```

```json
{"state": "present", "episodeId": "ep_3f9a1c2b7d4e5f60", "version": 1, "summaryId": "sm_8c41d07e2a9b3f15", "summary": "", "paragraphs": [{"index": 1, "part": "discussion", "text": "Ada asks why cities build so few homes near jobs.", "cues": ["c0002"], "startMs": 1680, "endMs": 5600}, {"index": 2, "part": "claim", "text": "Grace blames zoning.", "cues": ["c0004", "c0005"], "startMs": 7341, "endMs": 12221}]}
```

- The body is one paragraph `{text, cues, part?}`, or a JSON array of them, appended in order. Each paragraph is validated before any is written.
- `cues` are stored in transcript order. `startMs` is the start of the earliest cited cue and `endMs` is the end of the latest. Leave both out. If you send one, it must match.
- `part` is optional: one of `news`, `explanation`, `advice`, `claim`, `discussion`, `story`, `items`.
- Errors: an unknown cue, a cue listed twice, a time that doesn't match, an unknown `part`, an `index`, or any other key. In an array, the message starts with `paragraph N:`.

### `podcue put-paragraph <episodeId> <summaryId> '<json>'`

This replaces an existing paragraph. It takes the same fields as `post-paragraph` plus `index`:

```sh
podcue put-paragraph ep_3f9a1c2b7d4e5f60 sm_8c41d07e2a9b3f15 '{"index": 2, "text": "Grace blames zoning: most central land allows only single-family houses.", "cues": ["c0004", "c0005"], "part": "claim"}'
```

It prints the whole summary, like `post-paragraph`. An index that doesn't exist exits 1 with `putParagraph index does not exist`.

### `podcue set-summary <episodeId> <summaryId> '{"summary": "…"}'`

```json
{"state": "present", "episodeId": "ep_3f9a1c2b7d4e5f60", "version": 1, "summaryId": "sm_8c41d07e2a9b3f15", "summary": "Grace Hopper says zoning keeps cities from building homes near jobs.", "paragraphs": [{"index": 1, "part": "discussion", "text": "Ada asks why cities build so few homes near jobs.", "cues": ["c0002"], "startMs": 1680, "endMs": 5600}, {"index": 2, "part": "claim", "text": "Grace blames zoning: most central land allows only single-family houses.", "cues": ["c0004", "c0005"], "startMs": 7341, "endMs": 12221}]}
```

It rejects an empty summary, and a summary that has no paragraphs yet.

### `podcue summary <episodeId> <summaryId>`

This prints the same object as `set-summary`, read back from the file.

### `podcue check <episodeId> <summaryId or path>`

`check` strictly validates a summary file against the episode's `cues.json` and `labels.json`. The file can be one the commands wrote (pass its id) or one written by hand (pass a `.yaml` or `.json` path). It never changes the file.

```json
{"state": "valid", "episodeId": "ep_3f9a1c2b7d4e5f60", "path": "/Users/me/podcue/episodes/ep_3f9a1c2b7d4e5f60/summaries/sm_8c41d07e2a9b3f15.yaml", "paragraphs": 2, "summaryId": "sm_8c41d07e2a9b3f15", "warnings": []}
```

```json
{"kind": "error", "message": "2 problem(s): paragraph 1: startMs 1 does not match c0004 startMs 7341; paragraph 1: endMs 2 does not match c0004 endMs 8781", "problems": ["paragraph 1: startMs 1 does not match c0004 startMs 7341", "paragraph 1: endMs 2 does not match c0004 endMs 8781"], "warnings": []}
```

It fails, listing every problem, when:

- The file isn't a mapping of exactly `version` (must be `1`), `episodeId` (must be this episode), `summary` (a non-empty string), `paragraphs` (a non-empty list), and an optional `summaryId` (`sm_` + 16 hex).
- A paragraph has keys other than `index`, `part`, `text`, `cues`, `startMs`, `endMs`, or `index` isn't its 1-based position, or `text` is empty, or `part` isn't in the list above.
- `cues` is empty, names a cue that doesn't exist, lists a cue twice, or isn't in transcript order.
- `startMs` or `endMs` is missing, or doesn't equal the earliest cited cue's start or the latest cited cue's end.
- A group cited through a paragraph's cues has no entry in `labels.json`. `ungrouped` cues need no label, and `quote` and `ending` count as labels. A cited group labeled `undetermined` adds a `warnings` entry but doesn't fail the check, because labeling stops at about 95% of spoken time on purpose.

### `podcue fetch <URL> [--out DIR]`

```sh
uvx --python 3.12 --from "podcue[recap]==0.2.4" podcue fetch https://cdn.example.com/ep.mp3 --out ./ep
uvx --python 3.12 --from "podcue[recap]==0.2.4" podcue fetch https://www.youtube.com/watch?v=VIDEO --out ./ep
```

```json
{"kind": "fetched", "url": "https://cdn.example.com/ep.mp3", "path": "/Users/me/ep/audio.wav", "durationMs": 3600000, "via": "direct"}
```

- Writes `<DIR>/audio.wav` (default directory: `.`): 16-bit mono 16 kHz PCM, the format `start` converts to. Pass that path to `podcue start`.
- `via` is `direct` when the URL path ends in a media suffix (`.mp3`, `.m4a`, `.wav`, `.mp4`, and the other audio and video suffixes), query string ignored. `via` is `yt-dlp` for any other `http(s)` URL. A non-URL exits 1.
- The direct path uses the standard library. The page path imports yt-dlp and exits 1, naming the `recap` extra, when that package is missing.
- ffmpeg converts the download. An empty or undecodable file exits 1 with `"reason": "not-audio"`.

### `podcue build <episodeId> <script.json> [--out FILE]`

```sh
uvx --python 3.12 --from "podcue[recap]==0.2.4" podcue build ep_3f9a1c2b7d4e5f60 script.json --out recap.mp3
```

```json
{"kind": "built", "episodeId": "ep_3f9a1c2b7d4e5f60", "output": "/Users/me/recap.mp3", "report": "/Users/me/report.md", "durationMs": 312000, "loudness": {"lufs": -16.0, "truePeak": -1.5}, "flags": [], "segments": [{"index": 1, "type": "narration", "text": "Intro.", "voice": "am_michael", "lufsIn": -23.0, "lufsOut": -16.0, "flags": []}]}
```

Takes the episode `start` already transcribed. Clips are cut from `episode.json`'s `source`, which has to be a local file. An episode started from a URL has no audio left to cut; `fetch`, then `start` on `audio.wav`.

The output is 128 kbps MP3, or 24 kHz mono WAV when the path ends in `.wav`. `--out` wins. Otherwise the script's `output` is used, else `produced_summary.mp3`, resolved from the working directory. `report.md` is written next to that file. Flags are listed in the JSON and in the report. A flagged clip still writes the file and exits 0; fix the script and build again. Narration and clip wavs are reused from `<episode>/recap-cache/` when the segment is unchanged.

What it does:

1. Snap each clip to cue word times (below), with a short fade.
2. Narrate each narration segment with Kokoro at 24 kHz on CPU. The default voice is `am_michael`.
3. Loudnorm each clip to -16 LUFS, true peak -1.5 dBTP. Narration is gained to that same target when Kokoro writes it, then limited so the peak stays at or under -1.5 dBTP. Join the segments with 0.4 s of silence. The mix is loudnormed only when it is already within 3 dB of -16 LUFS; a larger correction would be the late boost that lifts a quiet narration, so that mix is only peak-limited.
4. Flag a clip under 4 seconds or over 45 seconds, a clip with no overlapping cue, a clip with a second speaker, and a clip that contains a cue whose voice is `uncertain`, `mixed`, or `undetermined`. A second speaker counts when another voice covers at least 1 second and 8% of the clip. The voice is the cue's `speaker`: after `label-groups` that is the group's label, so two groups labeled the same person are one speaker. A cue that still has its group id as `speaker` uses the label from `labels.json` when there is one. An `uncertain`, `mixed`, or `undetermined` cue is flagged even when it is shorter than that 1 second.

Kokoro is imported only for a narration segment. On Python 3.13 or 3.14, or without the extra, that import exits 1 and names the 3.12 install command. The first narration downloads the Kokoro-82M weights from Hugging Face. English narration also needs spaCy's `en_core_web_sm` 3.8.0. That model is not on PyPI, and a direct wheel URL cannot be declared in the package metadata, so the first narration downloads the pinned wheel (sha256 `1932429db727d4bff3deed6b34cfc05df17794f4a52eeb26cf8928f7c1a0fb85`), prints one line on stderr, and installs it. A later narration in the same environment does not download it again.

#### `script.json`

An object, or a bare array of segments. Unknown keys are ignored.

```json
{
  "voice": "am_michael",
  "output": "produced_summary.mp3",
  "segments": [
    {"type": "narration", "text": "Intro."},
    {"type": "clip", "start": "40:43.5", "end": "40:58.0", "speaker": "the guest"},
    {"type": "narration", "text": "Bridge.", "voice": "american-female-heart"}
  ]
}
```

| Field | Where | Meaning |
| --- | --- | --- |
| `voice` | script or narration | Kokoro voice id. The default, when `voice` and `voice_preset` are omitted, is `am_michael`. `american-male-puck` maps to `am_puck` and `american-female-heart` maps to `af_heart`. Any other string is passed through as a Kokoro id. |
| `voice_preset` | script or narration | Same as `voice`, used when `voice` is absent. A narration's own `voice` or `voice_preset` wins over the script's. |
| `output` | script | File name used when `--out` is omitted. |
| `segments` | script | Non-empty array, in playback order. A bare array is the whole script. |
| `type` | segment | `narration` or `clip`. |
| `text` | narration | What Kokoro reads. A non-empty string. |
| `start`, `end` | clip | Seconds (`90.5` or `"90.5"`) or `[h:]mm:ss[.s]` (`"40:43.5"`, `"1:02:03"`). `end` must be after `start`. |
| `speaker` | clip | Name written into the report. Optional. The default is `?`. |

Cues are at most 12 seconds, so a clip of a few seconds spans consecutive cues. `cues.json` has no per-word timestamps. Each cue's words are spread across that cue by character length, and each requested edge snaps to the nearest of those word times, then pads at most 0.25 s before the first word and 0.35 s after the last, stopping at the neighbouring word. The clip text in the report is those words. A clip that should be one speaker needs consecutive cues whose `speaker` is that person.

### `podcue fetch-weights`

This is optional. It downloads the two pinned diarization weight files now instead of on the first `start`.

```json
{"kind": "ready", "weights": [{"file": "voxceleb_ECAPA512.onnx", "path": "/Users/me/Library/Caches/podcue/weights/voxceleb_ECAPA512.onnx", "sha256": "f901ae3571912bad4e11e157744cb8b7d77a0b46923b83c410bdbb6c77d69736", "license": "CC BY 4.0"}, {"file": "segmentation.onnx", "path": "/Users/me/Library/Caches/podcue/weights/segmentation.onnx", "sha256": "057ee564753071c0b09b5b611648b50ac188d50846bff5f01e9f7bbf1591ea25", "license": "MIT"}]}
```

## The episode folder

```
<root>/episodes/ep_3f9a1c2b7d4e5f60/
  episode.json        written when a start begins
  transcript.vtt      \
  groups.json          } written when the start succeeds; cues.json last, so it marks the episode ready
  cues.json           /
  labels.json         written by label-groups
  summaries/
    sm_8c41d07e2a9b3f15.yaml
  recap-cache/        narration and clip wavs reused by a later build of the same segment
  status.json         only while a start runs; removed when it ends
  failed.json         only after a failed start; removed by the next start
  start.lock          holds the running start's pid (empty otherwise)
  write.lock          serializes label and summary writes
```

Every file is written to a temp file and renamed into place, so a reader never sees half a file. The only coordination is the two lock files. `start.lock` is held with `flock` and holds the pid, so a start that crashed leaves a stale lock, which the OS has already released and the next start replaces.

| File | Schema |
| --- | --- |
| `episode.json` | `{episodeId, source, durationMs, language, createdAt}`. `source` is the absolute path of a local file, or the URL as given. `createdAt` is ISO 8601 UTC (`…Z`) of the start that wrote the cues. |
| `cues.json` | `[{id, startMs, endMs, text, groupId, speaker, nearestGroupId, nearestCosine}]` in time order. `id` is `c0001`, `c0002`, …. Times are ms from the start of the audio. `groupId` is `g` + the group's first cue id (for example `gc0001`), `ungrouped`, or `uncertain`. Before `label-groups`, `speaker` is that same id. After `label-groups`, `speaker` is the group's label (`Ada`, `quote`, `undetermined`, …). A cue pulled off a neighbor at a turn boundary keeps `speaker` `uncertain` (or `mixed` if it was marked that way). `nearestGroupId`/`nearestCosine` name the other group whose ECAPA centroid is most similar to this cue's group. Both are `null` for `ungrouped`, `uncertain`, and when there is one group. A high cosine hints that two groups are one person. |
| `groups.json` | `[{groupId, speakingMs, cueCount, nearestGroupId, nearestCosine}]` in order of first cue. `speakingMs` is the sum of the group's cue durations. |
| `transcript.vtt` | WebVTT. The cue identifier is the cue id, and grouped cues carry a voice tag: `c0001` / `00:00:00.080 --> 00:00:01.280` / `<v gc0001>Welcome back to the show.` |
| `labels.json` | `{"groups": [{groupId, label}]}` sorted by `groupId`. |
| `summaries/sm_*.yaml` | `version: 1`, `episodeId`, `summaryId`, `summary`, and `paragraphs: [{index, part?, text, cues, startMs, endMs}]`. |
| `status.json` | `{pid, stage, progress}`. `stage` is `weights`, `diarize`, `transcribe` or `write`, and `progress` runs 0 to 99. |
| `failed.json` | `{message, stage, at}`. |
| `recap-cache/` | Wav segments from `build`. Deleting it only makes the next build render those segments again. |

A summary file:

```yaml
version: 1
episodeId: ep_3f9a1c2b7d4e5f60
summaryId: sm_8c41d07e2a9b3f15
summary: Grace Hopper says zoning keeps cities from building homes near jobs.
paragraphs:
- index: 1
  part: discussion
  text: Ada asks why cities build so few homes near jobs.
  cues:
  - c0002
  startMs: 1680
  endMs: 5600
- index: 2
  part: claim
  text: 'Grace blames zoning: most central land allows only single-family houses.'
  cues:
  - c0004
  - c0005
  startMs: 7341
  endMs: 12221
```

## Privacy and network

podcue runs locally. It uses the network only to download the input when you pass a URL to `start` or `fetch`, to download diarization weights on first use, and, with the recap extra, to download Kokoro weights the first time a narration is built. `fetch` of a page also uses the network through yt-dlp. Each direct download has a 60-second connect/read timeout, so a stalled connection fails instead of hanging.

Podcue does not need `HF_TOKEN`. Every command calls the same `quiet_hub` helper before it does any work, narration calls it again immediately before Kokoro imports `huggingface_hub`, and built-in Whisper calls it before faster-whisper downloads weights. That sets `HF_HUB_DISABLE_TELEMETRY=1` unless you have set it, sets the hub log level to error unless you already set `HF_HUB_VERBOSITY`, and ignores a missing-token warning. Whisper and the diarization weights are public commits, pinned by revision, and the Whisper `model.bin` is checked against a SHA-256.

## Weights and licenses

On first use podcue downloads two pinned ONNX files into the user cache directory (`~/Library/Caches/podcue` on macOS, `~/.cache/podcue` on Linux). Each file is checked against its pinned size and SHA-256 before use. A file that fails the check is downloaded again.

| File | Source (pinned revision) | Bytes | SHA-256 | License |
| --- | --- | --- | --- | --- |
| `voxceleb_ECAPA512.onnx` | [Wespeaker/wespeaker-voxceleb-ecapa-tdnn512](https://huggingface.co/Wespeaker/wespeaker-voxceleb-ecapa-tdnn512) @ `89426e1` | 24,861,931 | `f901ae3571912bad4e11e157744cb8b7d77a0b46923b83c410bdbb6c77d69736` | CC BY 4.0 |
| `segmentation.onnx` | [onnx-community/pyannote-segmentation-3.0](https://huggingface.co/onnx-community/pyannote-segmentation-3.0) @ `733a93b` | 5,986,908 | `057ee564753071c0b09b5b611648b50ac188d50846bff5f01e9f7bbf1591ea25` | MIT |

- **ECAPA:** an ECAPA-TDNN512 trained with [WeSpeaker](https://github.com/wenet-e2e/wespeaker) (Apache-2.0 code) on VoxCeleb. Credit for the CC BY 4.0 weights goes to VoxCeleb (Nagrani, Chung, Xie, Zisserman), https://www.robots.ox.ac.uk/~vgg/data/voxceleb/. The model card has no license field. WeSpeaker says pretrained models follow the license of their training data, so the CC BY 4.0 attribution travels with the file.
- **Segmentation:** pyannote segmentation-3.0 powerset model, MIT license.
- **Speech-to-text (`en de fr es it nl pl`):** [faster-whisper](https://github.com/SYSTRAN/faster-whisper) 1.2.1 with [CTranslate2](https://github.com/OpenNMT/CTranslate2) 4.9.0, the `asr` extra. Both are MIT. English uses `Systran/faster-whisper-small.en` at `d1d751a5f8271d482d14ca55d9e2deeebbae577f` (`model.bin` 483,545,366 bytes, sha256 `62b2a45b05ee59acb4a5341b33ee35e041395d378d418a18acfe4c9e768ee37a`). German, French, Spanish, Italian, Dutch and Polish use `Systran/faster-whisper-small` at `536b0662742c02347bc0e980a01041f333bce120` (`model.bin` 483,546,902 bytes, sha256 `3e305921506d8872816023e4c273e75d2419fb89b24da97b4fe7bce14170d671`). Those are OpenAI Whisper small weights converted by Systran, MIT, run on CPU at int8. `podcue[recap]` includes this extra. `ja ko zh` stay on Moonshine.
- **Moonshine (`ja ko zh` only):** moonshine-voice 0.1.5, from the `moonshine` extra, downloads the model into `<cache>/moonshine` and verifies it against its own CRC32C catalog. The non-English Moonshine models are under the non-commercial [Moonshine Community License](https://www.moonshine.ai/license). Those terms apply only when you run with `--language ja|ko|zh`. No other language is routed to Moonshine.

The weights are not part of this package and are not covered by its MIT license. See [NOTICE](NOTICE).

### Recap extra

Installed only with `podcue[recap]`. Plain `uvx podcue` does not install these.

| Piece | Version | License |
| --- | --- | --- |
| `kokoro` | 0.9.4 | Apache-2.0. Requires Python 3.10 or 3.12 (`Requires-Python: <3.13`). |
| Kokoro-82M weights | `hexgrad/Kokoro-82M` | Apache-2.0. Downloaded on first narration, not shipped here. A few hours of the training audio are CC BY (Koniwa, SIWIS); see [NOTICE](NOTICE). |
| `soundfile` | 0.14.0 | BSD-3-Clause. Its wheels ship libsndfile, LGPL-2.1-or-later, loaded as a shared library. |
| `yt-dlp` | 2026.10.26 | Unlicense. The PyPI wheel is Unlicense. The PyInstaller executables are not what this extra installs. |

`kokoro` depends on `misaki[en]` (Apache-2.0). That extra also installs `phonemizer-fork` (GPL-3.0-or-later) and `espeakng-loader`, which does not declare a license and bundles the espeak-ng shared library (GPL-3.0-or-later). English narration tries espeak only as a fallback for out-of-dictionary words and continues without it. Commercial use of those GPL pieces is allowed; distributing a combined work that links them is under GPL-3. podcue does not vendor them. `num2words`, also pulled in for English, is LGPL. `torch` is BSD-3-Clause and `transformers` is Apache-2.0.

The `moonshine` extra is separate and unchanged: its non-English models are non-commercial. The recap extra does not use Moonshine.

## Development

```sh
uv run pytest                                  # unit tests + smoke test
uv run pytest --ignore tests/test_smoke.py --ignore tests/test_start_en.py
uv run --extra moonshine pytest tests/test_smoke.py
PODCUE_LIVE_EN=1 uv run --extra asr pytest tests/test_start_en.py
```

Unit tests do not install the `asr` or `recap` extras. They stub Kokoro, the downloader, and diarization. `build` and `fetch` still need ffmpeg only when those stubs are not in place. `tests/test_start_en.py` runs a real `start` on espeak-ng speech, once with the `asr` extra and once with `--transcript`. It skips unless `PODCUE_LIVE_EN=1`.

The smoke test synthesizes a short two-voice clip and runs the whole flow through the real CLI and models: start, cues, label-groups, start-summary, post-paragraph, set-summary, summary, check. It uses macOS `say` (and expects two groups) or `espeak-ng` (whose voices usually cluster as one). It skips with the reason when ffmpeg or a synthesizer is missing.

## Releasing

Bump `version` in `pyproject.toml` and `__version__` in `podcue/__init__.py` to the same value, then run `uv lock` so the lockfile matches. Merge that to `main`.

Create a GitHub release tagged `v` plus that version, for example `v0.2.1`. Publishing the release runs [publish.yml](.github/workflows/publish.yml). It builds the sdist and wheel, checks that the built version matches the tag with a leading `v` removed, and publishes to PyPI with trusted publishing. The workflow stores no PyPI token. Running the workflow by hand (`workflow_dispatch`) publishes the selected commit the same way, without the tag check.

0.2.4 is the package version on this branch. It is not tagged and not published. See [CHANGELOG.md](CHANGELOG.md). Built-in ASR for `en de fr es it nl pl` is Whisper small (faster-whisper). `ja`, `ko`, and `zh` stay on Moonshine.

0.2.3, which is released, writes each cue's `speaker` from its group label and lets `label-groups` name every group that sits inside a cue range.

0.2.2 flags clip length only under 4 seconds or over 45 seconds. A short cue at a speaker boundary can be `uncertain`. `labeledShare` counts person names only, and `undeterminedShare` is reported beside it. `createdAt` is the UTC time of the start that wrote the cues.

0.2.1 moves built-in speech-to-text to the `asr` extra (`cactus-compute` 4.1.0, whose engine download is pinned to a commit) and adds `podcue start --transcript` for a model's own timed words. The transcript object is `{"language", "segments": [{startMs, endMs, text, words?}]}`.

## License

MIT License. Copyright 2026 Jianyu Zhao. Model weights are separate; see [Weights and licenses](#weights-and-licenses) and [NOTICE](NOTICE).
