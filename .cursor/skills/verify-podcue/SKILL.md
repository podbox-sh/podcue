---
name: verify-podcue
description: "Drive the podcue CLI the way a user does: install the local wheel with uvx, then start, cues, label-groups, and build, including start --fresh. Use to prove a podcue change end to end, before a release, or when narration, cues, or labels need a real Whisper and Kokoro run."
---

# Verify podcue

podcue is a local CLI. There is no server to keep running. One verification run builds the wheel, installs it with `uvx` into its own episode `--root`, drives the commands below, and writes proof outside that root.

The speech file is `tests/fixtures/gettysburg-75s.mp3`, a 75-second public-domain excerpt of the LibriVox Gettysburg Address (John Greenman, Internet Archive item `gettysburg_johng_librivox`).

The CI job `stability` runs this same driver. Do not add a second copy of these steps.

## Launch

From the repo root, with `uv` on `PATH`:

```sh
python3 .cursor/skills/verify-podcue/drive.py
```

That builds `dist/podcue-*-py3-none-any.whl` and drives:

```sh
uvx --python 3.12 --from 'dist/podcue-<version>-py3-none-any.whl[recap]' podcue --root <isolated-dir> ...
```

Ready means `podcue --version` prints `podcue <version>` and the driver has created an empty `--root` (`mkdtemp`, prefix `podcue-verify-`). The root is not `~/podcue`. Two runs do not share a root.

Set `PODCUE` only when the wheel is already built and you must pass the exact `uvx` command. Otherwise leave it unset so launch and CI stay the same.

## Doctor

The driver runs this before any episode command. It does not download models and does not write the episode root.

- `uv`, `ffmpeg`, and `ffprobe` are on `PATH`.
- `podcue --version` exits 0 and starts with `podcue `.
- Cache directories are listed, present or absent: `$HF_HOME` or `~/.cache/huggingface`, `$XDG_CACHE_HOME/podcue` or `~/.cache/podcue`, and `~/Library/Caches/podcue`.

The report is `verify-podcue-evidence/doctor.txt`. A missing model cache is reported and is not a failure; `start` and `build` download on first use. A missing `uv` or `ffmpeg` stops the run.

## Drive

Order, against the isolated `--root` and the committed 75-second clip:

1. `podcue --root <root> start tests/fixtures/gettysburg-75s.mp3 --language en`
2. `podcue --root <root> cues <episodeId> --limit 200`
3. `podcue --root <root> label-groups <episodeId> '[{"groupId":"<groupId>","label":"John Greenman"}]'`
4. `podcue --root <root> build <episodeId> <root>/script.json --out <root>/recap.mp3`
5. `podcue --root <root> start tests/fixtures/gettysburg-75s.mp3 --language en --fresh`

`script.json` has two narration lines, `About 400,000 people know this address.` and `The excerpt is 2.4% of a full hour.`, and one clip of about 12 seconds taken from the first cue. The group id comes from the `start` JSON. `uncertain`, `mixed`, and `ungrouped` are skipped when another group exists.

Each command must exit 0 and print one JSON object that is not `{"kind":"error",...}`. After each command, no process holds a `*.lock` file. The files may remain. `start` must report `created: true` and `asr.engine` `faster-whisper`. `cues` and `label-groups` must report `state: present`. `build` must report `kind: built`.

## Evidence

Proof is written to `verify-podcue-evidence/` in the repo (override with `VERIFY_PODCUE_EVIDENCE`). That directory is not inside `--root`.

- `launch.txt`, `doctor.txt`, `root.txt`
- `start.json`, `cues.json`, `label-groups.json`, `build.json`, `start-fresh.json`, plus each command's stdout and stderr
- `script.json`, `recap.mp3`, `report.md`
- `source-duration.txt` and `ffprobe-duration.txt` (recap duration must be between 8 and 90 seconds)
- `lufs.txt` (each narration `lufsOut` within 2 LU of -16)

The clip and the recap are real files. ffprobe reads the recap. LUFS is the `lufsOut` field on the narration segments in the build JSON, which is the loudness ffmpeg measured after the narration was normalized.

## Cleanup

The driver deletes only the `--root` it created (`shutil.rmtree` on that path). It does not delete `verify-podcue-evidence/`. It does not kill processes by name. There is no long-lived server. After the process exits, `root.txt` names a directory that no longer exists, and the evidence files are still in the evidence directory.

## Helpers

`.cursor/skills/verify-podcue/drive.py` is the only driver. CI job `stability` runs `python3 .cursor/skills/verify-podcue/drive.py`. Feature recipes are under `features/`.
