# podcue verification map

This directory is the maintained source for verifying the podcue CLI. Read the index, then use the matching feature file. The CI job `stability` runs `.cursor/skills/verify-podcue/drive.py`, which drives start, cues, label-groups, build, and `start --fresh` in that order.

## Baseline preconditions

- Repo checkout with `tests/fixtures/gettysburg-75s.mp3` present.
- `uv`, `ffmpeg`, and `ffprobe` on `PATH`.
- Python 3.12 available to `uvx --python 3.12`.
- An isolated `--root` created for this run, not `~/podcue`.
- `python3 .cursor/skills/verify-podcue/drive.py` is the full journey. A single feature file is a slice of that journey.

## Driving conventions

- Install from the built wheel: `uvx --python 3.12 --from 'dist/podcue-<version>-py3-none-any.whl[recap]' podcue`.
- Pass `--root <isolated-dir>` on every command.
- Treat stdout as one JSON object. `{"kind":"error",...}` is a failure even if you only meant to inspect it.
- Do not delete `verify-podcue-evidence/` during cleanup.

## Proof and skip reporting

- Save the command stdout, the parsed JSON, and the files the command wrote.
- A `kind: built` line is not proof until `recap.mp3` and `report.md` are on disk and ffprobe reads the mp3.
- Record the feature id with the artifact.
- If a precondition is missing, stop and name it. Do not skip the feature and call another one a substitute.

## Features

- [start](./start.md) turns the 75-second clip into an episode with cues. `start --fresh` recomputes that episode.
- [label-groups](./label-groups.md) names a diarized group.
- [build](./build.md) narrates with Kokoro, cuts one clip, and writes `recap.mp3`.
- [resume](./resume.md) continues a killed `start` from `slices.json`, and takes over a stale lock.
