# start

`podcue start` turns an audio file into an episode: a 16 kHz mono conversion, diarization, faster-whisper transcription, and cue files under `--root/episodes/<episodeId>/`.

## Sub-features

- `start-create` writes a new episode and returns `kind: started` with `created: true`.
- `start-asr` records `asr.engine` `faster-whisper` for `--language en`.
- `start-cues` makes `podcue cues` return the cue text.
- `start-fresh` recomputes the same episode id and moves `labels.json` aside when labels exist.
- `start-locks` leaves no `*.lock` file after the command exits.

## How to get to it (user POV)

- Run `podcue start <audio> --language en` on a local wav or mp3.
- Run `podcue start <audio> --language en --fresh` on an episode that already has cues.
- Run `podcue cues <episodeId>` to read the result back.

## Driving it with the CLI

Preconditions:

- `python3 .cursor/skills/verify-podcue/drive.py` has passed doctor, or you have run the same `uvx --python 3.12 --from '<wheel>[recap]'` install yourself.
- `--root` is an empty directory this run created.
- `tests/fixtures/gettysburg-75s.mp3` is the audio. ffprobe duration is between 60 and 90 seconds.

- **Create the episode.** Run `podcue --root <root> start tests/fixtures/gettysburg-75s.mp3 --language en`. Exit 0. Stdout is one JSON object with `kind` `started`, `created` true, and `episode.asr.engine` `faster-whisper`. `episode.episodeId` looks like `ep_` plus 16 hex digits.
- **Read the cues.** Run `podcue --root <root> cues <episodeId> --limit 200`. Exit 0. `state` is `present` and `cues` is a non-empty array. At least one cue has non-empty `text`.
- **Recompute.** Run `podcue --root <root> start tests/fixtures/gettysburg-75s.mp3 --language en --fresh`. Exit 0. `created` is true and `episodeId` is unchanged.
- **Proof.** Save stdout to `verify-podcue-evidence/start.json` and `verify-podcue-evidence/start-fresh.json`. The episode directory contains no file whose name ends in `.lock`.

## Gotchas

- `start` without the `recap` extra still runs when the `asr` extra is installed. This journey installs `podcue[recap]` so the later build can narrate. Do not swap in a transcript file; the check is real faster-whisper.
- The episode id is the hash of the converted wav. A different file, or a different trim, is a different episode.
- `start --fresh` deletes the effect of `label-groups` by moving `labels.json` to `labels.prev.json`. Run it after build when the journey must see the label on the recap.
- Speech the diarizer misses is not transcribed. An empty `cues` array is a failed start, not a reason to invent cue times.
- A crash can leave `start.lock`. A finished command, including a failed one, removes it. Do not delete locks by killing a process named `podcue`.
