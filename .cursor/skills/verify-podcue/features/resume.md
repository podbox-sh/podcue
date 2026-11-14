# resume

A killed `start` keeps each finished transcription slice in `slices.json`. The next `start` of the same audio continues there. A stale lock file does not block that run.

## Sub-features

- `resume-checkpoint` writes `slices.json` after every finished slice.
- `resume-continue` skips slices whose index and time bounds match the checkpoint.
- `resume-fresh` deletes `slices.json` at the start of `start --fresh`.
- `resume-done` deletes `slices.json` once `cues.json` is written.
- `resume-stale-lock` takes over an empty, dead, or 12-hour-old `start.lock` or `write.lock`.

## How to get to it (user POV)

- Run `podcue start <audio> --language en` and stop it while it is printing `transcribing slice`.
- Run the same command again. The log says how many slices were already finished.
- Run `podcue start <audio> --language en --fresh` to discard the checkpoint and recompute.
- If a previous crash left `start.lock` or `write.lock`, run `start` or `label-groups` again.

## Driving it with the CLI

Preconditions:

- The episode directory exists and `cues.json` is not there yet, so `start` does not return the finished episode.
- `slices.json` is `{"version": 1, "slices": [...]}`. A reusable slice has the same `index`, `startMs`, `endMs`, `keepStartMs`, `keepEndMs`, and `region` as the current plan.
- A stale lock is empty and at least 30 seconds old, or contains a pid that is not running, or is at least 12 hours old. A lock held by the current process with a live pid is not stale.

- **Resume.** Run `podcue --root <root> start <audio> --language en`. Exit 0, or exit 1 only for a new failure. Stderr contains `resuming transcription, N of M slices already finished`. Slices already in the checkpoint are not transcribed again.
- **Discard.** Run `podcue --root <root> start <audio> --language en --fresh`. `slices.json` is gone before transcription starts. `created` is true.
- **Take over a lock.** Leave `start.lock` empty and older than 30 seconds, still flocked by a dead holder. Run `start` again. Stderr contains `taking over stale start.lock`. The command is not `reason: running`.
- **Proof.** After a successful `start`, `slices.json` is absent and `cues.json` is present. Save the stderr lines that mention `resuming transcription` or `taking over stale`.

## Gotchas

- `start` returns immediately when `cues.json` already exists. Resume is only visible when that file is missing.
- A checkpoint slice whose bounds do not match the new plan is transcribed again. Do not assert that every old index was skipped.
- A lock that is empty and younger than 30 seconds, with a live holder, is still `another start ... is running`. Do not treat every empty file as stale.
- Taking over a lock unlinks that file and locks the new one. Do not kill processes by name to clear it.
- The full `verify-podcue` drive deletes its `--root`, so it does not leave a checkpoint to resume. This feature is checked by the unit tests, not by that drive.
