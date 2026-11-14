# resume

A killed `start` keeps each finished transcription slice in `slices.json`. The next `start` of the same audio continues there. A lock file left by a crash does not block that run.

## Sub-features

- `resume-checkpoint` writes `slices.json` after every finished slice.
- `resume-continue` skips slices whose index and time bounds match the checkpoint.
- `resume-fresh` deletes `slices.json` at the start of `start --fresh`.
- `resume-done` deletes `slices.json` once `cues.json` is written.
- `resume-leftover-lock` locks a `start.lock` or `write.lock` whose holder has exited. A flock a live process still holds is refused.

## How to get to it (user POV)

- Run `podcue start <audio> --language en` and stop it while it is printing `transcribing slice`.
- Run the same command again. The log says how many slices were already finished.
- Run `podcue start <audio> --language en --fresh` to discard the checkpoint and recompute.
- If a previous crash left `start.lock` or `write.lock`, run `start` or `label-groups` again.

## Driving it with the CLI

Preconditions:

- The episode directory exists and `cues.json` is not there yet, so `start` does not return the finished episode.
- `slices.json` is `{"version": 1, "audioSha256", "language", "slices": [...]}`. `audioSha256` is the episode id. A reusable slice has `state` `done` or `skipped-empty` and the same `index`, `startMs`, `endMs`, `keepStartMs`, `keepEndMs`, and `region` as the current plan. A different audio id or language starts over.
- A lock file whose holder has exited does not block. The flock is gone with that process. A lock a live process still holds is `another start ... is running`, including when the file is hours old or the pid written in it is not running.

- **Resume.** Run `podcue --root <root> start <audio> --language en`. Exit 0, or exit 1 only for a new failure. Stderr contains `resuming transcription, N of M slices already finished`. Slices already in the checkpoint are not transcribed again.
- **Discard.** Run `podcue --root <root> start <audio> --language en --fresh`. `slices.json` is gone before transcription starts. `created` is true.
- **Replace a leftover lock.** Leave `start.lock` containing a pid that is not running, and do not hold its flock. That is a crash: a clean finish clears the pid. Run `start` again. Stderr contains `replacing start.lock left by pid`. The command is not `reason: running`. A clean `start` followed by another command does not print that line.
- **Proof.** After a successful `start`, `slices.json` is absent and `cues.json` is present. Save the stderr lines that mention `resuming transcription` or `replacing start.lock`.

## Gotchas

- `start` returns immediately when `cues.json` already exists. Resume is only visible when that file is missing.
- A checkpoint slice whose bounds do not match the new plan is transcribed again. Do not assert that every old index was skipped.
- A lock a live process still holds is `another start ... is running`. Age and the pid text do not make it free.
- The lock file stays when the holder finishes. Its mtime is not read, so a laptop sleeping does not free a flock a process still holds. A child that inherited the fd is a live holder until that child exits. Do not delete locks by killing a process named `podcue`.
- The full `verify-podcue` drive deletes its `--root`, so it does not leave a checkpoint to resume. This feature is checked by the unit tests, not by that drive.
