# build

`podcue build` reads `script.json`, speaks narration with Kokoro, cuts a clip from the episode audio, and writes one mp3 plus `report.md`.

## Sub-features

- `build-narration` speaks two lines, one containing `400,000` and one containing `2.4%`.
- `build-clip` cuts one clip that overlaps cue text.
- `build-file` writes `recap.mp3` that ffprobe can time.
- `build-lufs` leaves each narration segment within 2 LU of -16 LUFS.
- `build-report` writes `report.md` next to the mp3.
- `build-locks` leaves no `*.lock` file after the command exits.

## How to get to it (user POV)

- Finish `start` and, when you want a name on the clip, `label-groups`.
- Write `script.json` with narration and clip segments.
- Run `podcue build <episodeId> script.json --out recap.mp3`.
- Open `recap.mp3` and the `report.md` beside it.

## Driving it with the CLI

Preconditions:

- `cues` returned cue text, so the clip has words to snap to.
- `--root` still points at the episode `start` wrote.
- The wheel install includes the `recap` extra, and the interpreter is Python 3.12. Kokoro 0.9.4 does not install on 3.13 or 3.14.
- `espeak-ng` is on `PATH`. Kokoro's English phonemizer uses it for words outside its dictionary.

- **Write the script.** Put two narration segments and one clip in `<root>/script.json`. The narration texts are `About 400,000 people know this address.` and `The excerpt is 2.4% of a full hour.` The clip `start` and `end` cover about 12 seconds from the first cue, and `speaker` is `John Greenman`.
- **Build.** Run `podcue --root <root> build <episodeId> <root>/script.json --out <root>/recap.mp3`. Exit 0. `kind` is `built`. `segments` has two `narration` entries and one `clip`.
- **Proof.** Copy `recap.mp3`, `report.md`, and `script.json` to `verify-podcue-evidence/`. `ffprobe` duration of the mp3 is between 8 and 90 seconds. Each narration `lufsOut` is within 2 of -16. Save that reading as `verify-podcue-evidence/lufs.txt`.

## Gotchas

- A narration file saved as `.raw` fails the build with `{"kind":"error","message":"No default subtype for major format 'raw'"}` before any mp3 exists. The proof is a playable `recap.mp3`, not a successful-looking log line from earlier commands.
- Flags for a short or long clip are warnings. They stay in `flags` and the command still exits 0. An empty snap or a missing source file is `kind: error` and exit 1.
- `report.md` is written next to `--out`, which may be outside the episode directory. Copy it out before deleting `--root`.
- The first narration downloads Kokoro weights and the pinned spaCy model. A later build in the same environment does not. The doctor report may show those caches as absent on a cold machine.
- Python 3.13 or 3.14 exits 1 and names the 3.12 install command. Do not treat that message as a Kokoro audio failure.
