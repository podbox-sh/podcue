# Changelog

## 0.2.2

Not tagged and not published. The package metadata is `0.2.2`.

- `build` flags a clip only when its snapped length is under 4 seconds or over 45 seconds. A clip between those lengths is not a length flag. Flags still exit 0.
- A cue under 1.5 seconds that sits between two cues of one speaker, while another diarization speaker's turn overlaps it, is placed in the `uncertain` group. It no longer inherits the neighbor's group.
- A small ECAPA cluster is folded into a large one only when the centroid distance is within the existing linkage threshold (`0.7045654963945799`). A farther cluster stays its own speaker. Per-cue embeddings are not stored, so this split happens at clustering time.
- `build` reads each cue's `speaker`. A cue whose speaker is `uncertain` or `mixed` flags the clip even when that voice is under 1 second and 8%.
- `labeledShare` counts groups with a person's name. `undetermined`, `quote`, and `ending` are left out.
- `undeterminedShare` is the same duration share for the `undetermined` label only.

`createdAt` is the UTC time of the `start` that writes the cues. The episode id is a hash of the audio, so a failed or interrupted start leaves `episode.json` in that folder. The next start of the same audio used to copy `createdAt` from that file. A run at 23:53Z could still report 19:22Z from the earlier attempt. The clock was already UTC. A start that finds the episode already ready does not rewrite the stamp.

`start` does not need `HF_TOKEN`. Before any Hugging Face import it sets `HF_HUB_VERBOSITY=error` when that variable is unset, turns off hub telemetry and implicit tokens the same way, and ignores warnings whose message mentions `HF_TOKEN`.

### JSON

Added, and older readers can ignore them:

- `speaker` on each cue. For an ordinary cue it equals `groupId`.
- `undeterminedShare` on `label-groups` and on the episode view, next to `labeledShare`.
- `groupId` and `speaker` may be `uncertain`.

`labeledShare` keeps its name and its formula. Its meaning changed: `quote` and `ending` used to count as labeled, and they no longer do. No field was removed.

## 0.2.1

Built-in speech-to-text is the `asr` extra (`cactus-compute` 4.1.0). `podcue start --transcript` accepts `{"language", "segments": [{startMs, endMs, text, words?}]}`.

## 0.2.0

`fetch` and `build` for a recap, in the `recap` extra. Kokoro narration defaults to `am_michael`.
