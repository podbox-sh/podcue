# Changelog

## 0.2.5

Not tagged and not published. The package metadata is `0.2.5`.

- A local file is stored in `episode.json` as its absolute path. A later `start` of the same audio updates that path when the file has moved.
- The first English narration downloads the pinned spaCy `en_core_web_sm` 3.8.0 wheel and says so on stderr. It is not a PyPI dependency.
- Narration is gained to -16 LUFS when it is synthesized and limited to -1.5 dBTP. The mix does not apply a boost larger than 3 dB. Torch deprecation warnings from Kokoro are filtered.
- `start --fresh` recomputes one episode's transcript, cues, and groups, and moves `labels.json` to `labels.prev.json` (replacing any older prev) so labels are done again. Other episodes, summaries, and the shared weight cache stay. `--root` with a new directory is a separate cold run.
- `start` records `asr` on the episode: faster-whisper with its pinned model and revision, Moonshine with arch `tiny` or `base`, or `external` for `--transcript`.
- Cue text joins number punctuation without a stray space (`400,000`, `2.4%`). A snapped clip stays inside its cues and does not overlap an adjacent cue of another speaker.
- `label-groups` accepts `"scope": "cues"` on a range. That sets `speaker` on the cues in the window and leaves the group's other cues alone. Later entries still win. `uncertain` and `mixed` are never relabeled.
- English narration reads numerals as words before Kokoro (`400,000` as "four hundred thousand", `2.4%` as "two point four percent", `$1.5 billion` as "one point five billion dollars", years, ordinals, `10x`, negatives, ranges, and times). Cue text and transcripts keep their digits. Other languages are unchanged.

## 0.2.4

Built-in speech-to-text for `en de fr es it nl pl` is Whisper small through faster-whisper 1.2.1, on CPU at int8. English uses `Systran/faster-whisper-small.en` at commit `d1d751a5f8271d482d14ca55d9e2deeebbae577f`. The other six languages use `Systran/faster-whisper-small` at `536b0662742c02347bc0e980a01041f333bce120`. `model.bin` is checked by size and SHA-256. `ja`, `ko`, and `zh` stay on Moonshine. cactus-compute is no longer a dependency.

## 0.2.3

Released.

- After `label-groups`, each cue's `speaker` is the label of its group (`Ada`, `quote`, `undetermined`, …). `groupId` is unchanged. Before any label, `speaker` is still the group id. A cue that is `uncertain` or `mixed` keeps that speaker when the group is labeled.
- `build` compares those `speaker` values. Two groups labeled the same person are one speaker. A cue whose speaker is `uncertain`, `mixed`, or `undetermined` flags the clip even when it is shorter than 1 second and 8% of the clip. A 0.2.2 cues file, where `speaker` is still the group id, still uses `labels.json` for that comparison.
- `label-groups` accepts `{"fromCue", "toCue", "label"}` beside the existing `{groupId, label}`. The range assigns that label to every group whose cues all fall inside the inclusive cue-id window. A group with a cue outside the window is not changed. That is one entry for a countdown that diarization split into many tiny quote groups, without labeling the host who continues past the window. `uncertain` and `mixed` are not selected by a range.
- The Hugging Face missing-token warning is silenced for every command, not only `start`. `quiet_hub` runs at the command entry and again immediately before Kokoro imports the hub.

### JSON

No fields were added or removed. `speaker` on a cue already existed. After `label-groups` its value is the group's label instead of a copy of `groupId`. `uncertain` and `mixed` are unchanged. The `label-groups` request accepts a second object shape; the response is the same `{labels, labeledShare, undeterminedShare}`.

## 0.2.2

Released.

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
