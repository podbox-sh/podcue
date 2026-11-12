# label-groups

`podcue label-groups` writes a speaker name onto a diarized group. Later cue reads show that name as `speaker`. The group id does not change.

## Sub-features

- `label-apply` accepts one `{groupId, label}` entry and returns `state: present`.
- `label-name` counts a person's name in `labeledShare`.
- `label-held` does not relabel a group whose id is `uncertain` or `mixed`.
- `label-locks` leaves no `write.lock` after the command exits.

## How to get to it (user POV)

- After `podcue start`, copy a `groupId` from the start JSON or from `podcue cues`.
- Run `podcue label-groups <episodeId> '<json array>'`.
- Run `podcue cues <episodeId>` again if you need to see the `speaker` field change.

## Driving it with the CLI

Preconditions:

- `start` on this `--root` has finished and returned at least one group.
- The chosen `groupId` is not `uncertain` or `mixed` when any other group exists. `ungrouped` is used only when it is the only group.

- **Apply a name.** Run `podcue --root <root> label-groups <episodeId> '[{"groupId":"<groupId>","label":"John Greenman"}]'`. Exit 0. `state` is `present`. `labels` contains that group id and `John Greenman`.
- **Proof.** Save stdout to `verify-podcue-evidence/label-groups.json`. The episode directory contains no `write.lock` or `start.lock`.

## Gotchas

- The JSON array is one shell argument. Quoting it wrong makes podcue print a usage error, which is `{"kind":"error",...}` and exit 1.
- An unknown group id returns exit 0 with `state: unknown`. That is not a successful label. Require `state: present`.
- `uncertain` and `mixed` are never relabeled. Picking one and expecting `speaker` to become `John Greenman` fails the check for the wrong reason.
- `labeledShare` ignores the marks `undetermined`, `quote`, and `ending`. The proof label is a person's name.
- `start --fresh` after this command moves `labels.json` to `labels.prev.json`. The label is then gone until you run `label-groups` again.
