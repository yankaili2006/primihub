# DoublePIR example task configs

Three starter configs paired with the `params.algorithm = "double_pir"`
multi-algo registry (openspec change `primihub-pir-multi-algo`, task
5.6 chunks 1-7).

| File | Mode | When |
|---|---|---|
| `double_pir_task_conf.json` | single-process | local dev / smoke; one server runs the full Init+Setup+Query+Answer+Recover loop |
| `double_pir_primary.json` | `hint_role="primary"` | production primary — runs `HintGen` once and `BroadcastHint` to every peer in `peer_nodes` |
| `double_pir_secondary.json` | `hint_role="secondary"` | production secondary — `ReceiveHint` from `peer_nodes[0]` replaces `HintGen`; saves O(L·M·n) Setup |

All three are **stubs**: every `FILL_IN_*` placeholder must be replaced
before the task will run end-to-end. The placeholders are:

| Placeholder | Where | What to put |
|---|---|---|
| `FILL_IN_DATASET_ID` | `party_datasets.SERVER.SERVER` | The registered dataset id on the SERVER party |
| `FILL_IN_PRIMARY_NODE_ID` | `party_info.<id>` | Stable node identifier for the primary peer |
| `FILL_IN_PRIMARY_IP` | `party_info.<id>.ip` | Primary peer's gRPC-reachable IP |
| `FILL_IN_SECONDARY_NODE_ID` | `party_info.<id>` | Stable node identifier for the secondary peer |
| `FILL_IN_SECONDARY_IP` | `party_info.<id>.ip` | Secondary peer's gRPC-reachable IP |

The `params.hint_path` value picks a writable cache location.
Production deployments typically mount a host directory at
`/var/cache/primihub/` (Docker bind-mount or NFS).

See `docs/pir/multi-algo-guide.md` § Two-peer hint distribution for
the cheat sheet on which mode to use, and `docs/pir/hint-lifecycle.md`
§ Implementation status (chunks 1-7) for the underlying mechanics.

## Quick start with the pcloud `primihub-pir` skill

```bash
# Single-process smoke (works the moment FILL_IN_DATASET_ID is set)
python3 skills/primihub-pir/cli.py run \
    --task-config example/double_pir_task_conf.json \
    --algorithm double_pir \
    --hint-path /tmp/double_pir_hint.bin

# Two-peer: run primary and secondary on their respective hosts
python3 skills/primihub-pir/cli.py run \
    --task-config example/double_pir_primary.json \
    --algorithm double_pir --hint-role primary \
    --hint-path /var/cache/primihub/double_pir_hint_primary.bin

# (on the secondary host)
python3 skills/primihub-pir/cli.py run \
    --task-config example/double_pir_secondary.json \
    --algorithm double_pir --hint-role secondary
```

The skill flags above `--algorithm` / `--hint-path` / `--hint-role`
already patch the matching `params.<key>` STRING entries, so the
copy in the JSON file is just a documented default — the skill's
override wins.
