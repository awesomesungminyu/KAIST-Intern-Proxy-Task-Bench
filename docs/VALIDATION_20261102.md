# Validation

The frozen evaluation protocol is in [validation/protocol.json](../validation/protocol.json).
For historical results, use the [Drive manifest](../drive_upload_20261123/README.md).

The shipped baseline JSON contains only smoke-report metadata. For a
scored legacy report, extract the original baseline to an untracked
directory using the [restricted Drive archive](../drive_upload_20261123/README.md),
then pass it explicitly:

```bash
python -m validation report --suite runs/example --baselines path/to/final16_v4.json
```

For the shell integration in full mode, set `EVO_BASELINES` to that file.
The portable CLI has its own public Transformer reference for retention.
