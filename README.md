# kinfer

This package is designed to support running real-time robotics models.

For more information, see the documentation [here](https://docs.kscale.dev/docs/k-infer).

To enable logging, set your log path in the environment variable `KINFER_LOG_PATH`. For example,
```
export KINFER_LOG_PATH=/home/dpsh/kinfer-logs
```




k-infer portable runtime file format
---------------------------------------------------------------------------------- 
k-infer runtime file is a polyglot ( model and comopressed archive ) meaning it is simultaneously a kinfer model and a ZIP file.
The ZIP file contains the original training and export code, checkpoint, config and metadata

The first 32bytes contain the ZIP local header, along with some magic, then the actual kinfer model followed by a compressed archive

┌──────────  32 B  ──────────┐
│ 0x0000  ZIP local-header #0  ← overlaps kinfer “front porch” (30 B)
│ 0x001E  0xCE  kinfer-polyglot magic
│ 0x001F  0x01  kinfer hdr-version / flags
└────────────────────────────┘
0x0020  ──── the original kinfer header & payload ────  (LEN = K)
          .
          .                   ← kinfer reader stops here (0x0020+K)
          .
┌──────────── additional ZIP entries (optional) ────────────────────────┐
│ 0x0020+K  local-hdr #1 (training_code.py”), data                      |
| -         local-hdr #1 (convert.py”), data                            │
| -         local-hdr #1 (joint_config_table.txt”), data                │
| -         local-hdr #1 (config.yaml”), data                           │
| -         local-hdr #1 (info.json”), data                             │
| -         local-hdr #1 (logs.txt”), data                              │
| -         local-hdr #1 (state.txt”), data                             │
| -         local-hdr #1 (checkpoints/*”), data                         │
│ …                                                                     │
│ local-hdr #N (“assets/img.png”), data                                 │
└───────────────────────────────────────────────────────────────────────┘
central-directory (1 + N records)   ← points back to *all* local hdrs  
EOCD
