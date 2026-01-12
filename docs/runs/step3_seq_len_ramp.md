# Step 3: seq_len ramp (ttt_llama_47b)

Environment: fresh **8×B200** node.

Common settings:
- `--config ttt_llama_47b`
- `--strategy fsdp --precision bf16`
- `--batch-size 1`
- `--tokenizer bytes`
- Dataset used on this node: `app4/ttt/data/assets/tiny.txt` (fallback; preferred tinyshakespeare file not present)
- Checkpoints: `/home/ubuntu/checkpoints` (outside repo)
- Logs: `/home/ubuntu/runs/logs` (outside repo)

## Stage summary

| dataset | seq_len | steps | toks/s range | step_s range | peak_mem_gb (max) | resume worked? | checkpoint dir |
| --- | ---: | ---: | ---: | ---: | ---: | :---: | --- |
| `app4/ttt/data/assets/tiny.txt` | 512 | 200 | n/a (done before node reset) | n/a | n/a | n/a | n/a (not present on this node) |
| `app4/ttt/data/assets/tiny.txt` | 1024 | 200 (+resume→220) | 453.1–651.3 | 12.578–18.079 | 72.99 | yes | `/home/ubuntu/checkpoints/step3_ts1024` |
| `app4/ttt/data/assets/tiny.txt` | 2048 | 200 (+resume→220) | 528.7–669.7 | 24.465–30.991 | 93.47 | yes | `/home/ubuntu/checkpoints/step3_ts2048` |
| `app4/ttt/data/assets/tiny.txt` | 4096 | 100 (+resume→110) | 578.4–684.4 | 47.876–56.657 | 134.43 | yes | `/home/ubuntu/checkpoints/step3_ts4096` |
| `/home/ubuntu/datasets/tinyshakespeare.txt` | 1024 | 200 (+resume→220) | 455.3–652.4 | 12.558–17.994 | 72.99 | yes | `/home/ubuntu/checkpoints/step3_ts1024_tinyshakespeare` |
| `/home/ubuntu/datasets/tinyshakespeare.txt` | 2048 | 200 (+resume→220) | 518.6–656.0 | 24.975–31.595 | 93.47 | yes | `/home/ubuntu/checkpoints/step3_ts2048_tinyshakespeare` |

