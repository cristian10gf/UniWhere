# Step 3 - Reproducible ACE Training Run (block_g_1200)

Date: 2026-03-13
Status: blocked by environment (Docker daemon unavailable)

## Target artifact
- preprocesamiento/data/block_g_1200/ace/models/step3_repro_seed2089.pt

## Fixed training parameters selected
- num_head_blocks: 1
- epochs: 2
- training_buffer_size: 65536
- samples_per_image: 256
- batch_size: 512
- learning_rate_min: 0.0005
- learning_rate_max: 0.005
- use_aug: False
- use_half: False

## Command prepared
bash preprocesamiento/run-ace.sh train block_g_1200/ace block_g_1200/ace/models/step3_repro_seed2089.pt --data-root preprocesamiento/data -- --num_head_blocks 1 --epochs 2 --training_buffer_size 65536 --samples_per_image 256 --batch_size 512 --learning_rate_min 0.0005 --learning_rate_max 0.005 --use_aug False --use_half False

## Attempts and outcomes
1) Wrapper invocation failed in WSL delegation with:
   - cannot execute: required file not found
2) After line-ending normalization, ACE docker runner started but Docker from WSL was unavailable:
   - The command 'docker' could not be found in this WSL 2 distro
3) Direct Docker execution from PowerShell was attempted and failed due daemon not running:
   - failed to connect to the docker API at npipe:////./pipe/dockerDesktopLinuxEngine

## Local adjustments applied to enable completion when daemon is available
- preprocesamiento/run-ace.sh now delegates to docker/run.sh through explicit bash invocation.
- preprocesamiento/run-ace.sh and preprocesamiento/models/ace/docker/run.sh line endings normalized to LF.
- preprocesamiento/models/ace/ace_trainer.py now selects CUDA only when available, otherwise CPU.

## Next action
Start Docker Desktop daemon (or enable engine), then rerun the command above. After successful run, record:
- artifact existence and size
- non-empty training log
- run metadata (timestamp, command, parameters, exit code)
