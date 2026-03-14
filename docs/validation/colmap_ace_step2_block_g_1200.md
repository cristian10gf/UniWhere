# Step 2 - Pose and Intrinsics Convention Verification (block_g_1200)

Date: 2026-03-13

## Scope
Validated plan step 2 for current COLMAP -> ACE artifacts:
- Pose convention: COLMAP world-to-camera to ACE camera-to-world conversion.
- Intrinsics path consistency: single focal vs 3x3 matrix output shape/value.

## Execution
Python was executed with `uv`:

```bash
uv run --python c:/Users/crist/Documents/proyectos/UniWhere/.venv/Scripts/python.exe \
  preprocesamiento/scripts/step2_pose_intrinsics_validation.py \
  --scene-dir preprocesamiento/data/block_g_1200 \
  --out docs/validation/colmap_ace_step2_block_g_1200.json
```

## Inputs used
- Scene: `preprocesamiento/data/block_g_1200`
- ACE dataset: `preprocesamiento/data/block_g_1200/ace`
- Auto-selected sparse model with text files and max stem overlap: `preprocesamiento/data/block_g_1200/sparse/0`

## Results
- Matched ACE entries: 4 / 4
- Missing stems: none
- Camera model path exercised: `SIMPLE_RADIAL` only (single-focal)
- Calibration shape mismatches: none

Numerical consistency:
- Camera-center error (m): median `2.79e-16`, max `1.90e-15`
- Rotation error (deg): median `0.0`, max `1.21e-6`
- Bottom row pose consistency (`[0,0,0,1]`): max `0.0`
- Focal extraction error (px): median `0.0`, max `0.0`

## Step-2 verdict
Pass for current converted dataset:
- world-to-camera to camera-to-world inversion is numerically consistent
- intrinsics extraction is consistent for single-focal camera models

## Limits
- This verification covers the currently converted 4-image ACE dataset mapped to sparse/0.
- The matrix intrinsics path (3x3 K) is not exercised in this scene/model.
