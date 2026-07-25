#!/usr/bin/env bash
# Render figures/qualitative.pdf from the chest sweep's W&B checkpoint artifacts.
#
# Run on the cluster (needs a GPU + the chest cache at /workspace/data.h5). It
# pulls the trained checkpoints straight from W&B artifacts -- the ones the
# post-training upload in train.py now leaves behind -- so no local checkpoint
# dir is required.
#
#   bash scripts/render_qualitative_chest.sh
#   PATIENT=L058 SLICE=140 ROI=190,210,90,90 bash scripts/render_qualitative_chest.sh
#
# Eyeball a few SLICE/ROI values: pick a slice with a low-contrast structure
# (vessel/nodule/lesion) so the "waxy" supervised over-smoothing and the
# multi-step SSFlow drift are visible in the 2x ROI row.
set -euo pipefail

ENTITY=${ENTITY:-timgsereda}
PROJECT=${PROJECT:-ctdenoiser-sweep}
H5=${H5:-/workspace/data.h5}
PATIENT=${PATIENT:-L057}          # a held-out chest val patient (L057 / L058)
SLICE=${SLICE:-120}
ROI=${ROI:-}                      # x,y,w,h ; empty -> centre 80x80

# 1. Pull the checkpoints we want to visualise into ./ckpts/<name>/.
#    :latest is fine while only the single-seed chest sweep has written these
#    <model>-<mode> names; pin :v<N> if you later accumulate other anatomies.
python - "$ENTITY" "$PROJECT" <<'PY'
import os, sys, wandb
entity, project = sys.argv[1], sys.argv[2]
api = wandb.Api()
for n in ("redcnn-supervised", "redcnn-n2sim", "flowmatching-n2sim"):
    art = api.artifact(f"{entity}/{project}/{n}:latest", type="model")
    d = art.download(root=f"ckpts/{n}")
    pt = next(f for f in os.listdir(d) if f.endswith(".pt"))
    print(f"  {n:22s} -> {os.path.join(d, pt)}  (seed={art.metadata.get('seed')}, "
          f"val_psnr={art.metadata.get('val_psnr')})")
PY

# 2. Resolve the .pt inside each artifact dir (named <args.model>.pt on save).
SUP=$(ls ckpts/redcnn-supervised/*.pt | head -1)
NSIM=$(ls ckpts/redcnn-n2sim/*.pt | head -1)
FLOW=$(ls ckpts/flowmatching-n2sim/*.pt | head -1)

# 3. Render. flowmatching+n2sim trained a SelfSupervisedFlow, so the registry
#    key is 'ssflow' (NOT 'flowmatching'); the trailing :N sets the Euler step
#    count, so :1 is the one-step posterior mean and :20 the multi-step drift.
python scripts/figure_qualitative.py \
  --h5 "$H5" --patient "$PATIENT" --slice "$SLICE" \
  ${ROI:+--roi "$ROI"} \
  --models \
    "LDCT input=input" \
    "Noise2Sim=redcnn:$NSIM" \
    "Supervised=redcnn:$SUP" \
    "SSFlow (1 step)=ssflow:$FLOW:1" \
    "SSFlow (20 step)=ssflow:$FLOW:20" \
    "Full dose=clean" \
  --out figures/qualitative.pdf

echo "wrote figures/qualitative.pdf"
