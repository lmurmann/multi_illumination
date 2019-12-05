# Runs on PyTorch 1.3.1
set -e

export PYTHONPATH=.


### Eval Multirelight
OUTDIR=eval_output/relight
mkdir -p $OUTDIR
python3 -m relight.eval \
	--checkpoint checkpoints/relight/epoch_13.pth \
	--output $OUTDIR