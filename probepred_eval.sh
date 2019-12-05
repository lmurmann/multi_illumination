# Runs on PyTorch 1.3.1
set -e
export PYTHONPATH=.

scene=everett_kitchen7

mkdir -p eval_output/probe_predict
for light_dir in `seq 6 7`; do

python3 -m probe_predict.eval \
--light_dir ${light_dir} \
--out eval_output/probe_predict/${scene}_dir${light_dir}.jpg \
${scene}
done

