"""Evaluation code for illumination estimation task.

Usage example:

scene=everett_kitchen7
mkdir -p eval_output/probe_predict
for light_dir in `seq 6 7`; do
  python3 -m probe_predict.eval \
  --light_dir ${light_dir} \
  --out eval_output/probe_predict/${scene}_dir${light_dir}.jpg \
  ${scene}
done

Last evaluated on PyTorch 1.0.0.dev20181024
"""

import argparse
import os
import urllib

from probe_predict import model

import numpy as np
import torch as th

import multilum

def main():
  parser = argparse.ArgumentParser()

  parser.add_argument('scene')
  parser.add_argument('--light_dir', type=int)
  parser.add_argument('-o', '--out')

  parser.add_argument("--cropx", type=int, default=512)
  parser.add_argument("--cropy", type=int, default=256)

  parser.add_argument("--cropsz", type=int, default=512)
  parser.add_argument("--checkpoint", required=False)
  parser.add_argument("--mip", type=int, default=2)
  parser.add_argument("--chromesz", type=int, default=64)
  parser.add_argument("--fmaps1", type=int, default=6)
  parser.add_argument("--gpu_id", type=int, default=0)
  opts = parser.parse_args()

  if opts.checkpoint is None:
    opts.checkpoint = \
      "checkpoints/probe_predict/t_1547304767_nsteps_000050000.checkpoint"
    multilum.ensure_checkpoint_downloaded(opts.checkpoint)

  model = SingleImageEvaluator(opts)

  I = multilum.query_images(opts.scene, mip=opts.mip,
      dirs=opts.light_dir)[0,0]

  # extract crop
  ox = opts.cropx
  oy = opts.cropy
  cropnp = I[oy:oy+opts.cropsz, ox:ox+opts.cropsz]

  # pre process input
  crop = np.moveaxis(cropnp, 2, 0)
  crop = crop / 255 - 0.5

  # run model
  pred = model.forward(crop)

  # post process prediction
  pred = np.moveaxis(pred, 0, 2) + 0.5
  pred = (np.clip(autoexpose(pred), 0, 1) * 255).astype('uint8')

  # write results
  multilum.writejpg(cropnp, "%s.input.jpg" % opts.out)
  multilum.writejpg(pred, "%s.pred.jpg" % opts.out)




def autoexpose(I):
  n = np.percentile(I[:,:,1], 90)
  if n > 0:
    I = I / n
  return I

class SingleImageEvaluator:
  def __init__(self, opts):
    self.net = model.make_net_fully_convolutional(
        chromesz=opts.chromesz,
        fmaps1=opts.fmaps1,
        xysize=opts.cropsz)
    self.net.cuda()
    self.net.load_state_dict(th.load(opts.checkpoint))

  def forward(self, image):
    """Accept (3,512,512) image in [-0.5; 0.5] range.

    Returns (3,64,64) image in [-0.5; 0.5] range.
    """
    x = image
    h, w = image.shape[-2:]

    x = th.Tensor(x).view(1, 3, h, w)
    self.net.zero_grad()
    with th.no_grad():
      pred = self.net(th.autograd.Variable(x).cuda())

    pred = pred.view(3, self.net.chromesz, self.net.chromesz)
    pred = pred.detach().cpu().numpy()
    return pred


if __name__ == "__main__":
  main()