"""Evaluate relighting model."""

import os
import argparse

import numpy as np
import torch as th

import multilum
from relight.model import Relighter


def main(opts):

  # The model is trained to expect input illumination 0 (light form behind the camera)
  src = [0]
  # In a single forward pass, we predict the scene under 8 novel light conditions.
  tgt = [5, 7, 12, 4, 16, 6, 17, 11]
  num_lights = len(tgt)

  model = Relighter(**{
      'n_in': 1,
      'n_out': 8,
      'normals': False})
  model.cuda()

  multilum.ensure_checkpoint_downloaded(opts.checkpoint)
  chkpt = th.load(opts.checkpoint)
  model.load_state_dict(chkpt["model"])

  eps = 1e-4

  sample = {
    "input": _preprocess(multilum.query_images(opts.scene, dirs=src, mip=3, hdr=True)[0]),
    "target": _preprocess(multilum.query_images(opts.scene, dirs=tgt, mip=3, hdr=True)[0])
  }
  with th.no_grad():

    in_ = sample["input"]
    in_ = th.log(eps + in_)
    mean = in_.mean()
    # in_ -= mean
    # in_ -= 1
    # in_ = th.exp(in_) - eps

    out = model.forward(sample)
    # undo normalization
    out = th.log(eps + out)
    out += 1
    out += mean
    out = th.exp(out) - eps

    # From here on on it is just gamma correction and file output
    in_ = sample["input"].detach()
    gt = sample["target"].detach()

    # gamma correction
    gamma = 1.0/2.2
    in_ = th.pow(in_, gamma)
    gt = th.pow(gt, gamma)
    out = th.pow(out, gamma)

    in_ = th.clamp(in_, 0, 1)
    gt = th.clamp(gt, 0, 1)
    recons = th.clamp(out.detach(), 0, 1)

    # write input file
    fname_in = os.path.join(opts.output, "input", "%s_dir%d.jpg" % (opts.scene, src[0]))
    _save(in_, fname_in)

    bs, _, h, w = recons.shape
    recons = recons.view(bs, num_lights, 3, h, w)
    gt = gt.view(bs, num_lights, 3, h, w)

    # write predictions and ground truth
    for l_idx in range(num_lights):
      _sname = "%s_dir%d.jpg" % (opts.scene, tgt[l_idx])
      r = recons[:, l_idx]
      g = gt[:, l_idx]
      fname_recons = os.path.join(opts.output, "relight", _sname)
      _save(r, fname_recons)
      fname_gt = os.path.join(opts.output, "gt", _sname)
      _save(g, fname_gt)
      print("write result", fname_recons)

  print("write input", fname_in)


def _preprocess(I):
  I = I.copy()
  I[I<0] = 0
  I = I / np.percentile(I, 90)
  return th.Tensor(np.moveaxis(I, 3, 1)).cuda()

def _save(data, fname, makedirs=True):
  if makedirs:
    os.makedirs(os.path.dirname(fname), exist_ok=True)
  npdata = data[0].cpu().permute(1, 2, 0).numpy()
  npdata = (np.clip(npdata, 0, 1) * 255).astype('uint8')
  multilum.writeimage(npdata, fname)



if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint')
  parser.add_argument('--output')
  parser.add_argument('--scene', default="everett_kitchen7")
  opts = parser.parse_args()
  main(opts)
