"""Pytorch model for illumination estimation task."""

from torch import nn
import math


def make_encoder(fmaps, fmaps1=None, xysize=None):
  if fmaps1 is None:
    fmaps1 = fmaps
  layers = []

  ndowns = int(math.log2(xysize))
  for i in range(ndowns):
    if i == 0:
      fmaps_out = fmaps1
    else:
      fmaps_out = fmaps * 2

    layers.extend([
      nn.Conv2d(fmaps, fmaps_out, 3, padding=1, bias=True),
      nn.ReLU(),
      nn.MaxPool2d(2)
    ])
    fmaps = fmaps_out
    xysize //= 2
  print("%d layer network downsample to %dx%dx%d" % (ndowns, fmaps, xysize, xysize))
  return layers, fmaps, xysize

def make_net_fully_convolutional(*,chromesz, fmaps1, xysize):
  layers, fmaps, _ = make_encoder(fmaps=3, fmaps1=fmaps1, xysize=xysize)
  fmaps_out = chromesz*chromesz*3
  layers.append(nn.Conv2d(fmaps, fmaps_out, 1, padding=0, bias=True))

  ret = nn.Sequential(*layers)
  ret.fully_convolutional = True
  ret.chromesz = chromesz
  ret.fmaps1 = fmaps1
  ret.cropsz = xysize
  return ret
