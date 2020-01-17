"""Lanczos filter implementation (uint8, uint16, float32, float64).

Evaluated alternatives:
  PIL has a lanczos filter but only for uint8.
  scipy.mis.imresize() silently converts float to uint8 and calls into PIL
  skimage only has lanczos upsampling but downsampling uses gaussian filter.

The implementation below is reasonably efficient thanks to scipy sparse matrix
multiply. For better performance this can be replaced with  a specialized lanczos
implementation such as Intel Integrated Performance Primitives.
"""

import argparse
from math import floor, ceil

import numpy as np
import scipy
from scipy.sparse import csr_matrix


__all__ = ('resize_lanczos')

def resize_lanczos(I, h, w):
  I = horizontal_lanczos(I, w)
  # we force the intermediate result after x-filter back into I.dtype.
  # this might lead to different roundoff error than an implementation that
  # stores the intermediate result as floats.
  I = np.swapaxes(I, 0, 1)
  I = horizontal_lanczos(I, h)
  return np.swapaxes(I, 0, 1)


def lanczos2(x):
  """Evaluate the three-lobed (a=2) lanczos filter. The image filter
  precomputes filter weights once per image and reuses the same weights
  for each image row (horizontal filter) or column (vertical filter)."""
  a = 2
  ret = np.zeros_like(x)
  ret[x == 0] = 1
  ret[x != 0] = np.sinc(x[x != 0]) * np.sinc(x[x != 0] / a)
  ret[np.abs(x) >= a] = 0
  return ret

def saturation_cast(I, dtype):
  """Utility for filtering integer images"""
  if dtype=='uint8':
    return np.clip(I, 0, 255).astype(dtype)
  elif dtype=='uint16':
    return np.clip(I, 0, 65535).astype(dtype)
  elif dtype in ['float32', 'float64']:
    return I.astype(dtype)
  else:
    raise ValueError("Saturation cast for type %s not implemented" % I.type)


def horizontal_lanczos(I, w, comment=None):
  """Apply 3-lobed Lanczos filter along horizontal (x) dimension.

  Returned image has shape (I.shape[0], w, I.shape[2]) and same
  dtype as the input image.

  Set "comment" argument for debug information
  """
  dt = I.dtype
  inh, inw = I.shape[:2]
  nchan = 1 if len(I.shape) == 2 else I.shape[2]
  winsz = 5
  if comment:
    from lum.tictoc import tictoc
    print("winsz is", winsz)


  if comment:
    print("compute %s filter" % comment)
    tic = tictoc()

  xs = inw/w
  xfltscale = max(1, xs)
  srange = np.arange(floor(-2*xfltscale), ceil(2*xfltscale)+1)

  # We treat image filtering as sparse linear matrix multiply R = M * A
  # with filter matrix M. We start by constructing the matrix.
  ntap = len(srange)
  row = np.zeros([w,ntap], dtype='int32')
  col = np.zeros([w,ntap], dtype='int32')
  data = np.zeros([w,ntap], dtype='float32')

  if comment:
    print("using %d tap filter" % len(srange))

  for x in range(w):
    center = x * xs
    # it is important to always round down (or up) here
    # round-to-even will cause misaligned indices.
    # this become especially noticeable when upsampling
    # by integer factors
    r = (center + srange + 0.5).astype('int32')
    weights = lanczos2((r - center) / xfltscale)

    # handle image bounds, normalize filter weight
    # in order to handle out of bounds, we clamp the indices
    # to [0, w) and set the associated weight to zero.
    weights[r<0] = 0
    weights[r>=inw] = 0
    r[r<0] = 0
    r[r>=inw] = 0
    weights /= np.sum(weights)

    for ti, c in enumerate(r):
      row[x, ti] = x
      col[x, ti] = c
      data[x, ti] = weights[ti]

  row = np.reshape(row, [-1])
  col = np.reshape(col, [-1])
  data = np.reshape(data, [-1])
  M = csr_matrix((data, (row, col)), shape=(w, inw))

  if comment:
    print("took", tic.toc())
    tic.tic()

  if comment:
    print("apply %s filter" % comment)

  # Construct right-hand side. R = M * A
  # Each column of A has one row of the input image.
  # A has (I.ncols * I.nchans) columns
  A = np.moveaxis(I, 1, 0)
  A = np.reshape(A, [inw, inh*nchan])
  R = (M * scipy.sparse.csr_matrix(A)).toarray()

  # Reshape R into [h,w,c] image format
  R = np.reshape(R, [w, inh, nchan])
  R = np.moveaxis(R, 1, 0)
  R = saturation_cast(R, dt)
  R = R.reshape([inh, w, nchan])

  if comment:
    print("took", tic.toc())
  return R

def main():
  """Test stub for development only"""
  from multilum import readexr, writeexr
  from lum import plt
  from lum import pfm

  parser = argparse.ArgumentParser()
  parser.add_argument("image")
  opts = parser.parse_args()

  if opts.image.endswith("jpg"):
    I = pfm.load_jpg3(opts.image, dtype="uint8")
  else:
    I = readexr(opts.image)

  writeexr(I, "outfsz.exr")
  Ism = resize_lanczos(I, 100, 150)
  writeexr(Ism, "out.exr")

  plt.gshow(Ism)
  plt.show()

if __name__ == "__main__":
  main()

