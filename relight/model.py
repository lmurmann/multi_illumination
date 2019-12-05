"""Reusable modules for relighting model."""

import torch as th
import torch.nn as nn
import torch.nn.functional as F

#########################################




class Relighter(nn.Module):
  """
  2018-04-08
  """
  def __init__(self, width=64, n_in=1, n_out=1, normals=False):
    super(Relighter, self).__init__()

    self.normals = normals

    w = width

    n_in = 3*n_in
    n_out = 3*n_out

    if normals:
      n_in += 3

    self.autoencoder = Autoencoder(
        n_in, n_out, num_levels=8,
        increase_factor=2,
        num_convs=2, width=w, ksize=3,
        activation="relu",
        pooling="max",
        output_type="linear")

  def forward(self, samples):
    im = samples["input"]
    if "normals" in samples.keys() and self.normals:
      im = th.cat([im, samples["normals"]], -3)
    out = self.autoencoder(im)
    return out


class ConvChain(nn.Module):
  def __init__(self, ninputs, noutputs, ksize=3, width=64, depth=3, stride=1,
               pad=True, normalize=False, normalization_type="batch",
               output_type="linear",
               activation="relu", weight_norm=True):
    super(ConvChain, self).__init__()

    assert depth > 0

    if pad:
      padding = ksize//2
    else:
      padding = 0

    layers = []
    for d in range(depth-1):
      if d == 0:
        _in = ninputs
      else:
        _in = width
      layers.append(
          ConvBNRelu(
            _in, ksize, width, normalize=normalize, normalization_type="batch", padding=padding,
            stride=stride, activation=activation, weight_norm=weight_norm))

    # Last layer
    if depth > 1:
      _in = width
    else:
      _in = ninputs

    conv = nn.Conv2d(_in, noutputs, ksize, bias=True, padding=padding)
    if weight_norm:
      conv = nn.utils.weight_norm(conv)  # TODO
    conv.bias.data.zero_()
    if output_type == "elu" or output_type == "softplus":
      nn.init.xavier_uniform_(
          conv.weight.data, nn.init.calculate_gain("relu"))
    else:
      nn.init.xavier_uniform_(
          conv.weight.data, nn.init.calculate_gain(output_type))
    layers.append(conv)

    # Rename layers
    for im, m in enumerate(layers):
      if im == len(layers)-1:
        name = "prediction"
      else:
        name = "layer_{}".format(im)
      self.add_module(name, m)

    if output_type == "linear":
      pass
    elif output_type == "relu":
      self.add_module("output_activation", nn.ReLU(inplace=True))
    elif output_type == "leaky_relu":
      self.add_module("output_activation", nn.LeakyReLU(inplace=True))
    elif output_type == "sigmoid":
      self.add_module("output_activation", nn.Sigmoid())
    elif output_type == "tanh":
      self.add_module("output_activation", nn.Tanh())
    elif output_type == "elu":
      self.add_module("output_activation", nn.ELU())
    elif output_type == "softplus":
      self.add_module("output_activation", nn.Softplus())
    else:
      raise ValueError("Unknon output type '{}'".format(output_type))

  def forward(self, x):
    for m in self.children():
      x = m(x)
    return x


class ConvBNRelu(nn.Module):
  def __init__(self, ninputs, ksize, noutputs, normalize=False,
               normalization_type="batch", stride=1, padding=0,
               activation="relu", weight_norm=True):
    super(ConvBNRelu, self).__init__()
    if activation == "relu":
      act_fn = nn.ReLU
    elif activation == "leaky_relu":
      act_fn = nn.LeakyReLU
    elif activation == "tanh":
      act_fn = nn.Tanh
    elif activation == "elu":
      act_fn = nn.ELU
    else:
      raise NotImplemented

    if normalize:
      conv = nn.Conv2d(ninputs, noutputs, ksize, stride=stride, padding=padding, bias=False)
      if normalization_type == "batch":
        nrm = nn.BatchNorm2d(noutputs)
      elif normalization_type == "instance":
        nrm = nn.InstanceNorm2D(noutputs)
      else:
        raise ValueError("Unkown normalization type {}".format(normalization_type))
      nrm.bias.data.zero_()
      nrm.weight.data.fill_(1.0)
      self.layer = nn.Sequential(conv, nrm, act_fn())
    else:
      conv = nn.Conv2d(ninputs, noutputs, ksize, stride=stride, padding=padding)
      if weight_norm:
        conv = nn.utils.weight_norm(conv)  # TODO
      conv.bias.data.zero_()
      self.layer = nn.Sequential(conv, act_fn())

    if activation == "elu":
      nn.init.xavier_uniform_(conv.weight.data, nn.init.calculate_gain("relu"))
    else:
      nn.init.xavier_uniform_(conv.weight.data, nn.init.calculate_gain(activation))

  def forward(self, x):
    out = self.layer(x)
    return out


class Autoencoder(nn.Module):
  def __init__(self, ninputs, noutputs, ksize=3, width=64, num_levels=3,
               num_convs=2, max_width=512, increase_factor=1.0,
               normalize=False, normalization_type="batch",
               output_type="linear",
               activation="relu", pooling="max"):
    super(Autoencoder, self).__init__()


    next_level = None
    for lvl in range(num_levels-1, -1, -1):
      n_in = min(int(width*(increase_factor)**(lvl-1)), max_width)
      w = min(int(width*(increase_factor)**(lvl)), max_width)
      n_us = min(int(width*(increase_factor)**(lvl+1)), max_width)
      n_out = w
      o_type = activation

      if lvl == 0:
        n_in = ninputs
        o_type = output_type
        n_out = noutputs
      elif lvl == num_levels-1:
        n_us = None

      next_level = AutoencoderLevel(
          n_in, n_out, next_level=next_level, num_us=n_us,
          ksize=ksize, width=w, num_convs=num_convs,
          output_type=o_type, normalize=normalize,
          normalization_type=normalization_type,
          activation=activation, pooling=pooling)

    self.add_module("net", next_level)

  def forward(self, x):
    return self.net(x)


class AutoencoderLevel(nn.Module):
  def __init__(self, num_inputs, num_outputs, next_level=None,
               num_us=None,
               ksize=3, width=64, num_convs=2, output_type="linear",
               normalize=True, normalization_type="batch", pooling="max",
               activation="relu"):
    super(AutoencoderLevel, self).__init__()

    self.is_last = (next_level is None)

    if self.is_last:
      self.left = ConvChain(
          num_inputs, num_outputs, ksize=ksize, width=width,
          depth=num_convs, stride=1, pad=True,
          normalize=normalize, normalization_type=normalization_type,
          output_type=output_type)
    else:
      assert num_us is not None

      self.left = ConvChain(
          num_inputs, width, ksize=ksize, width=width,
          depth=num_convs, stride=1, pad=True, normalize=normalize,
          normalization_type=normalization_type,
          output_type=activation, activation=activation)
      if pooling == "max":
        self.downsample = nn.MaxPool2d(2, 2)
      elif pooling == "average":
        self.downsample = nn.AvgPool2d(2, 2)
      elif pooling == "conv":
        self.downsample = nn.Conv2d(width, width, 2, stride=2)
      else:
        raise ValueError("unknown pooling'{}'".format(pooling))

      self.next_level = next_level
      self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
      self.right = ConvChain(
          num_us + width, num_outputs, ksize=ksize, width=width,
          depth=num_convs, stride=1, pad=True, normalize=normalize,
          normalization_type=normalization_type,
          output_type=output_type)

  def forward(self, x):
    left = self.left(x)
    if self.is_last:
      return left

    ds = self.downsample(left)
    next_level = self.next_level(ds)
    us = F.upsample(next_level, size=left.shape[-2:], mode='bilinear')
    # us = self.upsample(next_level)
    concat = th.cat([us, left], 1)
    output = self.right(concat)
    return output
