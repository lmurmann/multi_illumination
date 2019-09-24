import os
import io
import urllib.request
import zipfile

import tqdm
import numpy as np

from lanczos import resize_lanczos

# PIL and OpenEXR are imported lazily if needed

############################## DEV
import matplotlib.pyplot as plot
############################### END DEV


# ____________ Repository-wide functions ____________
BASE_URL = "http://vrbox.csail.mit.edu:8000/static/scratch/scenezips"
basedir = os.path.join(os.environ["HOME"], ".multilum")

def set_datapath(path):
  global basedir
  basedir = path

def ensure_basedir():
    os.makedirs(basedir, exist_ok=True)

# ____________ Image-level functions ____________
def readimage(path):
  if path.endswith("exr"):
    return readexr(path)
  elif path.endswith("jpg"):
    return readjpg(path)
  else:
    raise ValueError("Unrecognized file type for path %s" % path)

def readjpg(path):
  from PIL import Image
  return np.array(Image.open(path))

def readexr(path):
  import OpenEXR

  fh = OpenEXR.InputFile(path)
  dw = fh.header()['dataWindow']
  w, h = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

  rgb = [np.ndarray([h,w], dtype="float32", buffer=fh.channel(c)) for c in ['R', 'G', 'B']]
  ret = np.zeros([h, w, 3], dtype='float32')
  for i in [0,1,2]:
    ret[:,:,i] = rgb[i]
  return ret

def writeimage(I, path):
  if path.endswith("exr"):
    return writeexr(I, path)
  elif path.endswith("jpg"):
    return writejpg(I, path)
  else:
    raise ValueError("Unrecognized file type for path %s" % path)

def writejpg(I, path):
  from PIL import Image
  im = Image.fromarray(I)
  im.save(path, "JPEG", quality=95)

def writeexr(I, path):
  import OpenEXR
  import Imath
  h, w = I.shape[:2]
  head = OpenEXR.Header(w, h)
  head["compression"] = Imath.Compression(Imath.Compression.DWAB_COMPRESSION)
  of = OpenEXR.OutputFile(path, head)
  R, G, B = [I[:,:,c].tobytes() for c in [0,1,2]]
  of.writePixels({'R': R, 'G': G, 'B': B})


def impath(scene, dir, mip, hdr):
  return os.path.join(basedir, name(scene), "dir_%d_mip%d.%s" % (dir, mip, ext(hdr)))

def imshape(mip):
  return 4000 // 2 **mip, 6000 // 2 **mip


def probepath(scene, dir, material, size, hdr):
  return os.path.join(basedir, name(scene), "probes", "dir_%d_%s%d.%s" % (dir, material, size, ext(hdr)))


# ____________ Per-scene functions ____________

FRONTAL_DIRECTIONS = [2, 3, 19, 20, 21, 22, 24]
NOFRONTAL_DIRECTIONS = [i for i in range(25) if i not in FRONTAL_DIRECTIONS]

def has_larger_version(scene, mip, hdr):
  return get_larger_version(scene, mip, hdr) != -1

def get_larger_version(scene, mip, hdr):
  for testmip in range(mip-1, -1, -1):
    if scene_is_downloaded(scene, testmip, hdr):
      return testmip
  return -1

def generate_mipmap(scene, mip, hdr):
  print("generating mipmap %d for scene %s/hdr=%d" % (mip, scene, hdr))
  srcmip = get_larger_version(scene, mip, hdr)
  if srcmip == -1:
    raise ValueError("Cannot generate mip level %d for scene %s" % (mip, scene))

  outh, outw = imshape(mip)
  for dir in range(25):
    I = readimage(impath(scene, dir, srcmip, hdr))
    I = resize_lanczos(I, outh, outw)
    writeimage(I, impath(scene, dir, mip, hdr))



def scene_is_downloaded(scene, mip, hdr):
  testfile = impath(scene, 0, mip, hdr)
  return os.path.isfile(testfile)

def probe_is_downloaded(scene, material, size, hdr):
  testfile = probepath(scene, 0, material, size, hdr)
  # print("testing file", testfile)
  return os.path.isfile(testfile)

def download_scenes(scenes=None, *, mip=2, hdr=False, force=False):
  def download_scene(scene):
    fmt = "exr" if hdr else "jpg"
    url = BASE_URL + "/%s_mip%d_%s.zip" % (scene, mip, fmt)
    req = urllib.request.urlopen(url)
    archive = zipfile.ZipFile(io.BytesIO(req.read()))
    archive.extractall(basedir)

  iterator = tqdm.tqdm(scenes) if len(scenes) > 1 else scenes

  for scene in iterator:
    scene = name(scene)

    if force or not scene_is_downloaded(scene, mip, hdr):
      download_scene(scene)

def ensure_downloaded(scene, mip, hdr):
  # from pdb import set_trace as st
  # st()
  if not scene_is_downloaded(scene, mip, hdr):
    if has_larger_version(scene, mip, hdr):
      generate_mipmap(scene, mip, hdr)
    else:

      download_scenes([scene], mip=mip, hdr=hdr)

def ensure_probe_downloaded(scene, material, size, hdr):
  if not probe_is_downloaded(scene, material, size, hdr):
    raise ValueError("Do not have %s probe (size=%d, hdr=%d) for scene %s" % (material, size, hdr, scene))


def all_scenes():
  pass

def test_scene_names():
  pass

def train_scene_names():
  pass

# ____________ Utility functions ____________
def name(obj_or_name):
  if isinstance(obj_or_name, str):
    return obj_or_name
  else:
    return obj_or_name.name

def dtype(hdr):
  return 'float32' if hdr else 'uint8'

def ext(hdr):
  return "exr" if hdr else "jpg"

def sanitize_scenes_arg(scenes):
  if scenes is None:
    scenes = all_scenes()
  elif isinstance(scenes, (str, Scene)):
    scenes = [scenes]
  return scenes

def sanitize_dirs_arg(dirs):
  if dirs is None:
    dirs = list(range(25))
  elif isinstance(dirs, int):
    dirs = [dirs]
  return dirs


# ____________ Exported interface ____________
def query_images(scenes=None, dirs=None, *, mip=2, hdr=False):
  scenes = sanitize_scenes_arg(scenes)
  dirs = sanitize_dirs_arg(dirs)

  h, w = imshape(mip)
  ret = np.zeros([len(scenes), len(dirs), h, w, 3], dtype=dtype(hdr))

  i = 0
  for iscene, scene in enumerate(scenes):
    for idir, dir in enumerate(dirs):
      ensure_downloaded(scene, mip=mip, hdr=hdr)
      ret[iscene, idir] = readimage(impath(scene, dir, mip, hdr))
  return ret

def query_probes(scenes=None, dirs=None, material="chrome", *, size=256, hdr=False):
  scenes = sanitize_scenes_arg(scenes)
  dirs = sanitize_dirs_arg(dirs)


  h, w = size, size

  ret = np.zeros([len(scenes), len(dirs), h, w, 3], dtype=dtype(hdr))

  i = 0
  for iscene, scene in enumerate(scenes):
    for idir, dir in enumerate(dirs):
      ensure_probe_downloaded(scene, material=material, size=size, hdr=hdr)
      ret[iscene, idir] = readimage(probepath(scene, dir, material, size, hdr))
  return ret

# ____________ MAIN STUB FOR DEVELOPMENT ____________


def main():
  from lum import plt
  mip=2
  # scene_names = ["main_admin1", "summer_bathroom8"]
  hdr=True
  # download_scenes(scene_names, hdr=hdr)

  scene_names = ["main_admin1", "summer_bathroom8", "summer_bathroom7"]

  # download_scenes(scene_names, mip=mip, hdr=hdr, force=True)

  I = query_images(scene_names, dirs=NOFRONTAL_DIRECTIONS, hdr=hdr, mip=mip)
  P = query_probes(scene_names, dirs=NOFRONTAL_DIRECTIONS, hdr=hdr)
  print("I shape", I.shape)
  print("P shape", P.shape)
  for scene, probe in zip(I, P):
    for imdir, probedir in zip(scene, probe):
      plt.subplot(121)
      plt.cshow(imdir)
      plt.subplot(122)
      plt.cshow(probedir)
      plt.show()
      break



# ____________ STUBS NOT YET IMPLEMENTED ____________


def query_scenes(scenes=None, locations=None, room_types=None):
  scenes = sanitize_scenes_arg(scenes)

  if locations is None:
    scenes = all_locations()

  if room_types is None:
    room_types = all_room_types()


class Scene:
  def images(self, dirs=None, *, mip=2, hdr=False):
    pass

class Location:
  def images(self, scenes=None, *, dirs=None, mip=2, hdr=False):
    pass

  def scenes(self):
    pass

class RoomType:
  def images(self, dirs=None, *, mip=2, hdr=False):
    pass
  def scenes(self):
    pass




if __name__ == "__main__":
  main()