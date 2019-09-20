import os
import io
import urllib.request
import zipfile

import tqdm
import numpy as np

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
  ret = np.zeros([1000, 1500, 3], dtype='float32')
  for i in [0,1,2]:
    ret[:,:,i] = rgb[i]
  return ret

def impath(scene, dir, mip, hdr):
  return os.path.join(basedir, name(scene), "dir_%d_mip%d.%s" % (dir, mip, ext(hdr)))


# ____________ Per-scene functions ____________
def scene_is_downloaded(scene, mip, hdr):
  testfile = impath(scene, 0, mip, hdr)
  return os.path.isfile(testfile)

def download_scenes(scenes=None, *, mip=2, hdr=False, force=False):
  def download_scene(scene):
    fmt = "exr" if hdr else "jpg"
    url = BASE_URL + "/%s_mip2_%s.zip" % (scene, fmt)
    req = urllib.request.urlopen(url)
    archive = zipfile.ZipFile(io.BytesIO(req.read()))
    archive.extractall(basedir)

  iterator = tqdm.tqdm(scenes) if len(scenes) > 1 else scenes

  for scene in iterator:
    scene = name(scene)

    if force or not scene_is_downloaded(scene, mip, hdr):
      download_scene(scene)

def ensure_downloaded(scene, mip, hdr):
  if not scene_is_downloaded(scene, mip, hdr):
    download_scenes([scene], mip=mip, hdr=hdr)

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


  h = 4000 // 2 **mip
  w = 6000 // 2 **mip

  ret = np.zeros([len(scenes), len(dirs), h, w, 3], dtype=dtype(hdr))

  i = 0
  for iscene, scene in enumerate(scenes):
    for idir, dir in enumerate(dirs):
      ensure_downloaded(scene, mip=mip, hdr=hdr)
      ret[iscene, idir] = readimage(impath(scene, dir, mip, hdr))
  return ret



# ____________ MAIN STUB FOR DEVELOPMENT ____________


def main():
  scene_names = ["main_admin1", "summer_bathroom8"]
  hdr=False
  download_scenes(scene_names, hdr=hdr)

  scene_names = ["main_admin1", "summer_bathroom8", "summer_bathroom7"]

  I = query_images(scene_names, hdr=hdr)
  print(I.shape)
  for scene in I:
    for dir in scene:
      plt.imshow(dir)
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