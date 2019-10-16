import os
import io
import urllib.request
import zipfile
from collections.abc import Iterable

import tqdm
import numpy as np

from lanczos import resize_lanczos
import json
from collections import namedtuple
# PIL and OpenEXR are imported lazily if needed


# ____________ Repository-wide functions ____________
BASE_URL = "https://data.csail.mit.edu/multilum"
basedir = os.path.join(os.environ["HOME"], ".multilum")

def set_datapath(path):
  """Call this function before any other call to change the data directory.
  """
  global basedir
  basedir = path

# ____________ Image-level functions ____________
def readimage(path):
  """Generic image read helper"""
  if path.endswith("exr"):
    return readexr(path)
  elif path.endswith("jpg"):
    return readjpg(path)
  else:
    raise ValueError("Unrecognized file type for path %s" % path)

def readjpg(path):
  """JPG read helper. Requires PIL."""
  from PIL import Image
  return np.array(Image.open(path))

def readexr(path):
  """EXR read helper. Requires OpenEXR."""
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
  """Generic image write helper"""
  if path.endswith("exr"):
    return writeexr(I, path)
  elif path.endswith("jpg"):
    return writejpg(I, path)
  else:
    raise ValueError("Unrecognized file type for path %s" % path)

def writejpg(I, path):
  """JPG write helper. Requires PIL."""

  from PIL import Image
  im = Image.fromarray(I)
  im.save(path, "JPEG", quality=95)

def writeexr(I, path):
  """EXR write helper. Requires OpenEXR."""
  import OpenEXR
  import Imath
  h, w = I.shape[:2]
  head = OpenEXR.Header(w, h)
  head["compression"] = Imath.Compression(Imath.Compression.DWAB_COMPRESSION)
  of = OpenEXR.OutputFile(path, head)
  R, G, B = [I[:,:,c].tobytes() for c in [0,1,2]]
  of.writePixels({'R': R, 'G': G, 'B': B})


def impath(scene, dir, mip, hdr):
  """Generate path for image
  
  Args:
    scene: scene name
    dir: direction number
    mip (int): mip level
    hdr (bool): generate path for HDR or not
  """
  return os.path.join(basedir, name(scene), "dir_%d_mip%d.%s" % (dir, mip, ext(hdr)))

def imshape(mip):
  """Compute image size for different mip levels"""
  return 4000 // 2 **mip, 6000 // 2 **mip


def probepath(scene, dir, material, size, hdr):
  """Compute path for light probes
  
  Args:
    scene: scene name
    dir: direction number
    material: One of "gray" or "chrome"
    size (int): set this to 256
    hdr (bool): generate path for HDR or not
  """
  return os.path.join(basedir, name(scene), "probes", "dir_%d_%s%d.%s" % (dir, material, size, ext(hdr)))


# ____________ Per-scene functions ____________

"""Directions where the flash is directly visible"""
FRONTAL_DIRECTIONS = [2, 3, 19, 20, 21, 22, 24]

"""Directions where the flash is only visible indirectly"""
NOFRONTAL_DIRECTIONS = [i for i in range(25) if i not in FRONTAL_DIRECTIONS]

def has_larger_version(scene, mip, hdr):
  """Helper that returns whether a larger (lower) mip of a scene is downloaded
  """
  return get_larger_version(scene, mip, hdr) != -1

def get_larger_version(scene, mip, hdr):
  """Return index of next-largest miplevel

  Args:
    scene: scene name
    mip (int): mip that we want to generate
    hdr (bool): HDR or not

  Returns:
    integer mip level that exists on disk or -1 if no larger mip exists.
  """
  for testmip in range(mip-1, -1, -1):
    if scene_is_downloaded(scene, testmip, hdr):
      return testmip
  return -1

def generate_mipmap(scene, mip, hdr):
  """Generate single mip level for one scene.

  Args:
    scene: scene name
    mip (int): mip that we want to generate
    hdr (bool): HDR or not

  Raises:
    ValueError: If no larger version of the image exists.
  """
  print("generating mipmap %d for scene %s/hdr=%d" % (mip, scene, hdr))
  srcmip = get_larger_version(scene, mip, hdr)
  if srcmip == -1:
    raise ValueError("Cannot generate mip level %d for scene %s" % (mip, scene))

  outh, outw = imshape(mip)
  for dir in range(25):
    I = readimage(impath(scene, dir, srcmip, hdr))
    I = resize_lanczos(I, outh, outw)
    writeimage(I, impath(scene, dir, mip, hdr))

def generate_probe_size(scene, material, size, hdr):
  """Generate downsampled light probe.

  Requires that the original 256px resolution has already been downloaded.

  Args:
    scene: scene name
    material: "gray" or "chrome"
    size: target size to generate
    hdr: HDR or not
  """
  if size >= 256:
    raise ValueError("Can only generate probes that are smaller than 256px")

  print("generating %s probe size %d for scene %s/hdr=%d" % (material, size, scene, hdr))
  for dir in range(25):
    I = readimage(probepath(scene, dir, "chrome", 256, hdr))
    I = resize_lanczos(I, size, size)
    writeimage(I, probepath(scene, dir, "chrome", size, hdr))

def scene_is_downloaded(scene, mip, hdr):
  """Tests whether scene exists on disk as a particular (mip/hdr) version.

  Args:
    scene: scene name
    mip (int): mip that we want to generate
    hdr (bool): HDR or not

  Returns:
    bool: True if scene at given mip/hdr exists
  """
  testfile = impath(scene, 24, mip, hdr)
  return os.path.isfile(testfile)

def probe_is_downloaded(scene, material, size, hdr):
  """Tests whether probe exists on disk as a particular (size/hdr) version.
 
  Args:
    scene: scene name
    material: "gray" or "chrome"
    size: target size to generate
    hdr: HDR or not    

  Returns:
    bool: True if scene at given mip/hdr exists
  """
  testfile = probepath(scene, 24, material, size, hdr)
  return os.path.isfile(testfile)

def download_scenes(scenes=None, *, mip=2, hdr=False, force=False):
  """Download and unzip a list of scenes
  
  Args:
    scenes: list of scenes or scene names
    mip(int): mip level to download
    hdr(bool): whether to download JPG or EXR files
    force(bool): force download even if scene already exists on disk.
  """
  def download_scene(scene):
    fmt = "exr" if hdr else "jpg"
    url = BASE_URL + "/%s/%s_mip%d_%s.zip" % (scene, scene, mip, fmt)
    req = urllib.request.urlopen(url)
    archive = zipfile.ZipFile(io.BytesIO(req.read()))
    archive.extractall(basedir)

  print("Downloading %d scenes" % len(scenes))

  iterator = tqdm.tqdm(scenes) if len(scenes) > 1 else scenes
  for scene in iterator:
    scene = name(scene)

    if force or not scene_is_downloaded(scene, mip, hdr):
      download_scene(scene)

def ensure_downloaded(scenes, mip, hdr):
  """Download scenes (or generate from larger version) if needed
  
  Args:
    scenes: list of scenes or scene names
    mip(int): mip level to download
    hdr(bool): whether to download JPG or EXR files
  """
  if not isinstance(scenes, Iterable):
    scenes = [scenes]

  not_downloaded = []

  for scene in scenes:
    if not scene_is_downloaded(scene, mip, hdr):
      if has_larger_version(scene, mip, hdr):
        generate_mipmap(scene, mip, hdr)
      else:
        not_downloaded.append(scene)
  if not_downloaded:
    download_scenes(not_downloaded, mip=mip, hdr=hdr)

def ensure_probe_downloaded(scene, material, size, hdr):
  """Download light probes (or generate from larger version) if needed
  
  Args:
    scenes: list of scenes or scene names
    material(string): "gray" or "chrome"
    size(int): size in pixels of the requested probe set
    hdr(bool): whether to download JPG or EXR files
  """
  if not probe_is_downloaded(scene, material, size, hdr):
    ensure_downloaded(scene, 2, hdr)
    generate_probe_size(scene, material, size, hdr)

# ____________ Meta data functions ____________
Scene    = namedtuple('Scene', ['name', 'room'])
Room     = namedtuple('Room', ['name', 'building', 'room_type'])
RoomType = namedtuple('RoomType', ['name'])
Building = namedtuple('Building', ['name'])

_scene_json = None
def all_scenes():
  """List all scenes
  
  Returns:
    list of Scene objects
  """

  global _scene_json
  if _scene_json == None:
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "scenes.json")) as fh:
      _scene_json = json.loads(fh.read())
  
  ret = []
  for s in _scene_json:
    ret.append(Scene(s["name"], 
                Room(s["room"], 
                  Building(s["building"]), 
                  RoomType(s["room_type"]))))
  return ret

def all_buildings():
  """List all buildings
  
  Returns:
    list of Building objects
  """
  for scene in all_scenes():
    yield scene.room.building

def all_rooms():
  """List all rooms
  
  Returns:
    list of Room objects
  """
  for scene in all_scenes():
    yield scene.room

def all_room_types():
  """List all room types
  
  Returns:
    list of RoomType objects
  """
  for scene in all_scenes():
    yield scene.room.room_type

def query_scenes(scenes=None, *, buildings=None, rooms=None, room_types=None):
  """Query subset of scenes

  Args:
    scenes: list of scene names or scene objects
    buildings: list of building names or building objects
    rooms: list of room names "<Building>/<RoomName>" or room objects
    room_types: list of room types, e.g. ["kitchen", "basement"]
  
  Returns:
    list of Scene objects
  """
  scene_names = {name(s) for s in sanitize_scenes_arg(scenes)}
  building_names = {name(b) for b in sanitize_buildings_arg(buildings)}
  room_type_names = {name(rt) for rt in sanitize_room_types_arg(room_types)}
  room_ids = {rid for rid in sanitize_rooms_arg_to_ids(rooms)}

  ret = []
  for scene in all_scenes():
    scene_name, room_name, building_name, room_type_name = \
      scene.name, scene.room.name, scene.room.building.name, scene.room.room_type.name
    if scene_name in scene_names and \
        "%s/%s" % (building_name, room_name) in room_ids and \
        building_name in building_names and \
        room_type_name in room_type_names:
      ret.append(scene)
  return ret

def query_buildings(buildings=None):
  """Query subset of buildings

  Args:
    buildings: list of building names or building objects, e.g. ["elm", "main"]

  Returns:
    list of Building objects
  """
  building_names = {name(b) for b in sanitize_buildings_arg(buildings)}
  
  ret = []
  for b in all_buildings():
    if b.name in building_names:
      ret.append(b)
  return ret 


def query_rooms(rooms=None, *, room_types=None, buildings=None):
  """Query subset of rooms

  Args:
    rooms: list of room names, e.g. "everett/kitchen", or room objects
    buildings: list of building names or building objects, e.g. ["elm", "main"]
    room_types: list of room types, e.g. ["kitchen", "basement"]
  
  Returns:
    list of Room objects
  """
  building_names = {name(b) for b in sanitize_buildings_arg(buildings)}
  room_type_names = {name(rt) for rt in sanitize_room_types_arg(room_types)}
  room_ids = {rid for rid in sanitize_rooms_arg_to_ids(rooms)}

  ret = []
  for r in all_rooms():
    room_name, building_name, room_type_name = r.name, r.building.name, r.room_type.name
    if  "%s/%s" % (building_name, room_name) in room_ids and \
        building_name in building_names and \
        room_type_name in room_type_names:
      ret.append(r)
  return ret

def query_room_types(room_types=None):
  """Query subset of room types

  Args:
    room_types: list of room types, e.g. ["kitchen", "basement"]
  
  Returns:
    list of RoomType objects
  """
  room_type_names = {name(rt) for rt in sanitize_room_types_arg(room_types)}

  ret = []
  for rt in all_room_types():
    if rt.name in room_type_names:
      ret.append(rt)
  return ret

def test_scenes():
  """Return all scenes of the test set"""
  return [s for s in all_scenes() if s.name.startswith('everett')]

def train_scenes():
  """Return all scenes of the training set"""
  return [s for s in all_scenes() if not s.name.startswith('everett')]

# ____________ Utility functions ____________
def name(obj_or_name):
  if isinstance(obj_or_name, str):
    return obj_or_name
  else:
    return obj_or_name.name

def id(obj_or_id):
  if isinstance(obj_or_id, int):
    return obj_or_id
  else:
    return obj_or_id.id

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

def sanitize_room_types_arg(room_types):
  if room_types is None:
    room_types = all_room_types()
  elif isinstance(room_types, (str, RoomType)):
    room_types = [room_types]
  return room_types

def sanitize_buildings_arg(buildings):
  if buildings is None:
    buildings = all_buildings()
  elif isinstance(buildings, (str, Building)):
    buildings = [buildings]
  return buildings

def sanitize_rooms_arg_to_ids(rooms):
  if rooms is None:
    rooms = all_rooms()
  elif isinstance(rooms, (str, Room)):
    rooms = [rooms]
  
  ret = []
  for room in rooms:
    if isinstance(room, Room):
      room = "%s/%s" % (room.building.name, room.name)
    ret.append(room)
  return ret

def sanitize_dirs_arg(dirs):
  if dirs is None:
    dirs = list(range(25))
  elif isinstance(dirs, int):
    dirs = [dirs]
  return dirs


# ____________ Image Query Functions ____________
def query_images(scenes=None, dirs=None, *, mip=2, hdr=False):
  """Return a 5D tensor if images

  Args:
    scenes: list of scenes (name or object) or None for all scenes
    dirs: list of integer indices or None for all directions. Can use FRONTAL_DIRECTIONS and
NOFRONAL_DIRECTIONS directions
    mip: mip level index. smaller mip levels mean larger images. It is recommended to work
with mip=2 (1500x1000px) for most applications. Set to mip=0 for high resolution
(6000x4000px) images.
    hdr: boolean flag that selects between 8-bit images or linear HDR images.

  Returns
    5D numpy array with shape (num_scenes, num_dirs, height, width, 3). The dattype of the 
returned array is uint8 for hdr=False, float32 for hdr=True
  """

  scenes = sanitize_scenes_arg(scenes)
  dirs = sanitize_dirs_arg(dirs)

  h, w = imshape(mip)
  ret = np.zeros([len(scenes), len(dirs), h, w, 3], dtype=dtype(hdr))

  ensure_downloaded(scenes, mip=mip, hdr=hdr)
  for iscene, scene in enumerate(scenes):
    for idir, dir in enumerate(dirs):
      ret[iscene, idir] = readimage(impath(scene, dir, mip, hdr))
  return ret

def query_probes(scenes=None, dirs=None, material="chrome", *, size=256, hdr=False):
  """Return a 5D tensor if images

  Args:
    scenes: list of scenes (name or object) or None for all scenes
    dirs: list of integer indices or None for all directions. Can use FRONTAL_DIRECTIONS and
NOFRONAL_DIRECTIONS directions
    material: "chrome" for images of the chrome ball. "gray" for plastic gray ball
    size(int): size in pixels that will be loaded
    hdr(bool): boolean flag that selects between 8-bit images or linear HDR images.

  Returns
    5D numpy array with shape (num_scenes, num_dirs, size, size, 3). The dattype of the 
returned array is uint8 for hdr=False, float32 for hdr=True
  """
  scenes = sanitize_scenes_arg(scenes)
  dirs = sanitize_dirs_arg(dirs)


  h, w = size, size

  ret = np.zeros([len(scenes), len(dirs), h, w, 3], dtype=dtype(hdr))

  for iscene, scene in enumerate(scenes):
    for idir, dir in enumerate(dirs):
      ensure_probe_downloaded(scene, material=material, size=size, hdr=hdr)
      ret[iscene, idir] = readimage(probepath(scene, dir, material, size, hdr))
  return ret

# ____________ MAIN STUB FOR DEVELOPMENT ____________


def main():
  print("Buildings")
  for loc in all_buildings():
    print(loc)

  print("Rooms")
  for r in all_rooms():
    print(r)

  print("Room Types")
  for rt in all_room_types():
    print(rt)

  print("Scenes")
  for s in query_scenes():
    print(s)
  # return


  from matplotlib import pyplot as plt
  # download_scenes(all_scenes(), mip=2, hdr=False)
  # return

  # scenes = all_scenes()
  scenes = ['main_experiment120', 'kingston_dining10', 'elm_revis_kitchen14']

  # download_scenes(scene_names, mip=mip, hdr=hdr, force=True)

  print("get images")
  I = query_images(scenes, mip=4, hdr=True)

  # print("get probes")
  # P = query_probes(scenes)
  # print("P shape", P.shape)

  # (3 scenes, 25 light directions, height, width, rgb)
  print("I shape", I.shape)

  dir1 = 14
  dir2 = 24

  for i in range(3):
    plt.subplot(2,3,i+1)
    plt.imshow(I[i,dir1])
    plt.subplot(2,3,i+4)
    plt.imshow(I[i,dir2])

  plt.show()

if __name__ == "__main__":
  main()