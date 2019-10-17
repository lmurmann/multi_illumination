"""Python SDK for "A Multi-Illumination Dataset of Indoor Object Appearance"

The SDK provides access to image downloads, image resizing, pre-processed light probes,
material annotations, and scene meta data, such as room types.

Below are a example uses of the SDK

Load subset of scenes like this
I = query_images(['main_experiment120', 'kingston_dining10'])

Load all test images in low resolution
I = query_images(test_scenes(), mip=5)

Load light direction 0 in HDR floating point
I = query_images(test_scenes(), dirs=[0], mip=5, hdr=True)

Get matching light probes
P = query_probes(test_scenes())

And material annotations
M = query_materials(test_scenes(), mip=5)

List all kitchen scenes
K = query_scenes(room_types=['kitchen'])

List all kitchen scenes in the training set
K = query_scenes(train_scenes(), room_types=['kitchen'])

List all room types
T = query_room_types()

Batch data download, paper download and more:
https://projects.csail.mit.edu/illumination
"""


from collections.abc import Iterable
from collections import namedtuple
import io
import json
import os
import urllib.request
import zipfile

import numpy as np
import tqdm

from lanczos import resize_lanczos

# PIL and OpenEXR are imported lazily if needed


# ____________ Repository-wide functions ____________
BASE_URL = "https://data.csail.mit.edu/multilum"
basedir = os.path.join(os.environ["HOME"], ".multilum")

def set_datapath(path):
  """Call this function before any other call to change the data directory.
  """
  global basedir
  basedir = path


# ____________ Image Query Functions ____________
# Directions where the flash is directly visible
FRONTAL_DIRECTIONS = [2, 3, 19, 20, 21, 22, 24]

# Directions where the flash is only visible indirectly
NOFRONTAL_DIRECTIONS = [i for i in range(25) if i not in FRONTAL_DIRECTIONS]


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

  ensure_probes_downloaded(scenes, material=material, size=size, hdr=hdr)
  for iscene, scene in enumerate(scenes):
    for idir, dir in enumerate(dirs):
      ret[iscene, idir] = readimage(probepath(scene, dir, material, size, hdr))
  return ret


def query_materials(scenes=None, *, mip=2, apply_colors=False):
  """Return a numpy array of material masks

  Args:
    scenes: list of scenes (name or object) or None for all scenes
    mip: mip level index. smaller mip levels mean larger images. It is recommended to work
with mip=2 (1500x1000px) for most applications. Set to mip=0 for high resolution
(6000x4000px) images.
    apply_colors: If true, returns RGB masks as used in the paper. If false, returns scalar
integer indices.

  Returns
    if apply_colors is False, returns 3D numpy array with shape (num_scenes, height, width). if
apply_colors is True, returns 4D array with shape (num_scenes, height, width, 3). Returned array
is always type uint8.
  """
  scenes = sanitize_scenes_arg(scenes)

  h, w = imshape(mip)
  if apply_colors:
    shape = [len(scenes), h, w, 3]
  else:
    shape = [len(scenes), h, w]
  ret = np.zeros(shape, dtype='uint8')

  ensure_materials_downloaded(scenes, mip=mip)
  for iscene, scene in enumerate(scenes):
    ret[iscene] = read_material_image(material_impath(scene, mip=mip), apply_colors=apply_colors)
  return ret


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
  if _scene_json is None:
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
  retset = set()
  for scene in all_scenes():
    retset.add(scene.room.building)
  return list(retset)

def all_rooms():
  """List all rooms

  Returns:
    list of Room objects
  """
  retset = set()
  for scene in all_scenes():
    retset.add(scene.room)
  return list(retset)

def all_room_types():
  """List all room types

  Returns:
    list of RoomType objects
  """
  retset = set()
  for scene in all_scenes():
    retset.add(scene.room.room_type)
  return list(retset)

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
  room_ids = set(sanitize_rooms_arg_to_ids(rooms))

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
  room_ids = set(sanitize_rooms_arg_to_ids(rooms))

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






## =========== INTERNAL / ADVANCED FUNCTIONS ======================= ##

# ____________ Image-level functions ____________
def readimage(path):
  """Generic image read helper"""
  if path.endswith("exr"):
    return readexr(path)
  elif path.endswith("jpg"):
    return readjpg(path)
  else:
    raise ValueError("Unrecognized file type for path %s" % path)

def read_material_image(path, *, apply_colors=False):
  """Read material PNG mask

  Args:
    path: image path
    apply_colors: whether to apply the embedded color palette
  """
  return readpng_indexed(path, apply_palette=apply_colors)

def readjpg(path):
  """JPG read helper. Requires PIL."""
  from PIL import Image
  return np.array(Image.open(path))

def readpng_indexed(path, *, apply_palette):
  """Indexed PNG read helper. Requires PIL."""
  from PIL import Image

  im = Image.open(path)
  if not im.mode == "P":
    raise ValueError("Expected indexed PNG")

  if apply_palette:
    im = im.convert(mode="RGB")
    shape = (im.height, im.width, 3)
  else:
    shape = (im.height, im.width)

  npim = np.ndarray(shape=shape, dtype="uint8", buffer=im.tobytes())
  # palette = np.ndarray(shape=[256,3], dtype="uint8", buffer=im.palette.getdata()[1])
  return npim

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
  """Compute path for material math

  Args:
    scene: scene name
    mip (int): mip level
  """
  return os.path.join(basedir, name(scene),
      "probes", "dir_%d_%s%d.%s" % (dir, material, size, ext(hdr)))

def material_impath(scene, mip):
  """Generate path for material map of given scene / mip level"""
  return os.path.join(basedir, name(scene), "materials_mip%d.png" % (mip))

# ____________ Per-scene functions ____________

def scenepath(scene):
  """Generate path for scene directory"""
  return os.path.join(basedir, name(scene))

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

def generate_probe_size(scenes, *, material, size, base_size, hdr):
  """Generate downsampled light probe.

  Requires that the original 256px resolution has already been downloaded.

  Args:
    scene: scene name
    material: "gray" or "chrome"
    size: target size to generate
    hdr: HDR or not
  """
  if size == base_size:
    return
  elif size > base_size:
    raise ValueError("Can only generate probes that are smaller than 256px")

  print("generating %s probe size %d/hdr=%d for %d scenes" % (material, size,  hdr, len(scenes)))
  iterator = tqdm.tqdm(scenes) if len(scenes) > 3 else scenes

  for scene in iterator:
    for dir in range(25):
      I = readimage(probepath(scene, dir, "chrome", base_size, hdr))
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
    bool: True if light probe at given size/hdr exists
  """
  testfile = probepath(scene, 24, material, size, hdr)
  return os.path.isfile(testfile)

def material_is_downloaded(scene, mip):
  """Tests whether material map exists on disk as mip level.

  Args:
    scene: scene name
    mip (int): mip that is tested

  Returns:
    bool: True if material map at given mip exists
  """
  testfile = material_impath(scene, mip=mip)
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

def download_probes(scenes, *, material, size, hdr):
  """Download and unzip a light probes for list of scenes

  Args:
    scenes: list of scenes or scene names
    material(string): "gray" or "chrome"
    size(int): size in pixels of the requested probe set
    hdr(bool): whether to download JPG or EXR files
  """
  def download_probe(scene):
    fmt = "exr" if hdr else "jpg"
    url = BASE_URL + "/%s/%s_probes_%dpx_%s.zip" % (scene, scene, size, fmt)
    req = urllib.request.urlopen(url)
    archive = zipfile.ZipFile(io.BytesIO(req.read()))
    archive.extractall(basedir)

  print("Downloading probes for %d scenes" % len(scenes))
  iterator = tqdm.tqdm(scenes) if len(scenes) > 1 else scenes

  for scene in iterator:
    scene = name(scene)
    download_probe(scene)

def download_materials(scenes=None, *, mip):
  """Download material map PNG images

  Args:
    scenes: list of scenes or scene names
    mip(int): mip level to download
  """

  def download_materialmap(scene):
    os.makedirs(scenepath(scene), exist_ok=True)
    url = BASE_URL + "/%s/materials_mip%d.png" % (scene, mip)
    req = urllib.request.urlopen(url)
    outfile = open(material_impath(scene, mip), 'wb')
    outfile.write(req.read())
    outfile.close()

  print("Downloading %d material maps at mip %d" % (len(scenes), mip))
  iterator = tqdm.tqdm(scenes) if len(scenes) > 1 else scenes

  for scene in iterator:
    scene = name(scene)
    download_materialmap(scene)

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

def ensure_probes_downloaded(scenes, *, material, size, hdr):
  """Download light probes (or generate from larger version) if needed

  Args:
    scenes: list of scenes or scene names
    material(string): "gray" or "chrome"
    size(int): size in pixels of the requested probe set
    hdr(bool): whether to download JPG or EXR files
  """

  must_download = []
  must_generate = []
  for scene in scenes:
    probe_loaded = probe_is_downloaded(scene, material, size, hdr)
    baseprobe_loaded = probe_is_downloaded(scene, material, 256, hdr)

    if not probe_loaded:
      must_generate.append(scene)

    if not probe_loaded and not baseprobe_loaded:
      must_download.append(scene)

  if must_download:
    download_probes(must_download, material=material, size=256, hdr=hdr)

  if must_generate:
    generate_probe_size(scenes, material=material, size=size, base_size=256, hdr=hdr)

def ensure_materials_downloaded(scenes, *, mip):
  """Download material maps if needed

  Args:
    scenes: list of scenes or scene names
    mip(int): mip level to download
  """

  not_loaded = []

  for scene in scenes:
    if not material_is_downloaded(scene, mip):
      not_loaded.append(scene)

  if not_loaded:
    download_materials(not_loaded, mip=mip)

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


# ____________ MAIN FUNCTION --- Example Usage ____________


def demo_multi_illumination():
  from matplotlib import pyplot as plt

  print("=== Multi-Illumination Image Demo ===")
  scenes = ['main_experiment120', 'kingston_dining10', 'elm_revis_kitchen14']
  I = query_images(scenes, mip=4)

  dir1 = 14
  dir2 = 24

  for i in range(3):
    plt.subplot(2,3,i+1)
    plt.imshow(I[i,dir1])
    plt.subplot(2,3,i+4)
    plt.imshow(I[i,dir2])

  plt.show()

def demo_light_probes():
  from matplotlib import pyplot as plt
  scenes = ['main_experiment120', 'kingston_dining10', 'elm_revis_kitchen14']

  print("=== Light Probe Demo ===")
  P = query_probes(scenes)

  for i in range(25):
    plt.subplot(5,5,i+1)
    plt.imshow(P[0,i])
  plt.show()

def demo_materials():
  from matplotlib import pyplot as plt

  print("=== Material Demo ===")
  M = query_materials('main_experiment120', mip=4)
  print(M.shape)
  plt.subplot(121)
  plt.imshow(M[0])

  M = query_materials('main_experiment120', mip=4, apply_colors=True)
  print(M.shape)
  plt.subplot(122)
  plt.imshow(M[0])
  plt.show()

def demo_meta_data():
  print("Scenes")
  scenes = all_scenes()
  for s in scenes[:3]:
    print(s)
  print("... total %d scenes\n" % len(scenes))

  print("Buildings")
  buildings = all_buildings()
  for loc in buildings[:3]:
    print(loc)
  print("... total %d buildings\n" % len(buildings))

  print("Rooms")
  rooms = all_rooms()
  for r in rooms[:3]:
    print(r)
  print("... total %d rooms\n" % len(rooms))

  print("Room Types")
  room_types = all_room_types()
  for rt in room_types[:3]:
    print(rt)
  print("... total %d room types\n" % len(room_types))

def main():
  demo_multi_illumination()
  demo_light_probes()
  demo_materials()
  demo_meta_data()

if __name__ == "__main__":
  main()
