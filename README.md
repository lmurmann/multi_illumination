# Python SDK for "A Dataset of Multi-Illumination Images in the Wild" (ICCV 2019)

The SDK provides access to image downloads, image resizing, pre-processed light probes, material masks, and scene meta data. For batch data download, paper download and more please visit  https://projects.csail.mit.edu/illumination.


Below are a example uses of the SDK.
```
Load subset of scenes like this
I = query_images(['main_experiment120', 'kingston_dining10'])

# Load all test images in low resolution
I = query_images(test_scenes(), mip=5)

# Load light direction 0 in HDR floating point
I = query_images(test_scenes(), dirs=0, mip=5, hdr=True)

# Get matching light probes
P = query_probes(test_scenes())

# ... and material annotations
M = query_materials(test_scenes(), mip=5)

# List all kitchen scene meta data
K = query_scenes(room_types=['kitchen'])

# List all kitchen scenes in the training set
K = query_scenes(train_scenes(), room_types=['kitchen'])

# List all room types
T = query_room_types()
```
