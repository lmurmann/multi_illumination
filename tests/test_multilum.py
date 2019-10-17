import os
import unittest
import tempfile
import shutil

import numpy as np

import multilum
from multilum import query_scenes, query_rooms, query_images
from multilum import query_materials, query_probes

class TestMetaData(unittest.TestCase):

    def test_query_scenes(self):
        self.assertEqual(len(query_scenes()), 1015)
        self.assertEqual(len(query_scenes(scenes='everett_kitchen2')), 1)
        self.assertEqual(len(query_scenes(scenes='everett_kitchen2',
                                          room_types='bathroom')), 0)
        self.assertEqual(len(query_scenes(scenes=['elm_basebath2', 'elm_basebath8', 'elm_kitchen2'],
                                          room_types='bathroom')), 2)
        self.assertEqual(len(query_scenes(buildings=['everett', 'summer'])), 90)

        self.assertEqual(query_scenes('everett_kitchen2')[0].room,
                         query_rooms("everett/kitchen")[0])


def mse(A, B):
    if A.dtype == 'uint8':
        A = A / 255
    if B.dtype == 'uint8':
        B = B / 255

    diff = A - B
    return np.sum(diff*diff) / np.prod(diff.shape)


class TestImageLoader(unittest.TestCase):

    def setUp(self):
        test_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        self.tmpdir = tempfile.mkdtemp()
        tmp_data = os.path.join(self.tmpdir, 'data')
        shutil.copytree(test_src, tmp_data)
        multilum.set_datapath(tmp_data)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_query_images(self):
        mip=6
        h, w = multilum.imshape(mip)

        I = query_images('everett_kitchen2', dirs=[1,5,7, 9], mip=mip)
        self.assertEqual(I.shape, (1, 4, h, w, 3))

    def test_gen_mipmaps(self):
        generated_mip = query_images('everett_kitchen2', dirs=0, mip=7)[0,0]
        golden_mip = query_images('everett_kitchen2_golden', dirs=0, mip=7)[0,0]
        self.assertLess(mse(generated_mip, golden_mip), 1e-3)

    def test_gen_probes(self):
        """Test generating a 16px chrome ball from 32px base image.

        The ground truth 16px reference is stored in the "golden" directory.
        """
        self.assertTrue(multilum.probe_is_downloaded(
            'everett_kitchen2', material='chrome', size=32, hdr=False))
        self.assertTrue(multilum.probe_is_downloaded(
            'everett_kitchen2_golden', material='chrome', size=16, hdr=False))

        multilum.generate_probe_size(['everett_kitchen2'],
            material='chrome', size=16, hdr=False, base_size=32)

        self.assertTrue(multilum.probe_is_downloaded(
            'everett_kitchen2', material='chrome', size=16, hdr=False))

        generated_probe  = query_probes('everett_kitchen2', size=16)[0,0]
        golden_probe = query_probes('everett_kitchen2_golden', size=16)[0,0]
        self.assertLess(mse(generated_probe, golden_probe), 1e-3)


class TestMaterialLoader(unittest.TestCase):
    def test_query_materials(self):
        mip = 4
        h, w = multilum.imshape(mip)
        M = query_materials('everett_kitchen2', mip=mip)
        self.assertEqual(M.shape, (1, h, w))

        M = query_materials('everett_kitchen2', mip=mip, apply_colors=True)
        self.assertEqual(M.shape, (1, h, w, 3))


if __name__ == '__main__':
    unittest.main()
