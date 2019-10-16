import os
import unittest
import tempfile
import shutil

import multilum
from multilum import query_scenes, query_rooms, query_images
import numpy as np

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
        I = query_images('everett_kitchen2', dirs=[1,5,7, 9], mip=6)
        self.assertEqual(I.shape, (1, 4, 4000//2**6, 6000//2**6, 3))
        
    def test_gen_mipmaps(self):
        generated_mip = query_images('everett_kitchen2', dirs=0, mip=7)[0,0]
        golden_mip = query_images('everett_kitchen2_mip7', dirs=0, mip=7)[0,0]
        
        diff = generated_mip/255 - golden_mip/255 
        mse = np.sum(diff*diff) / np.prod(diff.shape)
        self.assertLess(mse, 1e-3)


if __name__ == '__main__':
    unittest.main()
