import numpy as np
import unittest
import sima_extract_roi_sig
import os
from sima.ROI import ROIList

class TestExtract(unittest.TestCase):

    def setUp(self):
        self.fdir = r'C:\Users\stuberadmin\Documents\GitHub\NAPE_imaging_analysis\sample_data\VJ_OFCVTA_7_260_D6_offset'
        self.fname = 'VJ_OFCVTA_7_260_D6_offset'

        self.sig = np.load(os.path.join(self.fdir, self.fname) + '_extractedsignals.npy')

    def test_extract_roi(self):
        # make sure number of loaded ROIs (from zip and extracted signals) are correct
        rois = ROIList.load(os.path.join(self.fdir, self.fname + '_RoiSet.zip'),
                            fmt='ImageJ')  # load ROIs as sima polygon objects (list)

        assert self.sig.shape[1] == 11  # CZ manually drew 11 ROIs
        assert len(rois) == 11

    def test_extract_signals(self):
        sig_test = np.load(os.path.join(self.fdir, self.fname) + '_extractedsignals_test.npy')

        assert self.sig.all() == sig_test.all()


if __name__ == "__main__":
    unittest.main()
    print("Everything passed")