# -*- coding: utf-8 -*-

import os
import sima
from sima.ROI import ROIList
import numpy as np


def extract(fpath):

    fdir = os.path.split(fpath)[0]
    fname = os.path.splitext(os.path.split(fpath)[1])[0]

    sima_mc_path = os.path.join(fdir, fname + '_mc.sima')

    if not os.path.exists(sima_mc_path):
        raise Exception('Data not motion corrected yet; can\'t extract ROI data')

    rois = ROIList.load(os.path.join(fdir, fname + '_RoiSet.zip'),
                        fmt='ImageJ')  # load ROIs as sima polygon objects (list)
    dataset = sima.ImagingDataset.load(os.path.join(fdir, fname + '_mc.sima'))  # reload motion-corrected dataset
    dataset.add_ROIs(rois, 'from_ImageJ')
    print('Extracting roi signals from %s' % fdir)
    signals = dataset.extract(rois)
    extracted_signals = np.asarray(signals['raw'])  # turn signals list into an np array
    np.save(os.path.join(fdir, fname + '_extractedsignals.npy'), extracted_signals)

    print('Done with extracting roi signals')