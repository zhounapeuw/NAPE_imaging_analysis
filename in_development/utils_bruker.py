# -*- coding: utf-8 -*-

import os
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import pickle
import re
import glob


def check_exist_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return path


def get_2p_time_vector(xml_path):

    root_recording_xml = ET.parse(xml_path).getroot()

    tvec = []
    for type_tag in root_recording_xml.findall('Sequence/Frame'):
        tvec.append(float(type_tag.attrib['relativeTime']))

    return tvec

# takes bruker xml data, parses for each frame's timing and cycle
def bruker_xml_make_frame_info_df(xml_path):
    xml_parse = ET.parse(xml_path).getroot()
    frame_info_df = pd.DataFrame()
    for idx, type_tag in enumerate(xml_parse.findall('Sequence/Frame')):
        # extract relative and absolute time from each frame's xml meta data
        frame_info_df.loc[idx, 'rel_time'] = float(type_tag.attrib['relativeTime'])
        frame_info_df.loc[idx, 'abs_time'] = float(type_tag.attrib['absoluteTime'])

        # grab cycle number from frame's name
        frame_fname = type_tag.findall('File')[0].attrib['filename']
        frame_info_df.loc[idx, 'cycle_num'] = int(re.findall('Cycle(\d+)', frame_fname)[0])
    return frame_info_df


def bruker_xml_get_2p_fs(xml_path):
    xml_parse = ET.parse(xml_path).getroot()
    for child in list(xml_parse.findall('PVStateShard')[0]):
        if 'framePeriod' in ET.tostring(child):
            return 1.0/float(child.attrib['value'])


# function for finding the index of the closest entry in an array to a provided value
def find_nearest_idx(array, value):

    if isinstance(array, pd.Series):
        idx = (np.abs(array - value)).idxmin()
        return idx, array.index.get_loc(idx), array[idx] # series index, 0-relative index, entry value
    else:
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx, array[idx]