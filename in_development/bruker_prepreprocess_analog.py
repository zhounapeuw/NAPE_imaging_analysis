# # Meta & Behavioral Data Preprocessing

# ### Define variables and load 2p recording xml

from functools import partial
import os
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import pickle
import re
import glob

import utils_bruker


fdir = r'D:\bruker_data\Adam\2020_07_13_aac13_d01_replete\2020_07_13_aac13_d01_replete_first10'
fname = '2020_07_13_aac13_d01_replete_first10'


behav_id_of_interest = [101, 102, 103]

bruker_analog = True  # set to true if analog/voltage input signals are present and are of interest
validation_plots = False  # set to true if want to plot traces of ttl pulses for visualizing and validating
valid_plot_channel = 'input_2'  # analog dataframe column names get cleaned up; AI's are "input_#"

flag_multicondition_analog = False  # if a single analog port contains multiple conditions that need to be split up, set to true 
ai_to_split = 2  # int, analog port number that contains TTLs of multiple conditions; events here will be split into individual conditions if flag_multicondition_analog is set to true

# In[6]:


# define file paths and output file names
bruker_tseries_xml_path = os.path.join(fdir, fname + '.xml')  # recording/tseries main xml

glob_analog_csv = glob.glob(os.path.join(fdir, "*_VoltageRecording_*.csv"))  # grab all analog/voltage recording csvs
glob_analog_xml = glob.glob(
    os.path.join(fdir, "*_VoltageRecording_*.xml"))  # grab all analog/voltage recording xml meta

# behavioral event identification files
behav_fname = fname + '_taste_reactivity.csv'  # csv containing each behav event and corresponding sample
behav_event_key_path = r'D:\bruker_data\Adam\key_event.xlsx'  # location of excel matching event names and id's

behav_save_path = os.path.join(fdir, 'framenumberforevents_{}.pkl'.format(fname))
behav_analog_save_path = os.path.join(fdir, 'framenumberforevents_analog_{}.pkl'.format(fname))


# In[70]:


# load in recording/tseries main xml and grab frame period
def bruker_xml_get_2p_fs(xml_path):
    xml_parse = ET.parse(xml_path).getroot()
    for child in list(xml_parse.findall('PVStateShard')[0]):
        if 'framePeriod' in ET.tostring(child):
            return 1.0 / float(child.attrib['value'])


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


# loads and parses the analog/voltage recording's xml and grabs sampling rate
def bruker_analog_xml_get_fs(xml_fpath):
    analog_xml = ET.parse(xml_fpath).getroot()
    return float(analog_xml.findall('Experiment')[0].find('Rate').text)


# concatenate the analog input csv files if there are multiple cycles
def bruker_concatenate_analog(fname, fpath):
    # grab all csv voltage recording csv files that aren't the concatenated full
    glob_analog_csv = [f for f in glob.glob(os.path.join(fpath, "*_VoltageRecording_*.csv")) if 'full' not in f]
    glob_analog_xml = glob.glob(os.path.join(fpath, "*_VoltageRecording_*.xml"))

    # xml's contain metadata about the analog csv; make sure sampling rate is consistent across cycles
    analog_xml_fs = set(map(bruker_analog_xml_get_fs,
                            glob_analog_xml))  # map grabs sampling rate across all cycle xmls; set finds all unique list entries  
    if len(analog_xml_fs) > 1:
        warnings.warn('Sampling rate is not consistent across cycles!')
    else:
        analog_fs = list(analog_xml_fs)[0]

    # cycle through analog csvs and append to a dataframe
    analog_concat = pd.DataFrame()
    for cycle_idx, cycle_path_csv in enumerate(glob_analog_csv):

        cycle_df = pd.read_csv(cycle_path_csv)
        num_samples = len(cycle_df['Time(ms)'])
        cycle_df['Time(s)'] = cycle_df['Time(ms)'] / analog_fs

        cycle_df['cycle_num'] = float(re.findall('Cycle(\d+)', cycle_path_csv)[0])  # get cycle # from filename
        if cycle_idx == 0:  # initialize pd dataframe with first cycle's data
            cycle_df['cumulative_time'] = cycle_df['Time(ms)'].values
            analog_concat = cycle_df
        else:
            last_cumulative_time = analog_concat['cumulative_time'].iloc[-1]
            cycle_df['cumulative_time'] = cycle_df[
                                              'Time(ms)'].values + last_cumulative_time + 1  # add 1 so that new cycle's first sample isn't the same as the last cycle's last sample
            analog_concat = analog_concat.append(cycle_df, ignore_index=True)

    analog_concat.columns = analog_concat.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(',
                                                                                                            '_').str.replace(
        ')', '')

    # loop through all analog columns and get the diff and threshold for event onsets
    analog_column_names = [column for column in analog_concat.columns if 'input' in column]
    num_analogs = len(analog_column_names)
    for analog_column_name in analog_column_names:
        analog_concat[analog_column_name + '_diff'] = np.append(np.diff(analog_concat[analog_column_name]) > 4,
                                                                False)  # add a false to match existing df length

    # save concatenated analog csv        
    save_full_csv_path = os.path.join(fpath, fname + '_VoltageRecording_full.csv')
    analog_concat.to_csv(save_full_csv_path, index=False)

    return analog_concat


# function for finding the index of the closest entry in an array to a provided value
def find_nearest_idx(array, value):
    if isinstance(array, pd.Series):
        idx = (np.abs(array - value)).idxmin()
        return idx, array.index.get_loc(idx), array[idx]  # series index, 0-relative index, entry value
    else:
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx, array[idx]


### Take in analog dataframe (contains analog tseries and thresholded boolean) and make dict of 2p frame times for each condition's event
def match_analog_event_to_2p(imaging_info_df, analog_dataframe, flag_multicondition_analog=False):
    analog_event_dict = {}  # will contain analog channel names as keys and 2p imaging frame numbers for each event/ttl onset
    all_diff_columns = [diff_column for diff_column in analog_df.columns if
                        'diff' in diff_column]  # grab all diff'd analog column names

    for ai_diff in all_diff_columns:

        if flag_multicondition_analog:  # if the trials in analog ports need to be split up later, make a subdict to accommodate conditions keys
            analog_event_dict[ai_diff] = {}
            analog_event_dict[ai_diff]['all'] = []
        else:
            analog_event_dict[ai_diff] = []

        # grab analog samples where TTL onset occurred
        # analog_df diff columns are booleans for each frame that indicate if TTL threshold crossed (ie. event occurred)
        analog_events = analog_df.loc[analog_df[ai_diff] == True, ['time_s', 'cycle_num']]

        # for each detected analog event, find nearest 2p frame index and add to analog event dict
        analog_event_samples = []
        for idx, analog_event in analog_events.iterrows():

            analog_event_samples.append(idx)
            this_cycle_imaging_info = imaging_info_df[imaging_info_df['cycle_num'] == analog_event['cycle_num']]

            whole_session_idx, cycle_relative_idx, value = find_nearest_idx(this_cycle_imaging_info['rel_time'],
                                                                            analog_event['time_s'])

            if flag_multicondition_analog:
                analog_event_dict[ai_diff]['all'].append(whole_session_idx)
            else:
                analog_event_dict[ai_diff].append(whole_session_idx)

    return analog_event_dict, analog_event_samples


# if all behav events of interest (different conditions) are recorded on a single AI channel
# and need to reference the behavioral events csv to split conditions up
def split_analog_channel(ai_to_split, fdir, behav_fname, behav_event_key_path, analog_event_dict):
    unicode_to_str = lambda x: str(x)  # just a simple function to convert unicode to string; 

    this_ai_to_split = \
    [analog_diff_name for analog_diff_name in analog_event_dict.keys() if str(ai_to_split) in analog_diff_name][0]

    # load id's and samples (camera samples?) of behavioral events (output by behavioral program)
    behav_df = pd.read_csv(os.path.join(fdir, behav_fname), names=['id', 'sample'])
    behav_event_keys = pd.read_excel(behav_event_key_path)

    # using the behav event id, grab the event name from the keys dataframe; names are in unicode, so have to convert to string
    behav_name_of_interest = map(unicode_to_str,
                                 behav_event_keys[behav_event_keys['event_id'].isin(behav_id_of_interest)][
                                     'event_desc'].values)

    # go into ordered behavioral event df, grab the trials with condition IDs of 'behav_id_of_interest' in order
    trial_ids = list(
        behav_df[behav_df['id'].isin(behav_id_of_interest)]['id'].values)  # grab 101, 102, 103 trials in order

    # loop through behav conditions, and separate event times for the conglomerate event times in analog_event_dict
    for behav_event_id, behav_event_name in zip(behav_id_of_interest, behav_name_of_interest):
        this_event_idxs = [idx for idx, val in enumerate(trial_ids) if val == behav_event_id]
        analog_event_dict[this_ai_to_split][behav_event_name] = [analog_event_dict[this_ai_to_split]['all'][idx] for idx
                                                                 in this_event_idxs]
        # analog_event_dict ultimately contains 2p frame indices for each event categorized by condition

    # save preprocessed behavioral event data
    with open(behav_analog_save_path, 'wb') as handle:
        pickle.dump(analog_event_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


# take in data from an analog input and plot detected ttls
def plot_analog_validation(AI_onsets, analog_tseries, analog_fs, save_dir=None):
    # following is just for visualizing ttls; here make tiles for indexing and extracting ttl data in trial manner
    num_AI = len(AI_onsets)
    rel_ind_vec = np.arange(-0.5 * analog_fs, 3 * analog_fs, 1)
    rel_ind_tile = np.tile(rel_ind_vec, (num_AI, 1))
    AI_onset_tile = np.tile(AI_onsets, (len(rel_ind_vec), 1)).T

    # extract analog values across flattened trial indices, get values of series, then reshape to 2d array
    AI_value_tile = analog_tseries[np.ndarray.flatten(AI_onset_tile + rel_ind_tile)].values.reshape(AI_onset_tile.shape)
    if AI_value_tile.shape[0] == num_AI:
        AI_value_tile = AI_value_tile.T

    fig, ax = plt.subplots(1, 3, figsize=(17, 5))

    ax[0].set_title('Full TTL series')
    ax[0].plot(analog_tseries)

    ax[1].set_title('{} ttls detected'.format(num_AI))
    ax[1].plot(AI_value_tile);
    ax[1].set_xlabel('Time (ms)')
    ax[1].set_ylabel('Volts');

    svec = np.arange(0, 15 * analog_fs)
    tvec_plot = svec / analog_fs
    ax[2].set_title('Specific window (first 15s)')
    ax[2].plot(tvec_plot, analog_tseries[svec])
    ax[2].set_xlabel('Seconds')

    if save_path:
        utils.check_exist_dir(save_dir)
        fig.savefig(os.path.join(save_dir, 'ttl_validation.png'));


# #### 1) Parse main time-series xml, 2) extract frame timing and cycle info into a pandas dataframe 

# In[8]:


xml_parse = ET.parse(bruker_tseries_xml_path).getroot()
frame_info_df = pd.DataFrame()
type_tags = xml_parse.findall('Sequence/Frame')

# lambda function to take in a list of xml frame meta data and pull out timing and cycle info 
grab_2p_xml_frame_time = lambda type_tag: [float(type_tag.attrib['relativeTime']),
                                           float(type_tag.attrib['absoluteTime']),
                                           int(re.findall('Cycle(\d+)', type_tag.findall('File')[0].attrib['filename'])[
                                                   0])
                                           # first grab this frame's file name, then use regex to grab cycle number in the fname
                                           ]

# make a dataframe of relative time, absolute time, cycle number for each frame
imaging_info_df = pd.DataFrame(map(grab_2p_xml_frame_time, type_tags), columns=['rel_time', 'abs_time', 'cycle_num'])
# get more timing meta data
fs_2p = bruker_xml_get_2p_fs(bruker_tseries_xml_path)
tvec_2p = imaging_info_df['rel_time']
num_frames_2p = len(tvec_2p)

# ### Load and process analog voltage recordings

# If you have analog signals, that indicate behavioral event onset, sent from your behavioral DAQ to the bruker GPIO box, the following code:
# 
# 1) parses the analog voltage recording xmls 
# 2) extracts the signals from the csvs
# 3) extracts the TTL onset times
# 4) and finally lines up which frame the TTL occurred on.

# In[79]:


# run following if bruker analog signals are of interest
if bruker_analog:

    ### get analog data sampling rate from xml
    analog_fs = bruker_analog_xml_get_fs(glob_analog_xml[0])

    ### either load concatenated voltage recording (across cycles), perform the concatenation, or load a single CSV (for single cycle)
    volt_rec_full_path = os.path.join(fdir, fname + '_VoltageRecording_full.csv')
    if os.path.exists(volt_rec_full_path):  # if a trial-stitched voltage recording was previously saved
        analog_df = pd.read_csv(volt_rec_full_path)
    else:
        analog_df = bruker_concatenate_analog(fname, fdir)

        ### match analog ttl event onsets to the corresponding 2p frame (for each event in each analog port)
    analog_event_dict, analog_event_samples = match_analog_event_to_2p(imaging_info_df, analog_df)

    ### if there are multiple conditions signaled on a single analog port, split them up
    if flag_multicondition_analog:
        split_analog_channel(ai_to_split, fdir, behav_fname, behav_event_key_path, analog_event_dict)

    if validation_plots:
        valid_save_dir = os.path.join(fdir, fname + '_output_images')
        utils_bruker.check_exist_dir(valid_save_dir)
        plot_analog_validation(analog_event_samples, analog_df[valid_plot_channel],
                               analog_fs, valid_save_dir);

# ## load behav data and ID event onset 2p frames

# Alternatively, if you have a separate event recorder that is synchronized to the 2p microscope (via frame onset TTL from the GPIO output), you can use the following code.

# In[ ]:


behav_fs = 1000.0  # sampling rate of behavioral csv

# In[10]:


# load id's and samples (camera samples?) of behavioral events (output by behavioral program)
behav_df = pd.read_csv(os.path.join(fdir, behav_fname), names=['id', 'sample'])
behav_event_keys = pd.read_excel(behav_event_key_path)

# In[11]:


# behav camera pulses are synced to 2p frames. To synchronize event times with the 2p frames, need to normalize to the first 
# camera frame.
try:
    camera_pulse_event_id = behav_event_keys['event_id'][behav_event_keys['event_desc'] == 'camera pulse'].values[0]
    first_cam_pulse_sample = behav_df[behav_df['id'] == camera_pulse_event_id].iloc[0]['sample']
except:
    print('No camera pulse events!')

# In[12]:


frame_events_dict = {}

# loop through each type of event
for idx, row in behav_event_keys.iterrows():
    # grab event id
    this_id_name = str(row['event_desc'])
    # grab rows of behav dataframe with this event's id
    this_id_rows = behav_df['id'].isin([row['event_id']])

    # convert to seconds
    event_times_seconds = (behav_df[this_id_rows]['sample'].values - first_cam_pulse_sample) / behav_fs
    # first_cam_pulse_sample subtracted to zero times relative to first camera frame

    # using zero'd event times in seconds, find closest 2p frame sample index
    frame_events_dict[this_id_name] = map(partial(find_nearest_idx, tvec_2p), event_times_seconds)

# save preprocessed behavioral event data
with open(behav_save_path, 'wb') as handle:
    pickle.dump(frame_events_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

# ## Run this is if you performed opto stim

# In[ ]:
