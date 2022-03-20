import numpy as np
from torch.utils.data import Dataset
import os
from PIL import Image

import imageio, torch, os

class KTH_Dataset(Dataset):
    
    categories = ["boxing", "handclapping", "handwaving","jogging", "running", "walking"]

    def __init__(self, directory, frames_per_item = 20, use_preloaded=True, transform=None):
        self.__transform = transform
        self.__directory = directory
        self.__use_preloaded = use_preloaded
        self.__frames_per_item = frames_per_item

    def prepare_data(self):       
        self.__data = self.__parse_sequence_file(self.__directory, self.__frames_per_item)
        if not self.__use_preloaded:
            self.__process_video_frames(self.__data, self.__directory + '/data', self.__transform)
    
    def __len__(self):
        return len(self.__data)
    
    def __getitem__(self, idx):
        item_data = torch.load(self.__directory + '/data/' + str(idx) + '.pt')
        return (item_data, self.__data[idx]['label'])
   
    def __parse_sequence_file(self, base_dir, num_frames):
        # parse file for example : "person01_boxing_d1		frames	1-95, 96-185, 186-245, 246-360"
        data = []
        with open(base_dir + '/00sequences.txt', 'r') as sequence_file:
            for sequence in sequence_file:
                split_1 = sequence.split('frames')   
                if len(split_1) > 1:
                    label_desc = split_1[0].split('_')[1]   
                    label = self.categories.index(label_desc)
                    person_num = split_1[0][6:8]
                    filepath = base_dir + '/' + label_desc + '/' + split_1[0].strip() + '_uncomp.avi' 
    
                    for overall_start, overall_end in [tuple(split_2.split('-')) for split_2 in split_1[1].strip().split(',')]:
                        for i in range(int(overall_start) , int(overall_end) - num_frames, num_frames):
                            end = i + num_frames
                            data.append({'video_filepath': filepath, 'start':i, 'end':end, 'label':label, 
                                         'label_desc' : label_desc, 'person':person_num})
        print(len(data))
        return data
    
    def __process_video_frames(self, data, base_dir, transform):
        if not os.path.isdir(base_dir) : os.mkdir(base_dir)
        for i, data_item in enumerate(data):
            vid = imageio.get_reader(data_item['video_filepath'], "ffmpeg")
            frames = []
            for frame_num in range(data_item['start'], data_item['end']):
                try:
                    frame=vid.get_data(frame_num)
                    frame = Image.fromarray(np.array(frame))
                    frame = frame.convert("L")
                    frame = np.array(frame.getdata(),
                                     dtype=np.uint8).reshape((120, 160))
                    print((frame))
                    frames.append([frame])
                except:
                    continue
            frames = torch.from_numpy(np.array(frames)) / 255
            torch.save(frames, base_dir + '/' + str(i) + '.pt')


    def get_indices_for_persons(self, person_ids):
        return [ i for i, x in enumerate(self.__data) if x['person'] in person_ids]
