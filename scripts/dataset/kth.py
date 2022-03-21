from torch.utils.data import Dataset, Subset
import imageio, torch, os
import numpy as np
from PIL import Image

class KTH(Dataset):
    
    categories = ["boxing", "handclapping", "handwaving","jogging", "running", "walking"]

    def __init__(self, data_root,  transform=None, target_transform=None, frames_per_item = 20, download=True):
        self.__data_root = data_root
        self.__transform = transform
        # self.__target_transform = target_transform
        self.__frames_per_item = frames_per_item
        self.download = download
        self.split = {'Train': ['11', '12', '13', '14', '15', '16', '17', '18','19', '20', '21', '23', '24', '25', '01', '04'],
                                     'Test' : ['22', '02', '03', '05', '06', '07', '08', '09', '10']}

    def prepare_data(self):       
        self.__data = self.__parse_sequence_file(self.__data_root, self.__frames_per_item)
        if self.download:
            self.__process_video_frames(self.__data, self.__data_root + '/data')
    
    def __len__(self):
        return len(self.__data)
    
    def __getitem__(self, idx):
        item_data = torch.load(self.__data_root + '/data/' + str(idx) + '.pt')
        return (item_data[:10,:,:,:], item_data[10:,:,:,:])

   
    def __parse_sequence_file(self, base_dir, num_frames):
        # parse file for example : "person01_boxing_d1		frames	1-95, 96-185, 186-245, 246-360"
        print("*" * 50)
        print("Parsing Seq File")
        print("*" * 50)
        data = []
        idx=0
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
                            if(idx<=9962):
                                data.append({'video_filepath': filepath, 'start':i, 'end':end, 'label':label, 
                                         'label_desc' : label_desc, 'person':person_num})
                            idx=idx+1
        print(data[0])
        return data
    
    def __process_video_frames(self, data, base_dir):
        print("*" * 50)
        print("Processing Vid Frames")
        print("*" * 50)
        
        if not os.path.isdir(base_dir) : os.mkdir(base_dir)
        count=0
        for i, data_item in enumerate(data):
            vid = imageio.get_reader(data_item['video_filepath'])
            frames = []
            for frame_num in range(data_item['start'], data_item['end']):
                try:
                    frame=vid.get_data(frame_num)

                    frame = Image.fromarray(np.array(frame))
                    frame = frame.convert("L")
                    # frame = self.__transform(frame) 
                    frame = np.array(frame.getdata(), dtype=np.uint8)
                    frames.append([frame])
                except Exception as e:
                    print(e)
                    
                    continue
            frames = torch.from_numpy(np.array(frames)) / 255
            
            if((not(i>=65 and i<=68))):
                torch.save(frames, base_dir + '/' + str(count) + '.pt')


    def get_indices_for_persons(self, person_ids):
        return [ i for i, x in enumerate(self.__data) if x['person'] in person_ids]
    
    def train_test(self, train):
        fulldataset = KTH(data_root='data/kth',transform=self.__transform)
        subset_datasets = {}
        fulldataset.prepare_data()
        
        for split, ids in self.split.items():
            idx = fulldataset.get_indices_for_persons(ids)
            subset_datasets[split] = Subset(fulldataset, idx)
            
        if train:
            return subset_datasets['Train']
        else:
            return subset_datasets['Test']