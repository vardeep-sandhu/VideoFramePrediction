from torch.utils.data import Dataset, Subset
import numpy as np
import cv2, torch, os
class KTH(Dataset):
    
    categories = ["boxing", "handclapping", "handwaving","jogging", "running", "walking"]


    def __init__(self, directory, download, train: bool, frames_per_item = 20, transform=None):
        self.__transform = transform
        self.__directory = directory
        self.__frames_per_item = frames_per_item
        self.__download=download
        self.train = train
        # self.data = self.prepare_data()

    def prepare_data(self):
        subset_datasets = {}
        split = {'Train': ['11', '12', '13', '14', '15', '16', '17', '18','19', '20', '21', '23', '24', '25', '01', '04'],
                                     'Test' : ['22', '02', '03', '05', '06', '07', '08', '09', '10']}

        self.__data = self.__parse_sequence_file(self.__directory, self.__frames_per_item)

        if(self.__download):
            self.__process_video_frames(self.__data, self.__directory + '/data', self.__transform)
        
        for split, ids in split.items():
            idx = self.get_indices_for_persons(ids)
            subset_datasets[split] = Subset(self, idx)
        
        if self.train:
            return subset_datasets['Train']
            
        else:
            return subset_datasets['Test']


    def __len__(self):
        return len(self.__data)
    

    def __getitem__(self, idx):
        item_data = torch.load(self.__directory + '/data/' + str(idx) + '.pt')

        item_data=self.__transform(item_data)
        
        item_data=torch.swapaxes(item_data, 0, 1)
        return (item_data[:10,:,:,:], item_data[10:,:,:,:])


    def __parse_sequence_file(self, base_dir, num_frames):
        # parse file for example : "person01_boxing_d1		frames	1-95, 96-185, 186-245, 246-360"
        data = []
        with open(base_dir + '/sequences.txt', 'r') as sequence_file:
            for sequence in sequence_file:
                split_1 = sequence.split('frames')   
                if len(split_1) > 1:
                    label_desc = split_1[0].split('_')[1]   
                    label = self.categories.index(label_desc)
                    person_num = split_1[0][6:8]
                    filepath = base_dir + '/' + label_desc + '/' + split_1[0].strip() + '_uncomp.avi' 
    
                    for overall_start, overall_end in [tuple(split_2.split('-')) for split_2 in split_1[1].strip().split(',')]:
                        for i in range(int(overall_start) , int(overall_end) - num_frames, 5):
                            end = i + num_frames
                            data.append({'video_filepath': filepath, 'start':i, 'end':end, 'label':label, 
                                         'label_desc' : label_desc, 'person':person_num})
        return data
    
    def __process_video_frames(self, data, base_dir, transform):
        if not os.path.isdir(base_dir) : os.mkdir(base_dir)
        for i, data_item in enumerate(data):
            print(data_item['video_filepath'])
            vid = cv2.VideoCapture(data_item['video_filepath'])
            frames = []
            for frame_num in range(data_item['start'], data_item['end']):

                vid.set(1, frame_num)
                _, frame = vid.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frames.append([frame])

            vid.release()
            frames = torch.from_numpy(np.moveaxis( np.array(frames), 1,0)) / 255
            torch.save(frames, base_dir + '/' + str(i) + '.pt')


    def get_indices_for_persons(self, person_ids):
        return [ i for i, x in enumerate(self.__data) if x['person'] in person_ids]

