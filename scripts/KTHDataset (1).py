#!/usr/bin/env python
# coding: utf-8

# In[1]:


from torch.utils.data import Dataset, Subset, DataLoader
import imageio, torch, os, shutil
import numpy as np
from PIL import Image

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

split = {'Train': ['11', '12', '13', '14', '15', '16', '17', '18','19', '20', '21', '23', '24', '25', '01', '04'],
                             'Test' : ['22', '02', '03', '05', '06', '07', '08', '09', '10']}

fulldataset = KTH_Dataset(directory='C://Users//aysha//Cuda Lab//KTH')
subset_datasets = {}
fulldataset.prepare_data()

for split, ids in split.items():
    idx = fulldataset.get_indices_for_persons(ids)
    subset_datasets[split] = Subset(fulldataset, idx)
    
    
train_dataloader=DataLoader(subset_datasets['Train'] , batch_size=16, shuffle=True)
    
test_dataloader=DataLoader(subset_datasets['Test'] , batch_size=16, shuffle=True)


# In[2]:


print('==>>> total trainning batch number: {}'.format(len(train_dataloader)))
print('==>>> total testing batch number: {}'.format(len(test_dataloader)))

for seq, seq_target in train_dataloader:
    print('--- Sample')
    print('Input:  ', seq.shape)
    print('Target: ', seq_target.shape)
    break


# In[3]:


test_input, test_target = next(iter(test_dataloader))
print(test_input.shape)


# In[4]:


from torchvision.utils import make_grid
import matplotlib.pyplot as plt

def show(grid, name):
    fix, axs = plt.subplots()
    fix.set_size_inches(25,8)

    axs.imshow(grid.cpu().numpy().transpose(1,2,0))
    axs.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
#     fix.savefig(f"{name}.pdf", format="pdf", bbox_inches="tight")
#     fix.show()

def visualize_results(test_loader, device):
    test_input, test_target = next(iter(test_loader))
    
    test_input = test_input.to(device)
    test_target = test_target.to(device)
    print(test_target[0])
    full_gt_seq = test_input
    print(full_gt_seq.shape)

    
    grid_gt = make_grid(full_gt_seq[0])
    show(grid_gt, "gt")


# In[5]:


visualize_results(test_dataloader, 'cuda')


# In[ ]:




