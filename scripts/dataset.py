import gzip
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image

import imageio, torch, os, shutil

import errno
import urllib

class MNIST_Moving(Dataset):

    urls = [
        'https://github.com/tychovdo/MovingMNIST/raw/master/mnist_test_seq.npy.gz'
    ]

    raw_folder = 'raw'
    processed_folder = 'processed'
    train_file = 'moving_train.pt'
    test_file = 'moving_test.pt'

    def __init__(self, root, train=True, split=1_000, transform=None, target_transform=None, download=True) -> None:
        super().__init__()
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.split = split
        self.train = train

        if download:
            self._download()    
        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.train:
            self.train_data = torch.load(
                os.path.join(self.root, self.processed_folder, self.train_file))
        else:
            self.test_data = torch.load(
                os.path.join(self.root, self.processed_folder, self.test_file))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (seq, target) where sampled sequences are splitted into a seq
                    and target part
        """

        # need to iterate over time
        def _transform_time(data):
            new_data = None
            for i in range(data.size(0)):
                img = Image.fromarray(data[i].numpy(), mode='L')
                new_data = self.transform(img) if new_data is None else torch.cat([new_data, self.transform(img)], dim=0)
            return new_data

        if self.train:
            seq, target = self.train_data[index, :10], self.train_data[index, 10:]
        else:
            seq, target = self.test_data[index, :10], self.test_data[index, 10:]

        if self.transform is not None:
            seq = _transform_time(seq)

        if self.target_transform is not None:
            target = _transform_time(target)

        seq = torch.unsqueeze(seq, 1)
        target = torch.unsqueeze(target, 1)
        
        # [10 X 1 X 64 X 64], [10 X 1 X 64 X 64] 
        return seq, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)


    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.train_file)) and \
            os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file))

    def _download(self):
        # First check if folders exist
        
        if self._check_exists():
            return 

        # Make directories
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise
        # Start Download
        for url in self.urls:
            print('Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.root, self.raw_folder, filename)
            
            with open(file_path, 'wb') as f:
                f.write(data.read())
            with open(file_path.replace('.gz', ''), 'wb') as out_f, \
                    gzip.GzipFile(file_path) as zip_f:
                out_f.write(zip_f.read())
            os.unlink(file_path)
        
        print('Processing...')
        # Split into test and train
        training_set = torch.from_numpy(
            np.load(os.path.join(self.root, self.raw_folder, 'mnist_test_seq.npy')).swapaxes(0, 1)[:-self.split]
        )
        test_set = torch.from_numpy(
            np.load(os.path.join(self.root, self.raw_folder, 'mnist_test_seq.npy')).swapaxes(0, 1)[-self.split:]
        )

        with open(os.path.join(self.root, self.processed_folder, self.train_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')

        return 

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
