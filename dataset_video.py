import torch.utils.data as data
from PIL import Image
import os
import os.path
import numpy as np
from numpy.random import randint
from opts import parse_opts
opt=parse_opts()

class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])

    @property
    def id_label(self):
        return int(self._data[3])


class TSNDataSet(data.Dataset):
    def __init__(self, root_path, list_file,
                 num_segments=3, new_length=1,
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 force_grayscale=False, random_shift=True, test_mode=False):

        self.root_path = root_path
        self.list_file = list_file
        self.num_segments = num_segments
        self.new_length = new_length
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self._parse_list()

    def _load_image(self, directory, idx):
        if opt.dataset == 'oulu':
            return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(idx))).convert('RGB')]
        elif opt.dataset == 'ckplus':
            return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(idx))).convert('L').convert('RGB')]

    def _parse_list(self):
        # check the frame number is large >3:
        # usualy it is [video_id, num_frames, class_idx]
        tmp = [x.strip().split(' ') for x in open(os.path.join(self.root_path,self.list_file))]
        tmp = [item for item in tmp if int(item[1])>=3]
        self.video_list = [VideoRecord(item) for item in tmp]
        print('video number:%d'%(len(self.video_list)))

        
    def _sample_indices(self, record):
        tick = record.num_frames / (self.num_segments-1)
        mid_segments = self.num_segments - 2
        offsets = np.array([int((i+1) * tick + randint(-1,1)) for i in range(mid_segments)])
        offsets = np.append(0,offsets)
        offsets = np.append(offsets,record.num_frames-2)
        return offsets + 1

        
    def _get_val_indices(self, record):
        tick = record.num_frames / (self.num_segments-1)
        mid_segments = self.num_segments - 2
        offsets = np.array([int((i+1) * tick) for i in range(mid_segments)])
        offsets = np.append(0,offsets)
        offsets = np.append(offsets,record.num_frames-2)
        return offsets + 1
    

    def __getitem__(self, index):
        record = self.video_list[index]
        segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)

        return self.get(record, segment_indices)

    def get(self, record, indices):
        images = list()
        for seg_ind in indices:
            idx = int(seg_ind)
            seg_imgs = self._load_image(record.path, idx)
            images.extend(seg_imgs)

        process_data = self.transform(images)
        return process_data, record.label, record.id_label

    def __len__(self):
        return len(self.video_list)