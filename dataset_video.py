import torch.utils.data as data
from PIL import Image
import os
import os.path
import numpy as np
from numpy.random import randint

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
        return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(idx))).convert('RGB')]

    def _parse_list(self):
        # check the frame number is large >3:
        # usualy it is [video_id, num_frames, class_idx]
        tmp = [x.strip().split(' ') for x in open(self.list_file)]
        tmp = [item for item in tmp if int(item[1])>=3]
        self.video_list = [VideoRecord(item) for item in tmp]
        print('video number:%d'%(len(self.video_list)))

    def _sample_indices_ori(self, record):
        average_duration = (record.num_frames - self.new_length + 1) // self.num_segments    ##分段间隔
        if average_duration > 0:                       
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments)
        elif record.num_frames > self.num_segments:
            offsets = np.sort(randint(record.num_frames - self.new_length + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1
        
    def _sample_indices(self, record):
        tick = record.num_frames / (self.num_segments-1)
        mid_segments = self.num_segments - 2
        offsets = np.array([int((i+1) * tick + randint(-1,1)) for i in range(mid_segments)])
        offsets = np.append(0,offsets)
        offsets = np.append(offsets,record.num_frames-2)
        return offsets + 1
        
    def _get_val_indices_ori(self, record):
        if record.num_frames > self.num_segments + self.new_length - 1:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1
        
    def _get_val_indices(self, record):
        tick = record.num_frames / (self.num_segments-1)
        mid_segments = self.num_segments - 2
        offsets = np.array([int((i+1) * tick) for i in range(mid_segments)])
        offsets = np.append(0,offsets)
        offsets = np.append(offsets,record.num_frames-2)
        return offsets + 1
    
    def _get_test_indices(self, record):

        tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)

        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])

        return offsets + 1

    def __getitem__(self, index):
        record = self.video_list[index]
        # check this is a legit video folder
        while not os.path.exists(os.path.join(self.root_path, record.path, self.image_tmpl.format(1))):
            print(os.path.join(self.root_path, record.path, self.image_tmpl.format(2)))
            index = np.random.randint(len(self.video_list))
            record = self.video_list[index]

        if not self.test_mode:
            segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        else:
            segment_indices = self._get_test_indices(record)

        return self.get(record, segment_indices)

    def get(self, record, indices):

        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                seg_imgs = self._load_image(record.path, p)
                images.extend(seg_imgs)
                if p < record.num_frames:
                    p += 1

        process_data = self.transform(images)
        return process_data, record.label

    def __len__(self):
        return len(self.video_list)

########  函数调用逻辑为  当class TSNdataset被实例化时，python自带函数__getitem__被自动调用
########  __getitem__ -----> get ----->  _load_image     _load_image最终会完成通过路径拼接读取图像的任务
#######   该程序读入list文件共有三列  通过class VideoRecord将三列信息封装为record  分别为视频路径  帧数  类别  该类的实例化在parse list中完成

#######   函数_sample_indices为train时的帧选取方案   offsets为随机n帧的index，eg 12帧分3段  则offsets为[3 5 10]
#######   _get_val_indices与_get_train_indices的分支进入区别是  train_loader(random_shift = true)  val_loader(random_shift = false)
#######   测试时的读取方式为等距取三帧 每次取的都是固定的