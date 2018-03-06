import numpy as np


DSPRITES_PATH = '/home/stensootla/projects/dsprites-dataset/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'


class SpritesDataset():
    def __init__(self, batch_size, dataset_path=DSPRITES_PATH):
        dataset_zip = np.load(dataset_path, encoding='latin1')
        self.imgs = dataset_zip['imgs']
        self.batch_size = batch_size
        
    def __len__(self):
        return len(self.imgs)
    
    @property
    def nb_batches(self):
        return len(self) // self.batch_size + 1
    
    def gen(self, shuffle=True):
        if shuffle: np.random.shuffle(self.imgs)
        for i in range(0, len(self), self.batch_size):
            yield self.imgs[i:i+self.batch_size]

