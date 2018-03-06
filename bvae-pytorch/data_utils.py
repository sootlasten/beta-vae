import numpy as np
from torch.utils.data import Dataset, DataLoader


DSPRITES_PATH = '/home/stensootla/projects/dsprites-dataset/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'


class SpritesDataset(Dataset):
    def __init__(self, dataset_path):
        dataset_zip = np.load(DSPRITES_PATH, encoding='latin1')
        self.imgs = dataset_zip['imgs']
            
    def __len__(self):
        return self.imgs.shape[0]
        
    def __getitem__(self, idx):
        return self.imgs[idx].astype(np.float32)
        

def get_dataloader(batch_size, shuffle=True):
    sprite_dataset = SpritesDataset(dataset_path=DSPRITES_PATH)
    return DataLoader(sprite_dataset, batch_size=batch_size, shuffle=shuffle)

