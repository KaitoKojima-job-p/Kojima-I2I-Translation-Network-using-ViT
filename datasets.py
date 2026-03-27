import glob
import random
import os
import numpy as np
import h5py

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, mode="train"):
        self.transform = transforms.Compose(transforms_)

        self.files = sorted(glob.glob(os.path.join(root, mode) + "/*.*"))
        #if mode == "train":
        #    self.files.extend(sorted(glob.glob(os.path.join(root, "test") + "/*.*")))

    def __getitem__(self, index):

        img_path = self.files[index % len(self.files)]
        img = Image.open(img_path)
        w, h = img.size
        img_A = img.crop((0, 0, w / 2, h))
        img_B = img.crop((w / 2, 0, w, h))
        
        if np.random.random() < 0.5:
            img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], "RGB")
            img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], "RGB")

        img_A = self.transform(img_A)
        img_B = self.transform(img_B)

        return {"A": img_A, "B": img_B, "path": img_path}

    def __len__(self):
        return len(self.files)
    
class HDF5dataset(Dataset):
    def __init__(self, root, transforms_=None, mode="train"):
        self.transform = transforms.Compose(transforms_)
        self.files = sorted(glob.glob(os.path.join(root,mode) + "/*.hdf5"))
        
    def __getitem__(self, index):
        with h5py.File(self.files[index % len(self.files)]) as f:
            color = np.array(f["colors"], dtype=np.uint8)
            normal = np.array(f["normals"], dtype=np.uint8)   
            
        color = Image.fromarray(color)
        normal = Image.fromarray(normal)

        
        if np.random.random() < 0.5:
            color = Image.fromarray(np.array(color)[:,::-1, :], "RGB")
            normal = Image.fromarray(np.array(normal)[:,::-1,:], "RGB")
        
        color = self.transform(color)
        normal = self.transform(normal)
        
        return {"A": color, "B": normal}
    
    def __len__(self):
        return len(self.files)