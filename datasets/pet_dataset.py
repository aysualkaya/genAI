import os
from PIL import Image
from torch.utils.data import Dataset

class PETDataset(Dataset):
    def __init__(self, inp_dir, gt_dir, transform=None):
        self.inp_dir = inp_dir
        self.gt_dir = gt_dir
        self.inp_files = sorted(os.listdir(inp_dir))
        self.gt_files = sorted(os.listdir(gt_dir))
        self.transform = transform

    def __len__(self):
        return len(self.inp_files)

    def __getitem__(self, idx):
        inp_path = os.path.join(self.inp_dir, self.inp_files[idx])
        gt_path = os.path.join(self.gt_dir, self.gt_files[idx])
        
        inp_img = Image.open(inp_path).convert('L')
        gt_img = Image.open(gt_path).convert('L')
        
        if self.transform:
            inp_img = self.transform(inp_img)
            gt_img = self.transform(gt_img)

        return inp_img, gt_img
