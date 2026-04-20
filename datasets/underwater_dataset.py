import os
import random
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as tfs
import torchvision.transforms.functional as TF

class UnderwaterDataset(Dataset):
    def __init__(self, root_dir, size=256):
        self.root_dir = root_dir
        self.size = size
        self.input_dir = os.path.join(root_dir, 'input')
        self.target_dir = os.path.join(root_dir, 'ground_truth')
        
        # Validation
        if not os.path.exists(self.input_dir):
            raise FileNotFoundError(f"Missing: {self.input_dir}")

        self.image_files = sorted([f for f in os.listdir(self.input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        self.resize = tfs.Resize((size, size))
        self.to_tensor = tfs.ToTensor()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        
        # Load
        input_img = Image.open(os.path.join(self.input_dir, img_name)).convert("RGB")
        target_img = Image.open(os.path.join(self.target_dir, img_name)).convert("RGB")
        
        # Resize
        input_img = self.resize(input_img)
        target_img = self.resize(target_img)
        
        # Augment (Flip Both)
        if random.random() > 0.5:
            input_img = TF.hflip(input_img)
            target_img = TF.hflip(target_img)
            
        # To Tensor (Output is 0 to 1, PERFECT for images)
        return self.to_tensor(input_img), self.to_tensor(target_img)