import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

class visual_data(Dataset):
    def __init__(self, args: dict, split: str = 'train', debug: bool = False, device: torch.device = torch.device("cpu")):
        self.data_dirs = args['video_dirs']
        self.sample_length = args['sample_length']
        self.w, self.h = args['w'], args['h']
        self.size = (self.w, self.h)
        self.split = split
        self.device = device

        assert split in ['train', 'validation', 'test'], "The mode can only be 'train', 'validation', and 'test'"

        self.noise_files = self.load_noise_files('/media/Storage2/yyz/UltraSound/updated_noise_files.txt')
        self.train_paths, self.val_paths, self.test_paths, self.train_labels, self.val_labels, self.test_labels = self.split_data()

        if split == 'train':
            self.data_paths = self.train_paths
            self.labels = self.train_labels
        elif split == 'validation':
            self.data_paths = self.val_paths
            self.labels = self.val_labels
        elif split == 'test':
            self.data_paths = self.test_paths
            self.labels = self.test_labels

        self.transform = self.get_transform()

    def load_noise_files(self, noise_file_path):
        with open(noise_file_path, 'r') as file:
            noise_files = file.read().splitlines()
        noise_files = [os.path.normpath(noise_file) for noise_file in noise_files]
        return set(noise_files)

    def split_data(self):
        data_paths = []
        labels = []

        for root, dirs, files in os.walk(self.data_dirs):
            valid_files = [os.path.join(root, file) for file in sorted(files) if file.endswith('.npy') and os.path.normpath(os.path.join(root, file)) not in self.noise_files]

            if len(valid_files) == 0:
                continue

            if len(valid_files) < self.sample_length:
                selected_files = valid_files + [valid_files[-1]] * (self.sample_length - len(valid_files))
            else:
                start_idx = random.randint(0, len(valid_files) - self.sample_length)
                selected_files = valid_files[start_idx:start_idx + self.sample_length]

            label = os.path.basename(os.path.dirname(selected_files[0])).split('_')[-1]

            data_paths.append(selected_files)
            labels.append(label)

        train_paths, temp_paths, train_labels, temp_labels = train_test_split(data_paths, labels, test_size=0.3, random_state=42)
        val_paths, test_paths, val_labels, test_labels = train_test_split(temp_paths, temp_labels, test_size=0.3333, random_state=42)

        return train_paths, val_paths, test_paths, train_labels, val_labels, test_labels

    def get_transform(self):
        if self.split == 'train':
            transform_list = [
                transforms.Resize(self.size, antialias=True),
                transforms.RandomResizedCrop(self.size),
                transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
            ]
        else:
            transform_list = [
                transforms.Resize(self.size, antialias=True),
                transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
            ]
        return transforms.Compose(transform_list)

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        data_sequence = self.data_paths[index]
        label = self.labels[index]

        frames = []
        # Label to integer mapping
        label_mapping = {
            'Tumor': 0,
            'Cyst': 1,
            'Inflammation': 2,
            'Stone': 3,
            'Nodule': 4,
            'Injury': 5,
            'Calcification': 6,
            'Occupancy': 7,
            'Hernia': 8,
            'Vascular': 9,
            'Polyp': 10,
            'Ectopic': 11,
            'Anomalies': 12
        }

        for path in data_sequence:
            frame = torch.from_numpy(np.load(path).astype(np.float32)).permute(2, 0, 1)
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame.to(self.device))

        data = torch.stack(frames, dim=0)
        label = label[:label.index(',')] if ',' in label else label
        # Map label to integer and move to GPU 
        label = label_mapping.get(label, -1)
        label = torch.tensor(label, device=self.device)

        return data, label,frames