import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import random

class load_dataset(Dataset):
    def __init__(self, args: dict, split: str = 'train', debug: bool = False, device: torch.device = torch.device("cpu")):
        self.data_dirs = args['video_dirs']
        self.sample_length = args['sample_length']
        self.w, self.h = args['w'], args['h']
        self.size = (self.w, self.h)
        self.split = split
        self.device = device  # Store the device

        # Label to integer mapping
        self.label_mapping = {
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

        # Load noise file paths
        self.noise_files = self.load_noise_files('/media/Storage2/yyz/UltraSound/updated_noise_files.txt')

        # Get and split data
        self.train_paths, self.val_paths, self.train_labels, self.val_labels = self.split_data()

        # Set dataset paths and labels based on split
        if split == 'train':
            self.data_paths = self.train_paths
            self.labels = self.train_labels
        elif split == 'validation':
            self.data_paths = self.val_paths
            self.labels = self.val_labels
        else:
            raise ValueError("split parameter must be 'train' or 'validation'")

        # Determine the number of channels in the input images and set normalization parameters accordingly
        if self.data_paths:
            sample_data = np.load(self.data_paths[0][0])
            if len(sample_data.shape) == 2:  # Single-channel image
                self.channels = 1
            else:
                self.channels = sample_data.shape[-1]  # Multi-channel image
        else:
            self.channels = 3  # Default value

        self.transform = self.get_transform()

    def get_transform(self):
        """Set up image transformations for different splits"""
        transform_list = []

        # Resize all images
        transform_list.append(transforms.Resize(self.size, antialias=True))

        # Normalize the images
        transform_list.append(transforms.Normalize(mean=[0.5] * self.channels, std=[0.5] * self.channels))

        return transforms.Compose(transform_list)

    def load_noise_files(self, noise_file_path):
        """Load noise file paths and standardize path separators"""
        with open(noise_file_path, 'r') as file:
            noise_files = file.read().splitlines()
        # Normalize path separators
        noise_files = [os.path.normpath(noise_file) for noise_file in noise_files]
        return set(noise_files)

    def split_data(self):
        data_paths = []
        labels = []

        for root, dirs, files in os.walk(self.data_dirs):
            # Filter out noise files
            valid_files = [os.path.join(root, file) for file in sorted(files) if file.endswith('.npy') and os.path.normpath(os.path.join(root, file)) not in self.noise_files]
            
            if len(valid_files) == 0:
                continue  # Skip the folder if it contains no valid frames

            # Check if valid_files has enough frames for sample_length
            if len(valid_files) < self.sample_length:
                # If not enough frames, repeat the last frame until sample_length is met
                selected_files = valid_files + [valid_files[-1]] * (self.sample_length - len(valid_files))
            else:
                # Randomly select a start index
                start_idx = random.randint(0, len(valid_files) - self.sample_length)
                selected_files = valid_files[start_idx:start_idx + self.sample_length]

            # Extract label
            label = os.path.basename(os.path.dirname(selected_files[0])).split('_')[-1].split(',')

            data_paths.append(selected_files)
            labels.append(label)
        """
        # Split the dataset into 10% of the original data
        total_data_size = len(data_paths)
        limited_size = int(0.1 * total_data_size)  # Calculate 10% of the total size
        
        # Select the first 10% of the data
        data_paths = data_paths[:limited_size]
        labels = labels[:limited_size]
        """
        # Split the dataset into 80% training and 20% validation
        train_paths, val_paths, train_labels, val_labels = train_test_split(data_paths, labels, test_size=0.2, random_state=42)

        return train_paths, val_paths, train_labels, val_labels
    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        # Get the corresponding frame sequence
        data_sequence = self.data_paths[index]
        label = self.labels[index]

        # Load and transform each frame
        frames = []
        for path in data_sequence:
            frame = torch.from_numpy(np.load(path).astype(np.float32))  # Load data
            frame = frame.permute(2, 0, 1)  # Move channel dimension to the front (512, 512, 3) -> (3, 512, 512)
            if self.transform:
                frame = self.transform(frame)  # Resize operation (3, 512, 512) -> (3, w, h)
            frames.append(frame.to(self.device))  # Move frame to GPU

        # Stack all frames into a tensor
        data = torch.stack(frames, dim=0)

        # If multiple labels, select the first one
        if isinstance(label, list):
            label = label[0]  # Select the first label
        else:
            label = label  # Single label case

        # Map label to integer and move to GPU
        label = self.label_mapping.get(label, -1)
        label = torch.tensor(label, device=self.device)

        return data, label
