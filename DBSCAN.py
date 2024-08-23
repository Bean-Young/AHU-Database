import numpy as np
import os
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.functional import avg_pool2d
from sklearn.cluster import DBSCAN
from sklearn.impute import SimpleImputer
from tqdm import tqdm


# Custom Dataset class
class NpyDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        image = np.load(file_path)
        downsampled_image = self.downsample(image)
        normalized_image = self.normalize(downsampled_image)
        return normalized_image.flatten(), file_path

    def downsample(self, image):
        # Convert numpy array to PyTorch tensor
        if image.ndim == 3:  # If it's a color image
            image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        else:  # If it's a single-channel image
            image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # Perform four rounds of average pooling, each with a 2x2 pool size
        for _ in range(4):
            image_tensor = avg_pool2d(image_tensor, kernel_size=2)
        # Remove extra dimensions
        return image_tensor.squeeze().numpy()

    def normalize(self, image):
        return (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)  # Add 1e-8 to avoid division by zero


# Feature extraction
def extract_features(data, device):
    # Fill NaN values
    imputer = SimpleImputer(strategy='mean')
    data = imputer.fit_transform(data)
    return torch.tensor(data, dtype=torch.float32).to(device)


# Clustering analysis - DBSCAN
def cluster_data_dbscan(features, eps=5, min_samples=5):
    features = features.cpu().numpy()  # DBSCAN only handles numpy arrays
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)  # Adjust eps and min_samples parameters
    labels = dbscan.fit_predict(features)
    return labels


def process_subdirectory(subdir, device, txt_file):
    file_paths = [os.path.join(subdir, file) for file in os.listdir(subdir) if file.endswith('.npy')]
    if not file_paths:
        return []

    # Create dataset and data loader
    dataset = NpyDataset(file_paths)
    dataloader = DataLoader(dataset, batch_size=32, num_workers=4, shuffle=False)

    # Load and process data
    all_data = []
    filenames = []

    for batch_data, batch_filenames in dataloader:
        all_data.append(batch_data)
        filenames.extend(batch_filenames)

    all_data = torch.cat(all_data, dim=0)

    # Feature extraction
    features = extract_features(all_data.cpu().numpy(), device)

    # Perform DBSCAN clustering
    labels = cluster_data_dbscan(features, eps=15, min_samples=(max(1, int(features.size(0) / 2))))

    # Get the most common label
    unique, counts = np.unique(labels, return_counts=True)
    most_common_label = unique[np.argmax(counts)]

    # Set all non-majority labels to -1 (noise labels)
    labels[labels != most_common_label] = -1

    # Output paths of files with label -1 (noise)
    noise_labels = []
    for filename, label in zip(filenames, labels):
        if label == -1:
            noise_labels.append(filename)

    # Write to TXT file
    if noise_labels:
        txt_file.write(f'Subdirectory: {subdir}\n')
        for noise_file in noise_labels:
            txt_file.write(f'{noise_file}\n')
        txt_file.flush()  # Flush the buffer to ensure data is written to the file

    return noise_labels


def main():
    main_directory = 'E:\\Privatizing Dataset Task\\Pretreat'
    output_txt = 'noise_files.txt'

    # Read processed subdirectories
    processed_subdirs = set()
    if os.path.exists(output_txt):
        with open(output_txt, 'r') as f:
            lines = f.readlines()
            processed_subdirs = {line.strip().split(': ')[1] for line in lines if line.startswith('Subdirectory: ')}

    # Set device to GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    subdirectories = [os.path.join(main_directory, d) for d in os.listdir(main_directory) if
                      os.path.isdir(os.path.join(main_directory, d))]

    with open(output_txt, 'a') as txt_file:  # Open the file in append mode
        # Use tqdm to display progress bar
        for subdir in tqdm(subdirectories, desc="Processing Subdirectories"):
            if subdir in processed_subdirs:
                continue  # Skip already processed subdirectories
            process_subdirectory(subdir, device, txt_file)

    print(f'Total number of subdirectories processed: {len(subdirectories)}')


if __name__ == '__main__':
    main()