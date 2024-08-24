import os, re, random


label_index = {
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

def get_csv_file(root_dir):
    label_data = {}
    for root, dirs, files in os.walk(root_dir):
        for dir_name in dirs:
            label_names = os.path.basename(dir_name).split(',')
            if label_names[0] in label_index:
                label = label_index[label_names[0]]  
                label_data[dir_name] = label
    return label_data, sorted(label_data, key=str2int)

def get_file_name(file_path):
    for root, dirs, _ in os.walk(file_path):
        for i, dir_name in enumerate(dirs):
            dirs[i] = os.path.join(root, dir_name)
        return sorted(dirs, key=str2int)


def get_frame(video_path):
    imagelist = []
    for root, _, filenames in os.walk(video_path):
        for filename in filenames:
            if filename.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                imagelist.append(os.path.join(root, filename))
        return sorted(imagelist, key=str2int)


def get_ref_index(length, sample_length):
    if random.uniform(0, 1) > 0.5:
        ref_index = random.sample(range(length), sample_length)
        ref_index.sort()
    else:
        pivot = random.randint(0, length-sample_length)
        ref_index = [pivot+i for i in range(sample_length)]
    return ref_index

def str2int(v_str):
    def tryint(s):
        try:
            return int(s)
        except ValueError:
            return s
    return [tryint(sub_str) for sub_str in re.split('([0-9]+)', v_str)]