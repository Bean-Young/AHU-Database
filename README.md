# AHU-Dataset
A Heterogeneous Ultrasound Dataset Collected from Publicly Available in China for Machine Learning Application

## Ver 2.0（PyTorch）

#### Project created by Yuezhe Yang

This project is created by Yuezhe Yang for paper "A Heterogeneous Ultrasound Dataset Collected from Publicly Available in China for Machine Learning Application
" ([Paper Link]()). 


## Install 

You need to build the relevant environment first, please refer to : [**requirements.txt**](requirements.txt)

It is recommended to use Anaconda to establish an independent virtual environment, and python > = 3.8.0; (3.8.19 is used for this experimental platform).


## Data Preparation

Our data is publicly available on [Figshare](), and the data suitable for this project should be filtered data. 

Firstly, you need to download the filtered data, and then run the [**To_NPY.py**](To_NPY.py) file to convert the video and image collections into numpy array format with a `.npy` extension. 

Secondly, you need to run the [**DBSCAN.py**](DBSCAN.py) file to identify the noise files in each folder to ensure optimal deep learning classification performance. 

We have also provided the processed data files, resulting from these two steps, on Figshare under the title [Processed Data]().
## Technical Validation
We use ***RVit*** to validate the dataset's validity. 

You need to run the [**Main.py**](RVIT-main/main.py) file in RVIT-main. 

For specific steps and model details, please refer to the provided reference.
## Reference
* [RViT](https://github.com/Jiewen-Yang/RViT/)
* [Visualizer](https://github.com/luo3300612/Visualizer)

## Citations
