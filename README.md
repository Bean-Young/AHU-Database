# US-Dataset

This project is created by Yuezhe Yang for paper "**An Ultrasound Dataset in the Wild for Machine Learning of Disease Classification**" ([Paper Link]()). 

## ***Abstract***

Ultrasound is a primary diagnostic tool commonly used to evaluate internal body structures, including organs, blood vessels, the musculoskeletal system, and fetal development. Due to challenges such as operator dependence, noise, limited field of view, difficulty in imaging through bone and air, and variability across different systems make diagnosing abnormalities in ultrasound images particularly challenging for less experienced clinicians. The development of artificial intelligence technology could assist in the diagnosis of ultrasound images. However, many databases are created using a single device type and collection site, limiting the generalizability of machine learning classification models. Therefore, we have collected a large, publicly accessible ultrasound challenge database that is intended to significantly enhance the performance of traditional ultrasound image classification. This dataset is derived from publicly available data on the Internet (douyin.com) and comprises a total of 1,833 distinct ultrasound data. It includes 13 different ultrasound image anomalies, and all data have been anonymized. Our data-sharing program aims to support benchmark testing of ultrasound image disease diagnosis and classification accuracy in multicenter environments.

## ***Prepare Data***

Our data is publicly available on [Figshare](). It includes Raw Data, Filtered Data, and Processed Data. If you choose the Raw Data, you will need to perform annotation and filtering yourself. We encourage you to explore and extract more valuable information from it.

## ***Set Up*** 

### Pytorch 2.0 (CUDA 11.8)
Our experimental platform is configured with RTX 3090 GPU (CUDA 11.8), and the code runs in a PyTorch 2.0 environment.
For details on the environment, please refer to the [`requirements.txt`](requirements.txt) file.

**Run the installation command:**
```
pip install -r requirements.txt
```

## ***Data Preprocessing***

> The suitable data for this step is the **Filtered Data**. If you choose the Processed Data, you can skip this step.

Data preprocessing is divided into two stages: 
1) We convert the data into a uniform format 
2) We generate a list of noisy data.

Firstly, you need to run the [`To_NPY.py`](To_NPY.py) file to convert the video and image collections into numpy array format with a `.npy` extension. In this process, the code also handles frame extraction from videos and image cropping. 

```
python To_NPY.py
```

Secondly, you need to run the [`DBSCAN.py`](DBSCAN.py) file to identify the noise files in each folder to ensure optimal deep learning classification performance. In the end, you will obtain a text file that records the paths of the noisy files.

```
python DBSCAN.py
```

## ***Technical Validation***

We use ***RVit*** to validate the dataset's validity. We used Top-3 Accuracy (*Top3-ACC*) as the primary evaluation metric.

You need to run the [**main.py**](RViT-main/main.py) file in RViT-main for the ultrasound symptom classification task on our dataset.

For specific steps and model details, please refer to the provided reference.

## ***Reference***
* [RViT](https://github.com/Jiewen-Yang/RViT/)
* [Visualizer](https://github.com/luo3300612/Visualizer)

## ***Citations***
