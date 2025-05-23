# AHU-Database

This project is created by [***Yuezhe Yang***](https://bean-young.github.io) for paper "**An Annotated Heterogeneous Ultrasound Database**" ([Paper Link](https://www.nature.com/articles/s41597-025-04464-4)). 

### [Project Page](https://ahudataset.github.io)

## ***Abstract***

Ultrasound is a primary diagnostic tool commonly used to evaluate internal body structures, including organs, blood vessels, the musculoskeletal system, and fetal development. Due to challenges such as operator dependence, noise, limited field of view, difficulty in imaging through bone and air, and variability across different systems, diagnosing abnormalities in ultrasound images is particularly challenging for less experienced clinicians. The development of artificial intelligence (AI) technology could assist in the diagnosis of ultrasound images. However, many databases are created using a single device type and collection site, limiting the generalizability of machine learning models. Therefore, we have collected a large, publicly accessible ultrasound challenge database that is intended to significantly enhance the performance of AI-assisted ultrasound diagnosis. This database is derived from publicly available data on the Internet and comprises a total of 1,833 distinct ultrasound data. It includes 13 different ultrasound image anomalies, and all data have been anonymized. Our data-sharing program aims to support benchmark testing of ultrasound disease diagnosis in multi-center environments.

## ***Prepare Data***

Our data is publicly available on [Figshare](https://springernature.figshare.com/articles/dataset/An_annotated_heterogeneous_ultrasound_database/26889334). It includes **Raw Data**, **Filtered Data**, and **Processed Data**.

- **Raw Data:** If you choose the Raw Data, you will need to perform annotation and filtering yourself. We encourage you to explore and extract more valuable information from it.
- **Filtered Data:** If you choose the Filtered Data, the Data Preprocessing guidelines provided in this documentation will apply to your usage.
- **Processed Data:** If you choose the Processed Data, please note that the provided `noise_files.txt` contains our absolute file paths. To ensure the data can be used correctly, run the [`rename.py`](rename.py) script to update the paths.

We hope this facilitates your research and exploration of our dataset.

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
1) We convert the data into a NPY format 
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

You need to run the [`main.py`](RViT-main/main.py) file in RViT-main for the ultrasound symptom classification task on our dataset.

For specific steps and model details, please refer to the provided reference.

## ***References***
* [RViT](https://github.com/Jiewen-Yang/RViT/)
* [Visualizer](https://github.com/luo3300612/Visualizer)

## ***Citations***

``````bibtex
@article{yang2025annotated,
  title={An annotated heterogeneous ultrasound database},
  author={Yang, Yuezhe and Chen, Yonglin and Dong, Xingbo and Zhang, Junning and Long, Chihui and Jin, Zhe and Dai, Yong},
  journal={Scientific Data},
  volume={12},
  number={1},
  pages={148},
  year={2025},
  publisher={Nature Publishing Group UK London}
}
