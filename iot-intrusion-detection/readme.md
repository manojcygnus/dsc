# IoT Intrusion Detection System

## Overview
This project implements a Multi-Layer Perceptron (MLP) neural network for IoT intrusion detection using the ACI-IoT-2023 dataset.

## Project Structure
- `data/`: Contains the dataset
- `src/`: Source code modules
- `notebooks/`: Jupyter notebooks for exploration
- `main.py`: Main execution script
- `requirements.txt`: Project dependencies

## Dataset
The dataset used for this project is the **ACI-IoT-2023.csv** file, which is hosted on Google Drive. This file is used for training and evaluating the intrusion detection system.

You can access the dataset using the following link:

[Download ACI-IoT-2023.csv from Google Drive](https://drive.google.com/file/d/1aKL_ixZXl7-xbE9ibP7dO42WCUbyWCkR/view?usp=drive_link)

To use this dataset in your local environment, simply reference the dataset URL in your code, as shown in the following example:

```python
import pandas as pd

def load_data():
    file_url = "https://drive.google.com/uc?export=download&id=1aKL_ixZXl7-xbE9ibP7dO42WCUbyWCkR"
    data = pd.read_csv(file_url)
    return data
