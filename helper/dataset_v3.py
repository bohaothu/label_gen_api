import os
import pandas as pd

dataset_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","dataset","v3")
available = pd.read_pickle(os.path.join(dataset_directory,"available.pkl"))

def available_datasets():
    return available.to_json(orient="records")

def load_dataset(dataset_name, variant, file_name):
    file_full_path = os.path.join(dataset_directory, dataset_name, str(variant), str(file_name)+'.pkl')
    return pd.read_pickle(file_full_path)

def load_dataset_overview(dataset_name, file_name):
    file_full_path = os.path.join(dataset_directory, dataset_name, str(file_name)+'.pkl')
    return pd.read_pickle(file_full_path)