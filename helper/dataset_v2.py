import os
import pickle

package_directory = os.path.dirname(os.path.abspath(__file__))
dataset_directory = os.path.join(package_directory,"..","dataset")

def show_cwd():
    return os.getcwd()

def get_package_dir():
    return package_directory

def available_datasets():
    return ["emotions","birds","medical","genbase","delicious","bibtex"]

def load_dataset(dataset_name, variant):
    with open(os.path.join(dataset_directory, dataset_name,str(variant)+'.pickle'), 'rb') as f:
        X_train, y_train, feature_names, label_names = pickle.load(f)
    return X_train, y_train, feature_names, label_names