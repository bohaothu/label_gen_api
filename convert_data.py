import os
from helper.dataset import load_dataset
from helper.dataset_v2 import get_package_dir, available_datasets
import pickle

dataset_storage_path = os.path.join(get_package_dir(), "..","dataset")
available_dataset = available_datasets()

if __name__ == "__main__":
    print("get dataset list:", ", ".join(available_dataset), ". start converting")
    for name in available_dataset:
        print("converting", name, "train")
        X_train, y_train, feature_names, label_names = load_dataset(name, 'train')
        with open(os.path.join(dataset_storage_path,name,'train.pickle'), 'wb') as f:
            pickle.dump([X_train, y_train, feature_names, label_names], f)
        print("converting", name, "test")
        X_test, y_test, _, _ = load_dataset(name, 'test')
        with open(os.path.join(dataset_storage_path,name,'test.pickle'), 'wb') as f:
            pickle.dump([X_test, y_test, feature_names, label_names], f)
    print("all done")