import numpy as np
import pandas as pd
import pymongo
import os
import pickle
import uuid
import ujson
from sklearn.manifold import TSNE
from skmultilearn.dataset import load_dataset

myclient = pymongo.MongoClient("mongodb://localhost:27017/")

def shortid(num):
    return [str(uuid.uuid4())[:8] for i in range(num)]

def label_encode(string):
    return string.replace("\\'",".").replace(" ","_")

def dataset_to_mongodb(dataset_name, dataset_type):
    collection_name = f"{dataset_name}_{dataset_type}"

    # open file
    X_train, y_train, feature_names, label_names = load_dataset(dataset_name,dataset_type)

    # assign ids
    index=shortid(X_train.shape[0])

    # insert feature data
    X_train=pd.DataFrame(X_train.todense(),columns=[feature_names[x][0] for x in range(X_train.shape[1])],index=index)
    X_train["_id"]=index
    myclient[collection_name]["features"].insert_many(X_train.to_dict('records'))

    # insert label data
    y_train=pd.DataFrame(y_train.todense(),columns=[label_encode(label_names[x][0]) for x in range(y_train.shape[1])],index=index)
    y_train["_id"]=index
    myclient[collection_name]["labels"].insert_many(y_train.to_dict('records'))

    # insert feature tsne
    t_sne = TSNE()
    t_sne.fit(X_train.drop("_id",axis=1))
    t_sne_df = pd.DataFrame(t_sne.embedding_, columns=["v1","v2"], index=index)
    t_sne_df["_id"]=index
    myclient[collection_name]["tsne_features"].insert_many(t_sne_df.to_dict('records'))

    # insert label tsne
    t_sne = TSNE()
    t_sne.fit(y_train.drop("_id",axis=1))
    t_sne_df = pd.DataFrame(t_sne.embedding_, columns=["v1","v2"], index=index)
    t_sne_df["_id"]=index
    myclient[collection_name]["tsne_labels"].insert_many(t_sne_df.to_dict('records'))
    
    # insert feature combination tsne
    t_sne = TSNE()
    y_unique = np.unique(y_train.drop("_id",axis=1).to_numpy().astype(int),axis=0)
    t_sne.fit(y_unique)
    t_sne_df = pd.DataFrame(t_sne.embedding_, columns=["v1","v2"])
    member_list = [list(y_train.index[y_train.drop("_id",axis=1)
                                              .apply(lambda x: np.array_equal(np.array(x.values).astype(int),row),axis=1)]) for row in y_unique]
    t_sne_df["_member"] = member_list
    myclient[collection_name]["tsne_labels_combination"].insert_many(t_sne_df.to_dict('records'))

def label_to_mongodb(dataset_name):
    X_train, y_train, feature_names, label_names = load_dataset(dataset_name,"train")
        
    z=[]
    
    for item in label_names:
        z.append(label_encode(item[0]))
        
    print(z)
        
    myclient["config"]["avaliable_dataset"].insert_one({"dataset_name": dataset_name,"label_names": z})


if __name__ == "__main__":
    from skmultilearn.dataset import available_data_sets
    print(set([x[0] for x in available_data_sets().keys()]))
    
    wanted_dataset = ["tmc2007_500", "genbase", "Corel5k", "rcv1subset2"]

    for item in wanted_dataset:
        dataset_to_mongodb(item,"train")
        dataset_to_mongodb(item,"test")
        label_to_mongodb(item)
