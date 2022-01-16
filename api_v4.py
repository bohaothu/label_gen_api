from flask import Flask, request
from flask_cors import CORS
from flask_pymongo import PyMongo
import ujson
import numpy as np
import pandas as pd
from bson import json_util
from bson.objectid import ObjectId
from nltk.corpus import stopwords
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics as metrics
from sklearn.metrics import log_loss
from scipy.special import kl_div
import copy
from helper.helpers import hamming_score

app = Flask(__name__)
CORS(app)
mongo = PyMongo(app, uri="mongodb://localhost:27017")

br_kernel_dict = {"knn": KNeighborsClassifier(), "svc": SVC()}

def get_mutual_information(dfname, dftype, space, group_id):
    db = mongo.cx[f"{dfname}_{dftype}"]
    filter_result = db["graph_filters"].find_one({"_id": ObjectId(group_id) })

    #print(group_id)

    if filter_result["group_type"] == "selected":
        points = filter_result["points"]
    elif filter_result["group_type"] == "query":
        query_docs = db["labels"].find(filter_result["query"])
        points = [ query_doc["_id"] for query_doc in query_docs ]

    all_labels_doc =  db[space].find({})
    all_labels = [ doc for doc in all_labels_doc ]

    #print(all_labels)

    all_labels_index = [ x["_id"] for x in all_labels ]
    y_df = pd.DataFrame(all_labels,index=all_labels_index)

    group = y_df["_id"].apply(lambda x: 1 if x in set(points) else 0).to_numpy()

    mutual_information_dict = {}
    for tag in y_df.drop(columns="_id").columns:
        mutual_information_dict[tag] = normalized_mutual_info_score(y_df[tag].to_numpy(),group)

    result = sorted(mutual_information_dict.items(), key=lambda x:x[1], reverse=True)[:10]

    return result

@app.route("/hello/<string:dfname>/<string:dftype>")
def hello(dfname, dftype):
    db = mongo.cx[f"{dfname}_{dftype}"] # Database
    collection = db.list_collection_names()
    resp = ujson.dumps({"dataset": dfname, "dataset_type": dftype,
    "result": collection})
    return resp

@app.route("/hello/<string:dfname>/<string:dftype>/<string:colname>")
def hello_colname(dfname, dftype, colname):
    db = mongo.cx[f"{dfname}_{dftype}"]
    resp = db[colname].find_one() # Collection
    return resp

@app.route("/tsne/<string:tsnetype>/<string:dfname>/<string:dftype>", methods=["GET","POST"])
def tsne(dfname, dftype, tsnetype):
    drop_nolabel = True if int(request.args.get("nolabel",default="0")) else False
    filter_id = request.args.get("filter_id")

    db = mongo.cx[f"{dfname}_{dftype}"]
    col_group = db["graph_filters"]

    # if nolabel=1, drop data with 0 label
    nolabel_arr = []
    if drop_nolabel:
        label_col = mongo.cx[f"{dfname}_{dftype}"]["labels"]
        label_docs = label_col.find({})
        
        for label_doc in label_docs:         
            doc_dict = label_doc
            doc_id = label_doc["_id"]
            del doc_dict["_id"]
            if(sum(doc_dict.values()) == 0):
                nolabel_arr.append(doc_id)
                
    # if filter_id exist, add that filter into query            
    filter_result = col_group.find_one({"_id": ObjectId(filter_id) })
    
    # when filter_id is given in post body, filter_result will not be None
    if filter_result: # filter_result != None
        if tsnetype != "labels_combination":
            if filter_result["group_type"] == "selected" or filter_result["group_type"] == "selected_combination":
                final_query = {"_id": {"$in": filter_result["points"], "$nin": nolabel_arr}}
            elif filter_result["group_type"] == "query":
                query_docs = db["labels"].find(filter_result["query"])
                query_result = [ query_doc["_id"] for query_doc in query_docs ]
                final_query = {"_id": {"$in": query_result, "$nin": nolabel_arr}}
        elif tsnetype == "labels_combination":
            if filter_result["group_type"] == "selected" or filter_result["group_type"] == "selected_combination":
                final_query = {"_member": {"$in": filter_result["points"], "$nin": nolabel_arr}}
            elif filter_result["group_type"] == "query":
                query_docs = db["labels"].find(filter_result["query"])
                query_result = [ query_doc["_id"] for query_doc in query_docs ]
                final_query = {"_member": {"$in": query_result, "$nin": nolabel_arr}}

        docs=db[f"tsne_{tsnetype}"].find(final_query)

    # when filter_id is not given, search all group and exculde those points in result 
    else: # filter_result == None
        nogroup = set()
        docs_group = col_group.find({})
        for doc_group in docs_group: # if there is no group, this loop will not execute
            if doc_group["group_type"] == "selected" or doc_group["group_type"] == "selected_combination":
                filter_points = doc_group["points"]
            elif doc_group["group_type"] == "query":
                query_docs = db["labels"].find(doc_group["query"])
                filter_points = [query_doc["_id"] for query_doc in query_docs]
            nogroup = nogroup.union(set(filter_points))
        if tsnetype != "labels_combination":
            docs=db[f"tsne_{tsnetype}"].find({"_id": {"$nin": list(set(nolabel_arr).union(nogroup))}}).limit(500)
        elif tsnetype == "labels_combination":
            docs=db[f"tsne_{tsnetype}"].find({"_member": {"$nin": list(set(nolabel_arr).union(nogroup))}}).limit(500) 

    res = []
    for doc in docs:
        doc = ujson.loads(json_util.dumps(doc))
        res.append([doc["v1"],doc["v2"],doc["_id"]])

    resp = ujson.dumps({"dataset": dfname, "dataset_type": dftype,
    "tsne_type": tsnetype, "result": res, "filter_id": filter_id})

    return resp

@app.route("/heatmap/<string:dfname>/<string:dftype>")
def heatmap(dfname,dftype):
    filter_id = request.args.get("filter_id") if request.args.get("filter_id") else "null"
    db = mongo.cx[f"{dfname}_{dftype}"]

    if filter_id != "null":
        doc_group = db["graph_filters"].find_one({"_id": {"$eq": ObjectId(filter_id)}})

        if doc_group["group_type"] == "selected" or doc_group["group_type"] == "selected_combination":
            doc_query = {"_id": {"$in": doc_group["points"]}}
        elif doc_group["group_type"] == "query":
            query_docs = db["labels"].find(doc_group["query"])
            doc_query = {"_id" : {"$in": [query_doc["_id"] for query_doc in query_docs]} }
    elif filter_id == "null":
        doc_query = {}

    # read X from db
    X_docs = db["features"].find(doc_query)
    X_df = [doc for doc in X_docs]
    X_df = pd.DataFrame(X_df, index=[x["_id"] for x in X_df]).drop("_id",axis=1)
    # drop stop words
    sw = stopwords.words('english')
    tobedrop = set(X_df.columns) & set(sw)
    X_df = X_df.drop(columns=tobedrop)
    # keep the most 400 elements
    X_df = X_df[X_df.sum().sort_values(ascending=False)[:400].index]

    # read y from db
    y_docs = db["labels"].find(doc_query)
    y_df = [doc for doc in y_docs]
    y_df = pd.DataFrame(y_df, index=[x["_id"] for x in y_df]).drop("_id",axis=1)

    # calculate dot
    echart_data = []
    for feat in X_df.columns:
        for labl in y_df.columns:
            echart_data.append({"feature": feat, "label": labl, "dot": np.dot(X_df[feat].to_numpy(), y_df[labl].to_numpy())})
    echart_data = pd.DataFrame(echart_data).sort_values("dot").tail(1000)
    echart_x = list(set(echart_data["feature"]))
    echart_y = list(set(echart_data["label"]))
  
    return ujson.dumps({"dataset": dfname, "dataset_type": dftype, "echart_x": echart_x,
    "echart_y": echart_y, "echart_data": echart_data.values.tolist(), "filter_id": filter_id})

@app.route("/heatmap2/<string:dfname>/<string:dftype>")
def heatmap2(dfname, dftype):
    group_id = request.args.get("filter_id")

    db = mongo.cx[f"{dfname}_{dftype}"]
    
    # 先求对最这个 group mutual information 最大的 future 和 label
    feature_mi = get_mutual_information(dfname, dftype, "features", group_id)
    label_mi = get_mutual_information(dfname, dftype, "labels", group_id)

    doc_group = db["graph_filters"].find_one({"_id": {"$eq": ObjectId(group_id)}})

    if doc_group["group_type"] == "selected" or doc_group["group_type"] == "selected_combination":
        doc_query = {"_id": {"$in": doc_group["points"]}}
    elif doc_group["group_type"] == "query":
        query_docs = db["labels"].find(doc_group["query"])
        doc_query = {"_id" : {"$in": [query_doc["_id"] for query_doc in query_docs]} }

    # 准备好 X_train 和 y_train
    feature_selector ={"_id": 0}
    for item in feature_mi:
        feature_selector[item[0]] = 1 

    X_train_doc = db["features"].find(doc_query, feature_selector)
    X_df = pd.DataFrame([doc for doc in X_train_doc])

    label_selector ={"_id": 0}
    for item in label_mi:
        label_selector[item[0]] = 1 

    y_train_doc =  db["labels"].find(doc_query, label_selector)
    y_df = pd.DataFrame([doc for doc in y_train_doc])

    echart_data = []
    for feat in X_df.columns:
        for labl in y_df.columns:
            echart_data.append([feat, labl, normalized_mutual_info_score(X_df[feat].to_numpy(), y_df[labl].to_numpy())])

    echart_x=list(X_df.columns)
    echart_y=list(y_df.columns)

    return ujson.dumps({"dataset": dfname, "dataset_type": dftype, "echart_x": echart_x,
    "echart_y": echart_y, "echart_data": echart_data, "filter_id": group_id})

@app.route("/example/<string:dfname>/<string:dftype>/<string:exid>", methods=["GET","POST"])
def example_details(dfname, dftype, exid):
    db = mongo.cx[f"{dfname}_{dftype}"]
    label_arr = dict(db["labels"].find_one({"_id": {"$eq": exid}}))
    del label_arr["_id"]
    return {"id": exid, "label": label_arr, "label_sum": sum(label_arr.values())}

@app.route("/group/list/<string:dfname>/<string:dftype>")
def list_group_name(dfname, dftype):
    db = mongo.cx[f"{dfname}_{dftype}"]
    docs = db["graph_filters"].find({})
    try:
        res = [{"id": str(doc["_id"]), "group_name": doc["group_name"]} for doc in docs]
    except:
        res = None
    return ujson.dumps(res)

@app.route("/group/load/<string:dfname>/<string:dftype>")
def load_group(dfname, dftype):
    db = mongo.cx[f"{dfname}_{dftype}"]
    docs = db["graph_filters"].find({})
    graph_filters = []
    for doc in docs:
        if doc["group_type"] == "selected":
            item = ujson.loads(json_util.dumps(doc))
        elif doc["group_type"] == "query":
            item = ujson.loads(json_util.dumps(doc))
            query_result = []
            query_docs = db["labels"].find(item["query"])
            print(item["query"])
            for query_doc in query_docs:
                query_result.append(query_doc["_id"])
            item["points"] = query_result
        elif doc["group_type"] == "selected_combination":
            item = ujson.loads(json_util.dumps(doc))
        graph_filters.append(item)
    return {"dataset_name": dfname, "dataset_type": dftype, "groups": graph_filters}

@app.route("/group/add",methods=["POST"])
def add_group():
    request_body = ujson.loads(request.data)
    dfname = request_body["dataset_name"]
    dftype = request_body["dataset_type"]
    group_name = request_body["group_name"]
    group_type = request_body["group_type"]

    db = mongo.cx[f"{dfname}_{dftype}"]

    if group_type == "selected":
        points = request_body["points"]   
        db["graph_filters"].insert_one({"dataset_name": dfname, "dataset_type": dftype,
    "group_name": group_name, "group_type": group_type, "points": points})
    elif group_type == "query":
        query = request_body["query"]
        db["graph_filters"].insert_one({"dataset_name": dfname, "dataset_type": dftype,
    "group_name": group_name, "group_type": group_type, "query": query})
    elif group_type == "selected_combination":
        comb_ids = request_body["points"]
        comb_ids = [ObjectId(x) for x in comb_ids]
        points_doc = db["tsne_labels_combination"].find({"_id":{"$in": comb_ids}})
        points = []
        for doc in points_doc:
            points += doc["_member"]
        db["graph_filters"].insert_one({"dataset_name": dfname, "dataset_type": dftype,
    "group_name": group_name, "group_type": group_type, "points": points})

    return {"status": "success"}

@app.route("/group/remove",methods=["POST"])
def remove_group():
    request_body = ujson.loads(request.data)
    dfname = request_body["dataset_name"]
    dftype = request_body["dataset_type"]
    group_id = request_body["group_id"]

    db = mongo.cx[f"{dfname}_{dftype}"]
    col = db["graph_filters"]
    col.delete_one({"_id": ObjectId(group_id)})

    return {"status": "success"}

@app.route("/available/list")
def available_list():
    db = mongo.cx["config"]
    docs = db["avaliable_dataset"].find({})
    res = []
    for doc in docs:
        feature_counts = sum(doc["features_count"].values())
        res.append({"dataset_name": doc["dataset_name"],
        "features_count": feature_counts,
        "labels_count": len(doc["label_names"])})
    resp = ujson.dumps(res)
    return resp

@app.route("/available/labels/<string:dfname>")
def get_label_names(dfname):
    db = mongo.cx[f"{dfname}_train"]
    doc = db["labels"].find_one({})
    label_names = list(doc.keys())
    label_names.remove("_id")
    return {"labels": label_names}

@app.route("/available/features/<string:dfname>")
def get_feature_names(dfname):
    db = mongo.cx[f"{dfname}_train"]
    doc = db["features"].find_one({})
    feature_names = list(doc.keys())
    feature_names.remove("_id")
    return {"features": feature_names}

@app.route("/feature/get_point/<string:dfname>/<string:dftype>")
def get_point(dfname,dftype):
    point_id = request.args.get("point_id")

    db = mongo.cx[f"{dfname}_{dftype}"]
    doc = db["labels"].find_one({"_id": {"$eq": point_id}})
    result = [ k for k,v in doc.items() if v == 1 ]

    return ujson.dumps(result)

@app.route("/feature/count_100", methods=["POST"])
def get_feature_count_100():
    request_body = ujson.loads(request.data)
    dfname = request_body["dataset_name"]
    dftype = request_body["dataset_type"]
    try:
        group_id = request_body["group_id"]
        #query = request_body["query"]
    except KeyError:
        group_id = "none"
        #query = None
    db = mongo.cx[f"cache_{dfname}_{dftype}"]
    
    if group_id == "none":
        docs = db[group_id].find({}).sort("count",-1).limit(100)
    else:
        top100_docs = db["none"].find({}).sort("count",-1).limit(100)
        top100_words = [ doc["key"] for doc in top100_docs ]
        docs = db[group_id].find({"key": {"$in": top100_words}})
    
    res=[{"key": doc["key"], "count": doc["count"]} for doc in docs ]
    
    return {"group_id": group_id, "result": res}
    
@app.route("/group/mutual_information")
def group_mutual_information():
    dfname = request.args.get("dataset_name")
    dftype = request.args.get("dataset_type")
    group_id = request.args.get("group_id")
    space = request.args.get("space")

    result = get_mutual_information(dfname, dftype, space, group_id) 

    return {"group_id": group_id, "result": result}

@app.route("/group/label")
def group_label():
    dfname = request.args.get("dataset_name")
    dftype = request.args.get("dataset_type")
    group_id = request.args.get("group_id")

    db = mongo.cx[f"{dfname}_{dftype}"]

    print(dfname,dftype)

    return 0

@app.route("/train/classifier",methods=["POST"])
def train_classifier():
    # 从 post 的请求 body 中获取参数
    request_body = ujson.loads(request.data)
    dfname = request_body["dataset_name"]
    select_feature = request_body["select_feature"]
    select_label = request_body["select_label"]
    classifer = request_body["classifier"]
    br_kernel = request_body["br_kernel"]

    db_train = mongo.cx[f"{dfname}_train"]
    db_test = mongo.cx[f"{dfname}_test"]

    # build X_train and X_test
    feature_selector = {"_id": 0}
    for feature in select_feature:
        feature_selector[feature] = 1

    X_train_doc =  db_train["features"].find({}, feature_selector)
    X_train_df = pd.DataFrame([doc for doc in X_train_doc])

    X_test_doc =  db_test["features"].find({}, feature_selector)
    X_test_df = pd.DataFrame([doc for doc in X_test_doc])

    # build y_train and y_test
    label_selector = {"_id": 0}
    for label in select_label:
        label_selector[label] = 1

    y_train_doc =  db_train["labels"].find({}, label_selector)
    y_train_df = pd.DataFrame([doc for doc in y_train_doc])

    y_test_doc =  db_test["labels"].find({}, label_selector)
    y_test_df = pd.DataFrame([doc for doc in y_test_doc])

    # 训练 Classifier
    if classifer == "BinaryRelevance":
        clf = BinaryRelevance(classifier=copy.deepcopy(br_kernel_dict[br_kernel]),require_dense=[False, True])
        clf.fit(X_train_df, y_train_df)

    prediction = clf.predict(X_test_df)
    prediction = prediction.todense()
    y_test_df = y_test_df.to_numpy()

    hamming_loss = metrics.hamming_loss(y_test_df, prediction)
    accuracy_score = metrics.accuracy_score(y_test_df, prediction)
    hamming_score_ = hamming_score(y_test_df, prediction)

    return {"dataset_name": dfname,"select_feature": select_feature,
    "select_label": select_label, "hamming_loss": hamming_loss,
    "accuracy_score": accuracy_score, "hamming_score": hamming_score_, "br_kernel": br_kernel}

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5001)