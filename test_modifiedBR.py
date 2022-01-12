from skmultilearn.dataset import load_dataset
from sklearn.neighbors import KNeighborsClassifier
from helper.modifiedBR import partialBR
import pandas as pd

X_train, y_train, feature_names, label_names = load_dataset("bibtex","train")
X_test, y_test, _, _ = load_dataset("bibtex","test")

feature_names = [x[0] for x in feature_names]
label_names = [y[0] for y in label_names]

X_train = pd.DataFrame(X_train.todense(), columns=feature_names)
y_train = pd.DataFrame(y_train.todense(), columns=label_names)

X_test = pd.DataFrame(X_test.todense(), columns=feature_names)
y_test = pd.DataFrame(y_test.todense(), columns=label_names)

params = [{"x_subset": ["networks", "network"], "y": "TAG_web"},
{"x_subset": ["networks", "network"], "y": "TAG_networks"}]

clf = partialBR(classifier=KNeighborsClassifier(), features_name=feature_names, labels_name=label_names)

clf.fit(X_train, y_train, params)
clf.predict(X_test)