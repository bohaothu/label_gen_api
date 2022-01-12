import copy
import pandas as pd

class partialBR:

    def __init__(self, classifier, features_name, labels_name):
        self.classifier = classifier
        self.features_name = features_name
        self.labels_name = labels_name

    def fit(self,X,y,params):
        """
        params example
        [{"x_subset": ["network","networks"], 
        "y": "network"}]
        """
        self.params = params # store classifier for each params

        for item in self.params:
            classifier = copy.deepcopy(self.classifier)
            X_subset = X[item["x_subset"]].to_numpy()
            y_subset = y[item["y"]].to_numpy()
            classifier.fit(X_subset, y_subset)
            item["classifier"] = classifier
        return self
    
    def predict(self,X):

        predictions = pd.DataFrame(index=X.index)

        for item in self.params:
            X_subset = X[item["x_subset"]].to_numpy()
            predictions[item["y"]] = item["classifier"].predict(X_subset)
        
        return predictions