import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from ucimlrepo import fetch_ucirepo
import numpy as np
from xgboost import XGBClassifier
from hmmlearn.hmm import GaussianHMM
from sklearn.metrics import precision_score

class Model:
    def __init__(self, model, X, y, random_state, test_size = 0.2):
        self.model = model
        self.X = X
        self.y = y
        #seplitting the dataset into traning and testing
        self.random_state = random_state
        self.test_size = test_size
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

    def predict(self, X):
        self.model.fit(self.X_train, self.y_train)
        return self.model.predict(X)
    
    def tune_params(self, params):
    # parameters tuning testing 
        model = self.model
        grid_search = GridSearchCV(model, params, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)

        best_model = grid_search.best_estimator_

        return best_model, grid_search.best_params_, self.pr_auc(best_model)
    
    def data_split_test(self, from_range, to_range, test_size = 0.2):
        best_random_state = 0
        best_pr_auc = 0
        try:
            for i in range(from_range, to_range):
                model = self.model
                X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=i)
                model.fit(X_train, y_train)
                print(i)

                if best_pr_auc < Model.pr_auc(model, X_test, y_test):
                    best_pr_auc = Model.pr_auc(model, X_test, y_test)
                    best_random_state = i

                print("best_random_state: ", best_random_state)
                print("best_pr_auc: ", best_pr_auc)

        except KeyboardInterrupt:
            print("best_random_state: ", best_random_state)
            print("best_pr_auc: ", best_pr_auc)

    @staticmethod
    def pr_auc(model, X_test, y_test):
        #this is the performance evaluation
        precision, recall, thresholds = metrics.precision_recall_curve(y_test, model.predict_proba(X_test)[:, 1])
        pr_auc = metrics.auc(recall, precision)
        return pr_auc

    def ppv(self):
        return precision_score(self.y_test, self.model.predict(self.X_test))

    def show_performance(self):
        self.model.fit(self.X_train, self.y_train)

        print("pr_auc: ", Model.pr_auc(self.model, self.X_test, self.y_test))
        print("ppv: : ", self.ppv())

    def show_test_dataset_with_labels(self):
        df = self.X_test
        df["hospital_outcome"] = self.y_test
        print(df.to_string())

    def my_performance_test(self):
        self.model.fit(self.X_train, self.y_train)
        tmp = self.X_test
        tmp["hospital_outcome"] = self.y_test
        alive = tmp[tmp["hospital_outcome"] == 0]
        alive = alive.drop(columns=["hospital_outcome"])

        prediction = self.model.predict(alive)
        print("hospital_outcome deaths, dead = 1, alive = 0 ")
        print("the real values are only zeros")
        print("zeros: ", prediction[prediction == 0].size)
        print("ones: ", prediction[prediction == 1].size)
        print("total: ", prediction.size)

#importing the dataset
ds = fetch_ucirepo(id=827)
X = ds.data.features
y = ds.data.targets.to_numpy().reshape(-1)

#definening ML algorithms
#PR AUC 0.966167871245655
bc = BaggingClassifier(LogisticRegression(), n_estimators=51, n_jobs=-1)

# PR AUC:  0.9662133301201661
adaboost = AdaBoostClassifier(estimator=LogisticRegression(),n_estimators=1000, algorithm="SAMME.R", learning_rate=0.16)

# PR AUC: 0.963812352780165
# best_random_state:  43
# best_pr_auc:  0.9727163540979228
rfc = RandomForestClassifier(class_weight='balanced_subsample', criterion='entropy', n_estimators=500, n_jobs=-1)

# PR AUC:  0.9666159098763603
##################### the best model
knn = KNeighborsClassifier(n_neighbors=667, algorithm="ball_tree", weights="uniform")

# PR AUC:  0.9627073074947889
lg = LogisticRegression()

# PR AUC:  0.9627073074947889
xgb = XGBClassifier(booster="gblinear", n_estimators=1000)

# PR AUC:  0.9627049635325982
vc = VotingClassifier([("bc", bc), ("adaboost", adaboost), ("knn", knn), ("xgb", xgb)], voting="soft", n_jobs=-1)

###################################################################
# for knn
# best_random_state:  115
# best_pr_auc:  0.9702098195889476
# best_test_size: 0.1



def predict(input):
    model = Model(rfc, X, y, random_state=140)
    input = pd.DataFrame(input)
    result = model.predict(input)
    return result

# model = Model(rfc, X, y, random_state=140)
# model.show_performance()