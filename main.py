import matplotlib.pyplot as plt
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

def pr_auc(model):
    #this is the performance evaluation
    model.fit(X_train, y_train)

    # y_pred = model.predict(X_test)

    # precision, recall, thresholds = metrics.precision_recall_curve(y_test, y_pred)
    precision, recall, thresholds = metrics.precision_recall_curve(y_test, model.predict_proba(X_test)[:, 1])
    pr_auc = metrics.auc(recall, precision)
    return pr_auc

#importing the dataset
ds = fetch_ucirepo(id=827)
X = ds.data.features
y = ds.data.targets.to_numpy().reshape(-1)

#seplitting the dataset into traning and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#definening ML algorithms
#PR AUC 0.966167871245655
bc = BaggingClassifier(LogisticRegression(), n_estimators=51, n_jobs=-1)

# PR AUC:  0.9662133301201661
adaboost = AdaBoostClassifier(estimator=LogisticRegression(),n_estimators=1000, algorithm="SAMME.R", learning_rate=0.16)

# PR AUC: 0.963812352780165
rfc = RandomForestClassifier(class_weight='balanced_subsample', criterion='entropy', n_estimators=500, n_jobs=-1)

# PR AUC:  0.9666159098763603
##################### the best model
knn = KNeighborsClassifier(n_neighbors=667, algorithm="ball_tree", weights="uniform")

# PR AUC:  0.9627073074947889
lg = LogisticRegression()

# hmm = GaussianHMM(3, covariance_type="full")
# hmm.fit(X_train, y_train)

# PR AUC:  0.9627073074947889
xgb = XGBClassifier(n_estimators=500, booster="gblinear")

# PR AUC:  0.9627049635325982
vc = VotingClassifier([("bc", bc), ("adaboost", adaboost), ("knn", knn)], voting="soft", n_jobs=-1)

###################################################################
# parameters tuning testing 
# params = {"n_estimators": [x for x in range(1000, 10000, 1000)], "booster": ["gblinear"]}

# grid_search = GridSearchCV(XGBClassifier(), params, scoring='roc_auc', n_jobs=-1)
# grid_search.fit(X_train, y_train)

# best_model = grid_search.best_estimator_

# print(grid_search.best_estimator_)
# print(grid_search.best_params_)
# print(pr_auc(best_model))
print(pr_auc(xgb))