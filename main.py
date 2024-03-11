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
from sklearn import metrics
from ucimlrepo import fetch_ucirepo
import numpy as np
from xgboost import XGBClassifier

ds = fetch_ucirepo(id=827)
X = ds.data.features
y = ds.data.targets.to_numpy().reshape(-1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#score 0.9267847432610891
# bc = BaggingClassifier(RandomForestClassifier(), n_estimators=60, max_samples=1.0, bootstrap=True, n_jobs=-1)
# bc.fit(X_train, y_train)

#score 0.9267200082860768
# vc = VotingClassifier([("lg", LogisticRegression()), ("rclf", RandomForestClassifier()), ("svc", SVC(probability=True))], voting="soft")
# vc.fit(X_train, y_train)

# score 0.926810637251094
# PR AUC:  0.9626837803706074
# rfc = RandomForestClassifier()
# rfc.fit(X_train, y_train)

#score 0.9267200082860768
# svc = SVC()
# svc.fit(X_train, y_train)

# knn = KNeighborsClassifier()
# knn.fit(X_train, y_train)

# score:  0.925414614989578
# PR AUC:  0.9627073074947889
lg = LogisticRegression()
lg.fit(X_train, y_train)

# score:  0.925414614989578
# AUC:  0.7006591336278148
# xgb = XGBClassifier(n_estimators=500, max_depth=1, objective="")
# xgb.fit(X_train, y_train)

# print("score: ", rfc.score(X_test, y_test))

# print("AUC: ", metrics.roc_auc_score(y_test, rfc.predict_proba(X_test)[:, 1]))

y_pred = lg.predict(X_test)

precision, recall, thresholds = metrics.precision_recall_curve(y_test, lg.predict(X_test))
# precision, recall, thresholds = metrics.precision_recall_curve(y_test, lg.predict_proba(X_test)[:, 1])
pr_auc = metrics.auc(recall, precision)
print("PR AUC: ", pr_auc)