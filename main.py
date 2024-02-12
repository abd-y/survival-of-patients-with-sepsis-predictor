import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from ucimlrepo import fetch_ucirepo
import numpy as np

ds = fetch_ucirepo(id=827)
X = ds.data.features
y = ds.data.targets.to_numpy().reshape(-1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#score 0.9267847432610891
bc = BaggingClassifier(RandomForestClassifier(), n_estimators=40, max_samples=1.0, bootstrap=True, n_jobs=-1)
bc.fit(X_train, y_train)

#score 0.9267200082860768
# vc = VotingClassifier([("lg", LogisticRegression()), ("rclf", RandomForestClassifier()), ("svc", SVC())], voting="hard")
# vc.fit(X_train, y_train)

# score 0.926810637251094
# rfc = RandomForestClassifier()
# rfc.fit(X_train, y_train)

#score 0.9267200082860768
# svc = SVC()
# svc.fit(X_train, y_train)

print(X.info())
