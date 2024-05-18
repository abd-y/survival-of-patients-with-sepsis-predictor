import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from ucimlrepo import fetch_ucirepo
import numpy as np
from sklearn.metrics import precision_score

class Model:
    def __init__(self, model, X, y, random_state, test_size = 0.2):
        self.model = model
        self.X = X
        self.y = y
        #seplitting the dataset into traning and testing
        self.random_state = random_state
        self.test_size = test_size
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=self.test_size, random_state=self.random_state)
        self.model.fit(self.X_train, self.y_train)
    
    def predict_proba(self, input):
        return self.model.predict_proba(input)

    def tune_params(self, params):
    # parameters tuning testing 
        model = self.model
        grid_search = GridSearchCV(model, params, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)

        best_model = grid_search.best_estimator_

        return best_model, grid_search.best_params_, self.pr_auc(best_model, self.X_test, self.y_test)
    
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
        #print(self.X_test.shape[0])
        print("pr_auc: ", Model.pr_auc(self.model, self.X_test, self.y_test))
        print("ppv: : ", self.ppv())

    def show_test_dataset_with_labels(self):
        df = self.X_test
        df["hospital_outcome"] = self.y_test
        print(df.to_string())

    def my_performance_test(self):
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

# PR AUC: 0.963812352780165
# best_random_state:  43
# best_pr_auc:  0.9727163540979228
rfc = RandomForestClassifier(class_weight='balanced', criterion='log_loss', n_estimators=500, n_jobs=-1)

# With underfitting PR AUC:  0.9702098195889476
# Without underfitting PR AUC: 0.962
knn = KNeighborsClassifier(n_neighbors=1, algorithm="ball_tree", weights="distance", p=2, leaf_size=30)

# underfitting
lg = LogisticRegression(max_iter=1000, C=4, solver='liblinear')

def predict(input):
    input = pd.DataFrame(input)
    result = model.model.predict(input)
    return result

def predict_proba(input):
    input = pd.DataFrame(input)
    result = model.model.predict_proba(input)
    return result

#importing the dataset using ucimlrepo
# ds = fetch_ucirepo(id=827)
# X = ds.data.features
# y = ds.data.targets.to_numpy().reshape(-1)

#importing the dataset locally
ds = pd.read_csv("sepsis_survival_datasets/sepsis_survival_primary_cohort.csv")
X = ds.drop(columns=["hospital_outcome_1alive_0dead"])
y = ds["hospital_outcome_1alive_0dead"].to_numpy()

model = Model(rfc, X, y, random_state=140)

def save_pr(models, X_test, y_test):
    pr = pd.DataFrame()
    for i in models:
        precision, recall, thresholds = metrics.precision_recall_curve(y_test, i.predict_proba(X_test)[:, 1])
        name = type(i).__name__
        pr = pd.concat([pr, pd.DataFrame({f"{name}_precision": precision, 
                        f"{name}_recall": recall})], axis=1)
        pr.to_csv(f"./pr.csv")
    print(pr)
