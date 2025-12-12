from sklearn.model_selection import train_test_split, GridSearchCV ,RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score,accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import joblib
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


def run():
    x_train=np.load('./data/embeddings/train_embeddings.npy')
    x_test=np.load('./data/embeddings/test_embeddings.npy')


    metadata_train=pd.read_csv("./data/metadata/train_metadata.csv")
    metadata_test=pd.read_csv("./data/metadata/test_metadata.csv")
    y_test=metadata_test["label"].values
    y_train=metadata_train["label"].values
    model = KNeighborsClassifier()

    model.fit(x_train,y_train)
    y_pred_train=model.predict(x_train)
    y_pred_test=model.predict(x_test)

    joblib.dump(model,f'./models/{name}_model.joblib')
    print("="*50)
   

    print(f"=== {name} ===")
    print("="*50)

    print("Model Accuracy:\n")
    print(f"accuracy_score train:\n{accuracy_score(y_train,y_pred_train)}\n")

    print(f"accuracy_score test:\n{accuracy_score(y_test,y_pred_test)}\n")
    print("="*50)
    print("="*50)

    print("Confusion Matrix:\n")

    print(f"confusion_matrix train : \n{confusion_matrix(y_train,y_pred_train)}")
    print("="*50)
    print(f"confusion_matrix test :\n{confusion_matrix(y_test,y_pred_test)}")
    print("="*50)
    print("="*50)

    print("classification report:\n")

    print(f"classification_report train \n:{classification_report(y_train,y_pred_train)}")
    print("="*50)
    print(f"classification_report test \n:{classification_report(y_test,y_pred_test)}")
    print("\n")
