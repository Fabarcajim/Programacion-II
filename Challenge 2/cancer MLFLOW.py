import warnings
import argparse
import logging
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import mlflow.sklearn 
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score,confusion_matrix
from sklearn.ensemble import RandomForestClassifier

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# evaluation function
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    
    data = pd.read_csv("data.csv")
    data.to_csv("data.csv", index=False)
    data.drop(columns=['id','Unnamed: 32'],inplace=True)

    columnas = data.select_dtypes(include=['float64', 'int64'])
    Q1 = columnas.quantile(0.25)
    Q3 = columnas.quantile(0.75)
    IQR = Q3 - Q1
    inf = Q1 - 1.5 * IQR
    sup = Q3 + 1.5 * IQR

    no_atipicos = (columnas >= inf) & (columnas <= sup)
    df_filtered = data[no_atipicos.all(axis=1)]

    x_pip=df_filtered.iloc[:,1:].copy() 
    y_pip=df_filtered['diagnosis']

    encoder=LabelEncoder()
    y_pip_encoder=encoder.fit_transform(y_pip)
    y_pip_encoder

    x_pip=df_filtered.iloc[:,1:].copy() 
    y_pip=df_filtered['diagnosis']
    y_pip_encoder=encoder.fit_transform(y_pip)
    x_train_pip, x_test_pip, y_train_pip, y_test_pip = train_test_split(x_pip, y_pip_encoder, test_size=0.3, shuffle=True, random_state=42)
    
    mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")

    print("the set tracking uri is",mlflow.get_tracking_uri())

    ###########First Model Naive Bayes #############

    print("First Model Naive Bayes")
    exp = mlflow.set_experiment(experiment_name="Naive Bayes")

    print("Name: {}".format(exp.name))
    print("Experiment_id: {}".format(exp.experiment_id))

    mlflow.start_run(run_name="run1.1")
    tags = {
        "engineering": "ML platform",
        "release.candidate":"RC1",
        "release.version": "2.0"
    }

    mlflow.set_tags(tags)

    current_run = mlflow.active_run()
    print("Active run id is {}".format(current_run.info.run_id))
    print("Active run name is {}".format(current_run.info.run_name))

    lr = GaussianNB()
    lr.fit(x_train_pip, y_train_pip)

    predicted_qualities = lr.predict(x_test_pip)
    (rmse, mae, r2) = eval_metrics(y_test_pip, predicted_qualities)

    ytest_predict_Std=lr.predict(x_test_pip)
    ytrain_predict_std=lr.predict(x_train_pip)
    precision=precision_score(y_test_pip,ytest_predict_Std,average='macro')
    recall=recall_score(y_test_pip,ytest_predict_Std,average='macro') #sensibildiad
    F1=f1_score(y_test_pip,ytest_predict_Std,average='macro')

    print('Accuracy Test:',accuracy_score(y_test_pip,ytest_predict_Std))
    print('Accuracy Train:',accuracy_score(y_train_pip,ytrain_predict_std))
    print('Presicion:',precision)
    print('Recall:',recall)
    print('F1_score:',F1)

    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)
   

    #log metrics
    metrics = {
        "rmse":rmse,
        "r2":r2,
        "mae":mae,
        "Accuracy":precision,
        "recall":recall,
        "F1_Score":F1
    }

    mlflow.log_metrics(metrics)
    #log model
    mlflow.sklearn.log_model(lr, "Model NaiveBayes")
    mlflow.log_artifacts("data.csv")

    artifacts_uri=mlflow.get_artifact_uri()
    print("The artifact path is",artifacts_uri )

    mlflow.end_run()

     ###########Second Model Random Forest #############

    print("Second Model Random Forest")
    exp = mlflow.set_experiment(experiment_name="Random Forest")

    print("Name: {}".format(exp.name))
    print("Experiment_id: {}".format(exp.experiment_id))

    mlflow.start_run(run_name="run1.1")
    tags = {
        "engineering": "ML platform",
        "release.candidate":"RC1",
        "release.version": "2.0"
    }

    mlflow.set_tags(tags)

    current_run = mlflow.active_run()
    print("Active run id is {}".format(current_run.info.run_id))
    print("Active run name is {}".format(current_run.info.run_name))

    lr = RandomForestClassifier(n_estimators=150, random_state=42)
    lr.fit(x_train_pip, y_train_pip)

    predicted_qualities = lr.predict(x_test_pip)
    (rmse, mae, r2) = eval_metrics(y_test_pip, predicted_qualities)

    ytest_predict_Std=lr.predict(x_test_pip)
    ytrain_predict_std=lr.predict(x_train_pip)
    precision=precision_score(y_test_pip,ytest_predict_Std,average='macro')
    recall=recall_score(y_test_pip,ytest_predict_Std,average='macro') #sensibildiad
    F1=f1_score(y_test_pip,ytest_predict_Std,average='macro')

    print('Accuracy Test:',accuracy_score(y_test_pip,ytest_predict_Std))
    print('Accuracy Train:',accuracy_score(y_train_pip,ytrain_predict_std))
    print('Presicion:',precision)
    print('Recall:',recall)
    print('F1_score:',F1)

    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)

    # log parameters
    params = {
        "n_estimators": 149,
        "random_state": 42,

    }
    mlflow.log_params(params)
   

    #log metrics
    metrics = {
        "rmse":rmse,
        "r2":r2,
        "mae":mae,
        "Accuracy":precision,
        "recall":recall,
        "F1_Score":F1
    }
    
    mlflow.log_metrics(metrics)
    #log model
    mlflow.sklearn.log_model(lr, "Random Forest")
    mlflow.log_artifacts("data.csv")

    artifacts_uri=mlflow.get_artifact_uri()
    print("The artifact path is",artifacts_uri )

    mlflow.end_run()

    ###########Third Model Linear Regresion #############

    print("Second Model Linear Regresion")
    exp = mlflow.set_experiment(experiment_name="Linear Regresion")

    print("Name: {}".format(exp.name))
    print("Experiment_id: {}".format(exp.experiment_id))

    mlflow.start_run(run_name="run1.1")
    tags = {
        "engineering": "ML platform",
        "release.candidate":"RC1",
        "release.version": "2.0"
    }

    mlflow.set_tags(tags)

    current_run = mlflow.active_run()
    print("Active run id is {}".format(current_run.info.run_id))
    print("Active run name is {}".format(current_run.info.run_name))

    lr = LogisticRegression()
    lr.fit(x_train_pip, y_train_pip)

    predicted_qualities = lr.predict(x_test_pip)
    (rmse, mae, r2) = eval_metrics(y_test_pip, predicted_qualities)

    ytest_predict_Std=lr.predict(x_test_pip)
    ytrain_predict_std=lr.predict(x_train_pip)
    precision=precision_score(y_test_pip,ytest_predict_Std,average='macro')
    recall=recall_score(y_test_pip,ytest_predict_Std,average='macro') #sensibildiad
    F1=f1_score(y_test_pip,ytest_predict_Std,average='macro')

    print('Accuracy Test:',accuracy_score(y_test_pip,ytest_predict_Std))
    print('Accuracy Train:',accuracy_score(y_train_pip,ytrain_predict_std))
    print('Presicion:',precision)
    print('Recall:',recall)
    print('F1_score:',F1)

    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)
   

    #log metrics
    metrics = {
        "rmse":rmse,
        "r2":r2,
        "mae":mae,
        "Accuracy":precision,
        "recall":recall,
        "F1_Score":F1
    }
    
    mlflow.log_metrics(metrics)
    #log model
    mlflow.sklearn.log_model(lr, "Linear Regresion")
    mlflow.log_artifact("E:/PROGRAMACION II/CHALLENGE/MLFLOW CURSE/data.csv")

    artifacts_uri=mlflow.get_artifact_uri()
    print("The artifact path is",artifacts_uri )

    mlflow.end_run()

    run = mlflow.last_active_run()
    print("Recent Active run id is {}".format(run.info.run_id))
    print("Recent Active run name is {}".format(run.info.run_name))