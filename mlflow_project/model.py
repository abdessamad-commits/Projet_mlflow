
import os
import pickle
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the file

    try:
        data = pd.read_csv("Ecommerce Customers")
    except Exception as e:
        logger.exception(
            "Unable to read the data set : %s", e
        )

    # Split the data into training and test sets.
    train, test = train_test_split(data)

    X = data[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
    y = data['Yearly Amount Spent']

    from sklearn.model_selection import train_test_split

    # In[16]:

    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=101)


    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.2
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.4

    with mlflow.start_run():
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)

        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        if tracking_url_type_store != "file":

            # Register the model
            mlflow.sklearn.log_model(lr, "model", registered_model_name="ElasticnetModel")
        else:
            mlflow.sklearn.log_model(lr, "model")

# Saving model to disk
pickle.dump(lr, open('model.pkl1','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl1','rb'))
print(model.predict([[32.187812, 14.715388,	38.244115,	1.516576]]))