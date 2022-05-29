# NLP-Pipeline-Tutorial
# مستندات ساخت پایپلاین پروژه‌های پردازش متن با استفاده از MLflow و Airflow
## مقدمه
برای نصب MLflow از دستور زیر استفاده کنید.
```
pip install mlflow
```
برای استفاده از بخش ترکینگ می‌توان از کد زیر استفاده کرد.
```
import os
from random import random, randint
from mlflow import log_metric, log_param, log_artifacts

if __name__ == "__main__":
    # Log a parameter (key-value pair)
    log_param("param1", randint(0, 100))

    # Log a metric; metrics can be updated throughout the run
    log_metric("foo", random())
    log_metric("foo", random() + 1)
    log_metric("foo", random() + 2)

    # Log an artifact (output file)
    if not os.path.exists("outputs"):
        os.makedirs("outputs")
    with open("outputs/test.txt", "w") as f:
        f.write("hello world!")
    log_artifacts("outputs")
```
و در کد خود می‌توان پارامترهای خود را لاگ کرد و هنگامی که کد اجرا شود یک دایرکتوری mlrun در کنار فایل می‌سازد که تاریخچه‌ی لاگ‌ها را نگه می‌دارد و با استفاده از دستور زیر می‌توان یک UI برای مشاهده‌ی لاگ‌ها داشت.
```
mlflow ui
```

در بخش پروژه می‌توان کد و ماژول‌ها مورد نیازش را به شکل یک پکیچ درآورد.
هر پروژه شامل کد و یک فایل MLproject است که نیازمندی‌های کد را مشخص می‌کند و برای اجرای پروژه می‌توان از دستور زیر هم به شکل لوکال هم با استفاده از یک لینک گیت‌هاب استفاده کرد:
```
mlflow run sklearn_elasticnet_wine -P alpha=0.5

mlflow run https://github.com/mlflow/mlflow-example.git -P alpha=5.0
```
فایل MLproject به شکل زیر است:
```
name: tutorial

conda_env: conda.yaml

entry_points:
  main:
    parameters:
      alpha: {type: float, default: 0.5}
      l1_ratio: {type: float, default: 0.1}
    command: "python train.py {alpha} {l1_ratio}"
```
که نیازمندی‌ها و پرامتر‌های کد و دستور اجرای کد را شامل می‌شود.
فایل conda.yaml نیز به فرمت زیر است:
```
name: tutorial
channels:
  - conda-forge
dependencies:
  - python=3.7
  - pip
  - pip:
      - scikit-learn==0.23.2
      - mlflow>=1.0
      - pandas
```
در بخش مدل می‌توان مدل‌های مختلف را ذخیره و اجرا کرد به عنوان مثال با استفاده از کد زیر می‌توان مدل را ذخیره کرد:
```
import numpy as np
from sklearn.linear_model import LogisticRegression

import mlflow
import mlflow.sklearn

if __name__ == "__main__":
    X = np.array([-2, -1, 0, 1, 2, 1]).reshape(-1, 1)
    y = np.array([0, 0, 1, 1, 1, 0])
    lr = LogisticRegression()
    lr.fit(X, y)
    score = lr.score(X, y)
    print("Score: %s" % score)
    mlflow.log_metric("score", score)
    mlflow.sklearn.log_model(lr, "model")
    print("Model saved in run %s" % mlflow.active_run().info.run_uuid)
```
و مدل‌های ذخیره شده در فولدر mlrun قرار می‌گیرند و می‌توان با UI مدل‌های ذخیره شده را دید.
همچنین می‌توان مدل‌های مختلف را بر روی سرور اجرا کرد به عنوان مثال خود MLflow برای پایتون یک سرور ساده پیاده سازی کرده که با دستور زیر می‌توان آن را ساخت:
```
mlflow models serve -m runs:/<RUN_ID>/model
```
که RUN_ID همان آیدی مدلی است که قبلا ذخیره شده حال می‌توان به سرور ایجاد شده ریکوئست داد:
```
curl -d '{"columns":["x"], "data":[[1], [-1]]}' -H 'Content-Type: application/json; format=pandas-split' -X POST localhost:5000/invocations
```
همچنین می‌توان روی یک سرور خارجی ترک کردن را انجام داد:
```
import mlflow
mlflow.set_tracking_uri("http://YOUR-SERVER:4040")
mlflow.set_experiment("my-experiment")
```
## پیاده سازی یک پروژه‌ی نمونه با استفاده از MLflow
در این بخش یک رگرسیون خطی را روی MLflow پیاده سازی می‌کنیم.
کد پروژه به شکل زیر است:
```
# The data set used in this example is from http://archive.ics.uci.edu/ml/datasets/Wine+Quality
# P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
# Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

import os
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

    # Read the wine-quality csv file from the URL
    csv_url = (
        "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    )
    try:
        data = pd.read_csv(csv_url, sep=";")
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s", e
        )

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

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

        # Model registry does not work with file store
        if tracking_url_type_store != "file":

            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(lr, "model", registered_model_name="ElasticnetWineModel")
        else:
            mlflow.sklearn.log_model(lr, "model")
```
حال با دستور زیر تنظیمات مختلف را اجرا می‌کنیم:
```
python train.py <alpha> <l1_ratio>
```
و می‌توان نتیجه‌ی مدل‌های مختلف را در UI دید همچنین اگر لازم باشد که پروژه را با تنظیمات مشخص بتوان همواره اجرا کرد می‌توان از MLflow project استفاده کرد.
حال برای دیپلوی کردن مدل‌های ذخیره شده از MLflow model استفاده می‌کنیم که در از یک فرمت استاندارد برای ذخیره کردن مدل استفاده می‌شود و کتابخانه‌های مختلفی مانند SpaCy و PyTorch را پشتیبانی می‌کند. 
## AirFlow
برای نصب از دستور زیر استفاده کنید:
```
#Configurations
export AIRFLOW_HOME=${PWD}/airflow
AIRFLOW_VERSION=2.0.1
PYTHON_VERSION="$(python --version | cut -d " " -f 2 | cut -d "." -f 1-2)"
CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"

#Install Airflow (may need to upgrade pip)
pip install "apache-airflow==${AIRFLOW_VERSION}" --constraint "${CONSTRAINT_URL}"

#Initialize DB (SQLite by default)
airflow db init
```
یک فولدر با ساختار زیر ساخته می‌شود:
‍‍
airflow/
├── logs/
└── airflow.cfg
├── airflow.db
├── unittests.cfg
└── webserver_config.py

با دستور زیر یک کاربر ادمین برای دسترسی به دیتابیس ساخته می‌شود:
```
airflow users create \
    --username admin \
    --firstname  \
    --lastname  \
    --role Admin \
    --email 
```
همچنین با دستور زیر سرور اجرا می‌شود:
```
#Launch webserver
export AIRFLOW_HOME=${PWD}/airflow
airflow webserver --port 8080  # http://localhost:8080
```
در airflow می‌توان یک DAG برای مشخص کردن مراحل اجرای یک تسک به شکل زیر تعریف کرد:
‍```
mkdir airflow/dags
touch airflow/dags/example.py
```
