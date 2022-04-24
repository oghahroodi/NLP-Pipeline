# NLP-Pipeline-Tutorial
# مستندات ساخت پایپلاین پروژه‌های پردازش متن با استفاده از MLflow و Airflow
##مقدمه
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

