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
در بخش پروژه‌ی MLflow  
