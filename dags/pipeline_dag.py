from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime


from pipeline.etl import run as etl_run
from pipeline.embeddings import run as embeddings_run
from pipeline.chroma_store import run as run_chroma_store
from pipeline.train import run as run_train


with DAG(
    "new_pipeline",
    start_date=datetime(2025,12,10),
    schedule_interval="*****",
    catchup=False
    ) as dag:

    t1=PythonOperator(
        task_id="etl",
        python_callable="etl_run"
    )

    t2 =PythonOperator(
        task_id="embeddings",
        python_callable="embeddings_run"
    )

    t3 = PythonOperator(
        task_id="chroma_store",
        python_callable="run_chroma_store"
    )

    t4 = PythonOperator(
        task_id="train",
        python_callable="run_train"
    )



t1>>t2>>t3>>t4