# airflow_dag.py (This goes into the 'dags/' folder in Airflow)
from airflow import DAG
from airflow.operators.python import PythonOperator

from datetime import datetime

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 5, 10),
    'retries': 1,
}

# Define the DAG
dag = DAG(
    'whichcompwon_dag',
    default_args=default_args,
    description='A DAG to fetch and process League of Legends data',
    schedule_interval='@monthly',  # or your preferred schedule
)

# Define tasks
fetch_task = PythonOperator(
    task_id='fetch_matches_match_data',
    python_callable=fetch_match_data,  # Function from your Flask app
    dag=dag,
)



# Task dependencies
fetch_task #>> train model >> move model to s3   # This means fetch_task runs before reload_task
