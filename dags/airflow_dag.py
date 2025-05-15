from airflow.decorators import dag, task
from airflow.utils.dates import days_ago

# Import your existing methods
from python_scripts.app import fetch_puuids, fetch_matches_to_search, fetch_matches

@dag(schedule_interval=None, start_date=days_ago(1), catchup=False, tags=["whichcompwon"])
def fetch_match_data_pipeline():

    @task
    def task_fetch_puuids():
        fetch_puuids()

    @task
    def task_fetch_matches_to_search():
        fetch_matches_to_search()

    @task
    def task_fetch_matches():
        fetch_matches()

    # Set execution order
    puuids = task_fetch_puuids()
    matches_to_search = task_fetch_matches_to_search()
    puuids >> matches_to_search >> task_fetch_matches()

dag = fetch_match_data_pipeline()
