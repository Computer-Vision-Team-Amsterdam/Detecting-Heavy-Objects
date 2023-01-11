from datetime import datetime, timedelta
from typing import Tuple


YMD_FORMAT = "%Y-%m-%d"


def get_start_date(arg_date: str) -> Tuple[str, str]:
    """Start date of the Airflow DAG in different formats
    Args:
        arg_date: Start date, string of form %Y-%m-%d %H:%M:%S.%f
    """
    start_date = datetime.strptime(arg_date, "%Y-%m-%d %H:%M:%S.%f")
    my_format = "%Y-%m-%d_%H-%M-%S"
    start_date_dag = start_date.strftime(my_format)
    start_date_dag_ymd = start_date.strftime(YMD_FORMAT)

    return start_date_dag, start_date_dag_ymd
