import pandas as pd

from db.postgres import initialize_db, DBEngine
from objective_functions.tf_objective_functions import FUNCTIONS

TABLE_NAME = "performance"

class Performance:
    def __init__(self, id: int, algorithm: str, trained_on: str, run_id: str, overall: float,
                 ackley: float, ackley_control: float,
                 langermann: float, langermann_control: float,
                 rastrigin: float, rastrigin_control: float,
                 rosenbrock: float, rosenbrock_control: float,
                 michalewicz: float, michalewicz_control: float,
                 sumsquares: float, sumsquares_control: float):
        self.id = id

def read_from_sql_for_correlation_matrix(threshhold):
    initialize_db()
    engine = DBEngine().engine

    sql_query = "SELECT * FROM {} WHERE in_distribution > {} AND translation > 0.1 AND (".format(TABLE_NAME, threshhold)
    for function_name in FUNCTIONS.keys():
        sql_query += "trained_on LIKE '{}' OR ".format(function_name)
    sql_query = sql_query.strip(" OR ")
    sql_query += ")"
    return pd.read_sql_query(sql_query, engine.connect())
