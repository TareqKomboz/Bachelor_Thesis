import numpy as np
import pandas as pd

from objective_functions.tf_objective_functions import FUNCTIONS
from db.postgres import initialize_db, DBEngine

COLUMNS = ["algorithm", "trained_on", "run_id", "step_counter", "run_on", "mean_perf_over_time", "std_perf_over_time"]

TABLE_NAME = 'runs'


def read_from_sql(non_learned, algorithms, trained_ons, run_ids, run_ons):
    initialize_db()
    engine = DBEngine().engine

    sql_query = "SELECT * FROM {} WHERE ".format(TABLE_NAME)
    if non_learned:
        trained_ons.append("")
        run_ids.append("")
    if 'all' in run_ons:
        run_ons = FUNCTIONS.keys()

    sql_query += "("
    for algorithm in algorithms:
        sql_query += "algorithm LIKE '{}' OR ".format(algorithm)
    sql_query = sql_query.strip("OR ")
    sql_query += ")"
    if trained_ons is not None:
        sql_query += " and ("
        for trained_on in trained_ons:
            sql_query += "trained_on LIKE '{}' OR ".format(trained_on)
        sql_query = sql_query.strip("OR ")
        sql_query += ")"
    if run_ids is not None:
        sql_query += " and ("
        for run_id in run_ids:
            sql_query += "run_id LIKE '{}' OR ".format(run_id)
        sql_query = sql_query.strip("OR ")
        sql_query += ")"
    sql_query += " and ("
    for run_on in run_ons:
        sql_query += "run_on LIKE '{}' OR ".format(run_on)
    sql_query = sql_query.strip("OR ")
    sql_query += ") and ("
    sql_query = sql_query.strip("OR ")
    sql_query += ")"
    '''
    sql_query = "select * " \
                "FROM runs " \
                "WHERE (algorithm LIKE 'reinforce' and run_id LIKE '25_obs_giant_vnet_deep_pnet_rel_env' and trained_on Like 'all')"
    '''
    return pd.read_sql_query(sql_query, engine.connect())
