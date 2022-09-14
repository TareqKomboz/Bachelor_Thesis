import pandas as pd

from db.postgres import initialize_db, DBEngine
from objective_functions.tf_objective_functions import FUNCTIONS

TABLE_NAME = "performance"

class Performance:
    def __init__(self, id: int, algorithm: str, trained_on: str, run_id: str, overall: float,
                 ackley: float, ackley_control: float, ackley_translation: float, ackley_rotation: float,
                 ackley_input_noise: float, ackley_output_noise: float,
                 himmelblau: float, himmelblau_control: float, himmelblau_translation: float,
                 himmelblau_rotation: float, himmelblau_input_noise: float, himmelblau_output_noise: float,
                 cross_in_tray: float, cross_in_tray_control: float, cross_in_tray_translation: float,
                 cross_in_tray_rotation: float, cross_in_tray_input_noise: float, cross_in_tray_output_noise: float,
                 rastrigin: float, rastrigin_control: float, rastrigin_translation: float, rastrigin_rotation: float,
                 rastrigin_input_noise: float, rastrigin_output_noise: float,
                 sphere: float, sphere_control: float, sphere_translation: float, sphere_rotation: float,
                 sphere_input_noise: float, sphere_output_noise: float,
                 camel: float, camel_control: float, camel_translation: float, camel_rotation: float,
                 camel_input_noise: float, camel_output_noise: float,
                 rosenbrock: float, rosenbrock_control: float, rosenbrock_translation: float,
                 rosenbrock_rotation: float, rosenbrock_input_noise: float, rosenbrock_output_noise: float,
                 michalewicz: float, michalewicz_control: float, michalewicz_translation: float,
                 michalewicz_rotation: float, michalewicz_input_noise: float, michalewicz_output_noise: float):
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
