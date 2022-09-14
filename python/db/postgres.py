import os.path

import gin

EVAL_CATEGORIES = ["control", "translation", "rotation", "input_noise", "output_noise"]

@gin.configurable
class DBEngine:
    def __init__(self, enable, host, dbname, user, password):
        self.enable = enable
        self.password = password
        self.user = user
        self.dbname = dbname
        self.host = host
        if enable:
            from sqlalchemy import create_engine
            self.engine = create_engine('postgresql://{}:{}@{}:5432/{}'.format(user, password, host, dbname))


def initialize_db():
    gin.parse_config_file(os.path.join(
        os.path.dirname(os.path.realpath(os.path.realpath(__file__))),
        "db_connection.gin"))

