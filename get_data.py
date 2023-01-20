import os
import json
from datetime import datetime
from src.storage import DataFromDB
from src.config import PROJECT_ROOT_DIR

with open(os.path.join(PROJECT_ROOT_DIR, "data", "statistics_parameters.json")) as st_f:
    stat_prmtrs = json.load(st_f)

"""1) добавление эталонов и ответов"""
db_credentials = stat_prmtrs["db_credentials"]
db_con = DataFromDB(**db_credentials)

today = datetime.today().strftime('%Y-%m-%d')
rows = db_con.get_rows(1, today)

print(rows[:10])
print(len(rows))