import os
import re
import json
from datetime import datetime
from src.storage import DataFromDB
from src.config import PROJECT_ROOT_DIR
import pandas as pd

with open(os.path.join(PROJECT_ROOT_DIR, "data", "statistics_parameters.json")) as st_f:
    stat_prmtrs = json.load(st_f)

"""1) добавление эталонов и ответов"""
db_credentials = stat_prmtrs["db_credentials"]
db_con = DataFromDB(**db_credentials)

today = datetime.today().strftime('%Y-%m-%d')
rows = db_con.get_rows(1, today)

patterns = [r"\xa0"]
# patterns_ = re.compile("|".join([r"\b" + tx + r"\b" for tx in patterns])
patterns_ = re.compile("|".join([pt for pt in patterns]))

txts = [patterns_.sub("", r.ShortAnswerText) for r in rows[:10]]
txts_df = pd.DataFrame(txts, columns=["ShortAnswer"])

print(txts_df)
txts_df.to_csv(os.path.join("data", "short_answers.csv"))