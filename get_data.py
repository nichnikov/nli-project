import os
import re
import json
from datetime import datetime
from src.storage import DataFromDB
from src.config import PROJECT_ROOT_DIR
import pandas as pd
from collections import namedtuple

ROW_FOR_ANSWERS = namedtuple("ROW_FOR_ANSWERS", "SysID, ID, ParentModuleID, "
                                                "ParentID, ChildBlockModuleID, ChildBlockID, ShortAnswerText")

with open(os.path.join(PROJECT_ROOT_DIR, "data", "statistics_parameters.json")) as st_f:
    stat_prmtrs = json.load(st_f)

"""1) добавление эталонов и ответов"""
db_credentials = stat_prmtrs["db_credentials"]
db_con = DataFromDB(**db_credentials)

today = datetime.today().strftime('%Y-%m-%d')
rows = db_con.get_rows(1, today)


rows_answers = [ROW_FOR_ANSWERS(r.SysID, r.ID, r.ParentModuleID, r.ParentID,
                                r.ChildBlockModuleID, r.ChildBlockID, r.ShortAnswerText) for r in rows]

rows_answers_unique = list(set(rows_answers))

patterns = [r"\xa0"]
# patterns_ = re.compile("|".join([r"\b" + tx + r"\b" for tx in patterns])
patterns_ = re.compile("|".join([pt for pt in patterns]))


queries_texts = [(r.ID, r.Cluster) for r in rows]
answers_txts = [(r.ID, patterns_.sub("", r.ShortAnswerText)) for r in rows_answers_unique]
answers_txts_df = pd.DataFrame(answers_txts, columns=["FastAnswId", "ShortAnswer"])

queries_txts_df = pd.DataFrame(queries_texts, columns=["FastAnswId", "Cluster"])
answers_txts_df.to_csv(os.path.join("data", "short_answers.csv"), sep="\t", index=False)
queries_txts_df.to_csv(os.path.join("data", "queries.csv"), sep="\t", index=False)