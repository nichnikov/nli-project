import os
import pandas as pd
import numpy as np
from copy import deepcopy
from random import shuffle
from src.config import PROJECT_ROOT_DIR

queries_df = pd.read_csv(os.path.join(PROJECT_ROOT_DIR, "data", "unique_queries.csv"), sep="\t")
# ids_answrs = list(zip(list(queries_df["FastAnswId"])[:1000], list(queries_df["lem_texts"])[:1000]))

ids_answrs_df = queries_df[["FastAnswId", "lem_texts"]][:10000]

ids_answrs_left = list(zip(ids_answrs_df["FastAnswId"], ids_answrs_df["lem_texts"]))
shuffle(ids_answrs_left)

ids_answrs_right = deepcopy(list(zip(ids_answrs_df["FastAnswId"], ids_answrs_df["lem_texts"])))
shuffle(ids_answrs_right)


ids_answrs_left_df = pd.DataFrame(ids_answrs_left, columns=["left_id", "left_texts"])
ids_answrs_right_df = pd.DataFrame(ids_answrs_right, columns=["right_id", "right_texts"])
df = pd.concat([ids_answrs_left_df, ids_answrs_right_df], axis=1)

print(df)
df.drop_duplicates(inplace=True)
df['label'] = np.where((df['left_id'] == df['right_id']), 1, 0)

print("количество позитивных примеров:", sum(df["label"]))
df.to_csv(os.path.join(PROJECT_ROOT_DIR, "data", "dataset.csv"), sep="\t", index=False)


unique_ids = set(list(ids_answrs_df["FastAnswId"]))
print("количество уникальных айди ответов:", len(unique_ids))

paraphrases = []
for ans_id in set(list(ids_answrs_df["FastAnswId"])):
    temp_df = ids_answrs_df[ids_answrs_df["FastAnswId"] == ans_id]
    temp_texts = list(temp_df["lem_texts"])
    temp_paraphrases = [tuple(sorted([tx1, tx2])) for tx1 in temp_texts for tx2 in temp_texts]
    paraphrases += list(set(temp_paraphrases))

# Замечание: т. к. фасттекст просто усредняет векторы слов н-граммы, последовательность слов не должна играть роли
# Для других моделей (например для трансформеров, лучше не сортировать тексты
print(paraphrases[:20])
print("количество позитивных примеров (каждый текст отсортирован по алфавиту):", len(paraphrases))

"""
from itertools import groupby
for k, v in groupby(sorted(ids_answrs, key=lambda x: x[0]), key=lambda y: y[0]):
    print(k, list([x[1] for x in v]))"""
