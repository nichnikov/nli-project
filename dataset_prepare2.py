import os
import pandas as pd
import numpy as np
from copy import deepcopy
from random import shuffle
from src.config import PROJECT_ROOT_DIR

queries_df = pd.read_csv(os.path.join(PROJECT_ROOT_DIR, "data", "unique_queries.csv"), sep="\t")
# ids_answrs = list(zip(list(queries_df["FastAnswId"])[:1000], list(queries_df["lem_texts"])[:1000]))

ids_answrs_df = queries_df[["FastAnswId", "lem_texts"]]

# left_id	left_texts	right_id	right_texts	label


unique_ids = set(list(ids_answrs_df["FastAnswId"]))
print("количество уникальных айди ответов:", len(unique_ids))

# Создание похожих пар (парафраз), по всем айди быстрого ответа
# получается 32'432'344 парафразов
paraphrases = []
for ans_id in unique_ids:
    temp_df = ids_answrs_df[ids_answrs_df["FastAnswId"] == ans_id]
    temp_texts = list(temp_df["lem_texts"])[:20]
    temp_paraphrases = [tuple(sorted([tx1, tx2])) for tx1 in temp_texts for tx2 in temp_texts
                        if type(tx1) is str and type(tx2) is str]
    paraphrases += list(set(temp_paraphrases))

print(paraphrases[:20])
print("количество позитивных примеров (каждый текст отсортирован по алфавиту):", len(set(paraphrases)))

no_unique_ids_pairs = list(set([tuple(sorted([id1, id2])) for id1 in unique_ids for id2 in unique_ids if id1 != id2]))
print(no_unique_ids_pairs[:10])
print("количество не уникальных пар индексов быстрых ответов", len(no_unique_ids_pairs))

shuffle(no_unique_ids_pairs)

no_paraphrases = []
for id1, id2 in no_unique_ids_pairs[:500000]:
    temp1 = list(ids_answrs_df[ids_answrs_df["FastAnswId"] == id1]["lem_texts"])[:1]
    temp2 = list(ids_answrs_df[ids_answrs_df["FastAnswId"] == id2]["lem_texts"])[:1]
    shuffle(temp1)
    shuffle(temp2)
    temp_paraphrases = [tuple(sorted([tx1, tx2])) for tx1, tx2 in zip(temp1, temp2) if type(tx1) is str
                        and type(tx1) is str]
    no_paraphrases += temp_paraphrases

print(no_paraphrases[:10])
print("количество позитивных примеров (каждый текст отсортирован по алфавиту): ", len(no_paraphrases))

dataset = [(tx1, tx2, 1) for tx1, tx2 in paraphrases] + [(tx1, tx2, 0) for tx1, tx2 in no_paraphrases]

dataset_df = pd.DataFrame(dataset, columns=["text1", "text2", "label"])
dataset_df.to_csv(os.path.join("data", "dataset_for_paraphrases2.csv"), sep="\t", index=False)

# Замечание: т. к. фасттекст просто усредняет векторы слов н-граммы, последовательность слов не должна играть роли
# Для других моделей (например для трансформеров, лучше не сортировать тексты

"""
print("количество позитивных примеров:", sum(df["label"]))
df.to_csv(os.path.join(PROJECT_ROOT_DIR, "data", "dataset.csv"), sep="\t", index=False)

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

df_0 = df[df["label"]==0]
no_paraphrases = [(tuple(sorted([x, y]))) for x, y in zip(df_0["left_texts"], df_0["right_texts"]) if type(x) is str and
                  type(y) is str]
print(no_paraphrases[:10])
print("количество не похожих предложений:", len(set(no_paraphrases)))


from itertools import groupby
for k, v in groupby(sorted(ids_answrs, key=lambda x: x[0]), key=lambda y: y[0]):
    print(k, list([x[1] for x in v]))"""
