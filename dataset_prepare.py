import os
import pandas as pd
import numpy as np
from copy import deepcopy
from random import shuffle
from src.config import PROJECT_ROOT_DIR
from src.utils import jaccard_similarity

queries_df = pd.read_csv(os.path.join(PROJECT_ROOT_DIR, "data", "unique_queries.csv"), sep="\t")
# ids_answrs = list(zip(list(queries_df["FastAnswId"])[:1000], list(queries_df["lem_texts"])[:1000]))

ids_answrs_df = queries_df[["FastAnswId", "lem_texts"]]

'''
# не нужно, т. к. слишком мало получается
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

# left_id	left_texts	right_id	right_texts	label
df_0 = df[df["label"]==0]
no_paraphrases = [(tuple(sorted([x, y]))) for x, y in zip(df_0["left_texts"], df_0["right_texts"]) if type(x) is str and
                  type(y) is str]
print(no_paraphrases[:10])
print("количество не похожих предложений:", len(set(no_paraphrases)))

print("количество позитивных примеров:", sum(df["label"]))
df.to_csv(os.path.join(PROJECT_ROOT_DIR, "data", "dataset.csv"), sep="\t", index=False)
'''

unique_ids = set(list(ids_answrs_df["FastAnswId"]))
print("количество уникальных айди ответов:", len(unique_ids))

# Создание похожих пар (парафраз), по всем айди быстрого ответа
# получается 32'432'344 парафразов
paraphrases = []
for ans_id in unique_ids:
    temp_df = ids_answrs_df[ids_answrs_df["FastAnswId"] == ans_id]
    temp_texts = list(temp_df["lem_texts"])[:20]
    temp_paraphrases = [tuple(sorted([" ".join(sorted(tx1.split())),
                                      " ".join(sorted(tx2.split()))])) for tx1 in temp_texts for tx2 in temp_texts
                        if type(tx1) is str and type(tx2) is str]
    paraphrases += list(set(temp_paraphrases))

paraphrases = [(tx1, tx2) for tx1, tx2 in paraphrases if jaccard_similarity(tx1, tx2) < 0.5]
paraphrases_df = pd.DataFrame(paraphrases, columns=["text1", "text2"])
paraphrases_df.to_csv(os.path.join("data", "paraphrases.csv"), sep="\t")

# Замечание: т. к. фасттекст просто усредняет векторы слов н-граммы, последовательность слов не должна играть роли
# Для других моделей (например для трансформеров, лучше не сортировать тексты
print(paraphrases[:20])
print("количество позитивных примеров (каждый текст отсортирован по алфавиту):", len(set(paraphrases)))

diff_anws_ids = list(set([tuple(sorted([id1, id2])) for id1 in unique_ids for id2 in unique_ids if id1 != id2]))
print("пример списка разных айди БО", diff_anws_ids[:20])
print("количество пар разных айди БО", len(diff_anws_ids))

shuffle(diff_anws_ids)

no_paraphrases = []
for id1, id2 in diff_anws_ids[:200000]:
    temp_df1 = ids_answrs_df[ids_answrs_df["FastAnswId"] == id1]
    temp_df2 = ids_answrs_df[ids_answrs_df["FastAnswId"] == id2]
    temp_texts1 = list(temp_df1["lem_texts"])[:2]
    temp_texts2 = list(temp_df2["lem_texts"])[:2]
    no_paraphrases += [tuple(sorted(sorted([" ".join(sorted(tx1.split())),
                                            " ".join(sorted(tx2.split()))]))) for tx1 in temp_texts1 for tx2 in
                       temp_texts2
                       if type(tx1) is str and type(tx2) is str]

no_paraphrases = [(tx1, tx2) for tx1, tx2 in no_paraphrases if jaccard_similarity(tx1, tx2) < 0.5]
no_paraphrases_df = pd.DataFrame(no_paraphrases, columns=["text1", "text2"])
no_paraphrases_df.to_csv(os.path.join("data", "no_paraphrases.csv"), sep="\t")

print(no_paraphrases[:10])
print(len(set(no_paraphrases)))

"""
from itertools import groupby
for k, v in groupby(sorted(ids_answrs, key=lambda x: x[0]), key=lambda y: y[0]):
    print(k, list([x[1] for x in v]))"""
