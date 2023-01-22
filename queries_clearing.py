import os
import pandas as pd
import hashlib
from src.texts_processing import TextsTokenizer

m = hashlib.sha256()
tokenizer = TextsTokenizer()
q_df = pd.read_csv(os.path.join("data", "queries.csv"), sep="\t")


lem_texts = [" ".join(sorted(lm_tx)) for lm_tx in tokenizer(list(q_df["Cluster"]))]
lem_texts_df = pd.DataFrame(lem_texts, columns=["lem_texts"])
lem_texts_df_ = pd.concat([q_df, lem_texts_df], axis=1)
q_dict = lem_texts_df_.to_dict(orient="records")
res_dict = {}
for d in q_dict:
    my_str = d["lem_texts"]
    my_hash = hashlib.sha256(my_str.encode('utf-8')).hexdigest()
    res_dict[my_hash] = d

res_df = pd.DataFrame([res_dict[i] for i in res_dict])
res_df.to_csv(os.path.join("data", "unique_queries.csv"), sep="\t", index=False)

# df2 = pd.unique(q_df[['Courses', 'Fee']].values.ravel("lem_texts"))
# df2 = pd.unique(q_df[["FastAnswId", "Cluster", "lem_texts"]].values.ravel("A"))
# print(df2[:100])
# print(len(df2))
# 'C', 'F', 'A', 'K'
'''
lem_texts_df_ = lem_texts_df.drop_duplicates()
print(lem_texts_df_)

lem_texts_df_ = pd.merge(lem_texts_df_, lem_texts_df, on="lem_texts", how="left")
print(lem_texts_df_)
'''