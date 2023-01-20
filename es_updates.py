import os
import pandas as pd
from datetime import datetime
import json
from src.utils import texts_tokenize
from src.storage import DataFromDB
from src.utils import timeit
from collections import namedtuple
from src.config import PROJECT_ROOT_DIR


def data_lematization(data_dicts: [{}], **kwargs):
    """
    лемматизация исходных данных
    """
    sws_roots = []
    for fail_name in kwargs["stopwords_files"]:
        sws_roots.append(os.path.join(PROJECT_ROOT_DIR, "data", fail_name))

    clusters = [str(x["Cluster"]) for x in data_dicts]
    lem_clusters = texts_tokenize(clusters, sws_roots)
    for d, l_c in zip(data_dicts, lem_clusters):
        d["LemCluster"] = l_c

    if kwargs["LemDocName"]:
        doc_names = [str(x["DocName"]) for x in data_dicts]
        lem_doc_names = texts_tokenize(doc_names, sws_roots)
        for d, l_dn in zip(data_dicts, lem_doc_names):
            d["LemDocName"] = l_dn

    if kwargs["LemShortAnswerText"]:
        short_answers = [str(x["ShortAnswerText"]) for x in data_dicts]
        lem_short_answers = texts_tokenize(short_answers, sws_roots)
        for d, l_sa in zip(data_dicts, lem_clusters, lem_short_answers):
            d["LemShortAnswerText"] = l_sa


@timeit
def msdb2es(es: ElasticClient, db_con, **kwargs):
    """Перезапись данных из MS статистики в индекс "clusters" Эластика"""
    ROW_FOR_ANSWERS = namedtuple("ROW_FOR_ANSWERS",
                                 "SysID, ID, ParentModuleID, ParentID, ChildBlockModuleID, ChildBlockID")

    def data_for_answer_create(pubs_urls: [], row_tuples: [ROW_FOR_ANSWERS]):
        pubs_answers = []
        for pub, sys_url in pubs_urls:
            for row_tuple in row_tuples:
                if row_tuple.ParentModuleID == 16 and row_tuple.ChildBlockModuleID in [86, 12]:
                    module_id = row_tuple.ChildBlockModuleID
                    document_id = row_tuple.ChildBlockID
                else:
                    module_id = row_tuple.ParentModuleID
                    document_id = row_tuple.ParentID

                query_url = "/".join([sys_url, str(module_id), str(document_id), "actual/"])
                answer_text = "Вот ссылка по вашему вопросу "
                pubs_answers.append({"pubId": int(pub), "templateId": int(row_tuple.ID),
                                     "templateText": answer_text + str(query_url)})
        return pubs_answers

    es.delete_index(kwargs["clusters_index_name"])
    es.create_index(kwargs["clusters_index_name"])
    es.delete_index(kwargs["answers_index_name"])
    es.create_index(kwargs["answers_index_name"])
    today = datetime.today().strftime('%Y-%m-%d')
    for sys_id in kwargs["sys_pub_url"]:
        rows = db_con.get_rows(int(sys_id), today)
        data_dicts = [nt._asdict() for nt in rows]
        data_lematization(data_dicts, **kwargs)
        es.add_docs(kwargs["clusters_index_name"], data_dicts)
        rows_answers = [ROW_FOR_ANSWERS(r.SysID, r.ID, r.ParentModuleID, r.ParentID,
                                        r.ChildBlockModuleID, r.ChildBlockID) for r in rows]
        rows_answers_unique = list(set(rows_answers))
        answers_dicts = data_for_answer_create(kwargs["sys_pub_url"][sys_id], rows_answers_unique)
        es.add_docs(kwargs["answers_index_name"], answers_dicts)


@timeit
def scv2es(es: ElasticClient, **kwargs):
    """обновление данных в индексе "clusters" из csv файлов"""
    csv_parameters = kwargs["csv_parameters"]
    for sys_id in csv_parameters:
        appendix = 100000000 * int(sys_id)
        file_name = csv_parameters[sys_id]["file_name"]
        pubs = csv_parameters[sys_id]["pubs"]
        df = pd.read_csv(os.path.join(PROJECT_ROOT_DIR, "data", file_name), sep="\t")
        data_dics = df.to_dict(orient="records")

        clusters_for_es = [{"SysID": int(sys_id),
                            "ID": int(appendix) + int(d["templateId"]),
                            "Cluster": d["text"],
                            "ParentModuleID": 0,
                            "ParentID": 0,
                            "ParentPubList": pubs,
                            "ChildBlockModuleID": 0,
                            "ChildBlockID": 0,
                            "ModuleID": 85,
                            "Topic": "нет",
                            "Subtopic": "нет",
                            "DocName": "нет",
                            "ShortAnswerText": d["templateText"]} for d in data_dics]

        answers_for_es = [{"pubId": pubid,
                           "templateId": d["ID"],
                           "templateText": d["ShortAnswerText"]} for d in clusters_for_es for pubid in pubs]

        data_lematization(clusters_for_es, **kwargs)

        # удаление вопросов и ответов с имеющимся айди:
        del_answ_ids = [d["ID"] for d in clusters_for_es]
        es.delete_in_field(kwargs["clusters_index_name"], "ID", del_answ_ids)
        es.delete_in_field(kwargs["answers_index_name"], "templateId", del_answ_ids)

        # добавление вопросов и ответов:
        es.add_docs(kwargs["clusters_index_name"], clusters_for_es)
        es.add_docs(kwargs["answers_index_name"], answers_for_es)


def answers_delete(es: ElasticClient, template_ids: []):
    """
    """
    es.delete_in_field("clusters", "ID", template_ids)
    es.delete_in_field("answers", "templateId", template_ids)


if __name__ == "__main__":
    with open(os.path.join(PROJECT_ROOT_DIR, "data", "statistics_parameters.json")) as st_f:
        stat_prmtrs = json.load(st_f)

    es = ElasticClient()
    """1) добавление эталонов и ответов"""
    db_credentials = stat_prmtrs["db_credentials"]
    db_con = DataFromDB(**db_credentials)
    msdb2es(es, db_con, **stat_prmtrs)

    """2) добавление из csv файлов"""
    with open(os.path.join(PROJECT_ROOT_DIR, "data", "csv_parameters.json")) as st_f:
        csv_prmtrs = json.load(st_f)

    stat_prmtrs["csv_parameters"] = csv_prmtrs
    scv2es(es, **stat_prmtrs)

    """3) удаление эталонов и ответов по списку"""
    df = pd.read_csv(os.path.join(PROJECT_ROOT_DIR, "data", "del_answers.csv"), sep="\t")
    template_ids = list(df["TemplateId"])
    answers_delete(es, template_ids)

    # DELETE:
    '''
    @timeit
    def sclite2es(es: ElasticClient, texts_storage_db):
        """Обновление данных в индексе "answers" Эластика из sclite"""
        # es = ElasticClient()
        es.delete_index("answers")
        es.create_index("answers")

        links_data = texts_storage_db.get_data_from_table("answers")

        answers_df = pd.DataFrame(links_data, columns=["templateId", "templateText", "pubId"])
        answers_dicts = answers_df.to_dict(orient="records")

        es.add_docs("answers", answers_dicts)
        
    """2) добавление ответов"""
    sclite2es(es, texts_storage_db)

    '''
