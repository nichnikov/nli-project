"""https://elasticsearch-py.readthedocs.io/en/latest/async.html"""
import asyncio

import pymssql
from elasticsearch import AsyncElasticsearch
from elasticsearch.helpers import async_bulk
from src.config import (ElasticSettings,
                        TextsDeleteSample,
                        logger, ROW)
from src.utils import jaccard_similarity


class DataFromDB:
    """
    class for getting data from MS Server with specific SQL Query.
    """

    def __init__(self, **kwargs):
        conn = pymssql.connect(host=kwargs["server_host"],
                               port=1433,
                               user=kwargs["user_name"],
                               password=kwargs["password"],
                               database="master")
        self.cursor = conn.cursor(as_dict=True)

    def fetch_from_db(self, sys_id: int, date: str):
        """
        return data (rows) from MS SQL DB Statistic on date (usually today)
        """
        self.cursor.execute("SELECT * FROM StatisticsRAW.[search].FastAnswer_RBD  "
                            "WHERE SysID = {} AND (ParentBegDate IS NOT NULL AND ParentEndDate IS NULL "
                            "OR ParentEndDate > {})".format(sys_id, "'" + str(date) + "'"))
        logger.info("Total rows for SysID {} is {}".format(sys_id, self.cursor.rowcount))
        return self.cursor.fetchall()

    def get_rows(self, sys_id: int, date: str) -> []:
        """
        Parsing rows from DB and returning list of unique tuples with etalons and list of tuples with data for answers
        """
        rows = []
        data_from_db = self.fetch_from_db(sys_id, date)
        for row in data_from_db:
            try:
                parent_pub_list = [int(pb) for pb in row["ParentPubList"].split(",") if pb != '']
                rows.append(ROW(row["SysID"], row["ID"], row["Cluster"], row["ParentModuleID"],
                                row["ParentID"], parent_pub_list, row["ChildBlockModuleID"],
                                row["ChildBlockID"], row["ModuleID"], row["Topic"], row["Subtopic"],
                                row["DocName"], row["ShortAnswerText"]))
            except ValueError as err:
                logger.exception("Parsing {} with row: {}".format(err, str(row)))
        logger.info("Unique etalons tuples rows quantity is {} for SysID {}".format(len(rows), str(sys_id)))
        return rows

