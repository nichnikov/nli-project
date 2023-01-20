import sqlite3


class TextsStorage:
    def __init__(self, db_path: str):
        self.con = sqlite3.connect(db_path, check_same_thread=False)
        self.con.execute("VACUUM")
        self.cur = self.con.cursor()

    def add(self, data: [], table_name: str):
        """"""
        if data:
            self.cur.executemany("insert into {} values({})".format(table_name, ','.join(['?'] * len(data[0]))), data)
            self.con.commit()

    def delete(self, ids: [str], column_name: str, table_name: str):
        """dictionary must include all attributes"""
        sql = "delete from {} where {} in ({})".format(table_name, column_name, ','.join(['?'] * len(ids)))
        self.cur.execute(sql, ids)
        self.con.commit()

    def delete_all_from_table(self, table_name: str):
        """"""
        self.cur.execute("delete from {}".format(table_name))
        self.con.commit()

    def search_return_one_col(self, ids: [], returned_column_name: str, column_name: str, table_name: str):
        """Возвращает текст вопроса с метаданными по входящему списку answer ids"""
        sql = "select {} from {} where {} in ({})".format(returned_column_name, table_name,
                                                          column_name, ','.join(['?'] * len(ids)))
        self.cur.execute(sql, ids)
        return self.cur.fetchall()

    def search_return_all_col(self, ids: [], returned_column_name: str, column_name: str, table_name: str):
        """Возвращает текст вопроса с метаданными по входящему списку answer ids"""
        sql = "select * from {} where {} in ({})".format(table_name,
                                                         column_name, ','.join(['?'] * len(ids)))
        self.cur.execute(sql, ids)
        return self.cur.fetchall()

    def get_data_from_column(self, table_name: str, column_name: str):
        """"""
        sql = "select {} from {} ".format(column_name, table_name)
        self.cur.execute(sql)
        return self.cur.fetchall()

    def get_data_from_table(self, table_name: str):
        """"""
        sql = "select * from {} ".format(table_name)
        self.cur.execute(sql)
        return self.cur.fetchall()

    def get_count(self, table_name: str):
        """"""
        sql = "select count(*) from {}".format(table_name)
        self.cur.execute(sql)
        return self.cur.fetchall()
