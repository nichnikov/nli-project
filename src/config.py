import os
import json
import logging
import pandas as pd
from pathlib import Path
from pydantic import BaseModel
from pydantic import BaseSettings
from collections import namedtuple
from src.texts_storage import TextsStorage


def get_project_root() -> Path:
    """"""
    return Path(__file__).parent.parent


# ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_DIR = get_project_root()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', )

logger = logging.getLogger()
logger.setLevel(logging.INFO)

texts_storage_db = TextsStorage(os.path.join(PROJECT_ROOT_DIR, "data", "queries.db"))


ROW = namedtuple("ROW", "SysID, ID, Cluster, ParentModuleID, ParentID, ParentPubList, "
                        "ChildBlockModuleID, ChildBlockID, ModuleID, Topic, Subtopic, DocName, ShortAnswerText")


class Etalon(BaseModel):
    """Схема данных для эталона (без ЭталонИД)"""
    TemplateId: int
    Text: str
    LemmText: str
    SysId: int
    ModuleId: int
    Pubs: list[int]


class TextsDeleteSample(BaseModel):
    """Схема данных для удаления данных по тексту из Индекса"""
    Index: str
    Texts: list[str]
    FieldName: str
    Score: float


class Settings(BaseSettings):
    """Base settings object to inherit from."""

    class Config:
        # print(os.path.join(ROOT_DIR, ".env"))
        env_file = os.path.join(PROJECT_ROOT_DIR, ".env")
        env_file_encoding = "utf-8"


class ElasticSettings(Settings):
    """Elasticsearch settings."""

    hosts: str
    index: str
    user_name: str | None
    password: str | None

    max_hits: int = 100
    chunk_size: int = 500

    @property
    def basic_auth(self) -> tuple[str, str] | None:
        """Returns basic auth tuple if user and password are specified."""
        print(self.user_name, self.password)
        if self.user_name and self.password:
            return self.user_name, self.password
        return None
