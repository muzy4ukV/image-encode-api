from .base import BaseDB
from .file_db import FileDB
from .mongo_db import MongoDB

class DBFactory:
    def __init__(self, db_type) -> None:
        if db_type == 'file':
            self.db = FileDB()
        elif db_type == 'mongo':
            self.db = MongoDB()
        else:
            raise Exception("Invalid database type")

    def get_db(self) -> BaseDB:
        return self.db