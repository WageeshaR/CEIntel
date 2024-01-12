import uuid

from cassandra.cqlengine import columns
from cassandra.cqlengine.models import Model

"""
Post schema object corresponds to 'post' Cassandra table 'intel_post' object of CollabEd core (https://github.com/wageeshar/collabed.git)
"""


class Post(Model):
    def __init__(self):
        super().__init__()
        self.__table_name__ = 'intel_post'
        id = columns.UUID(primary_key=True, default=uuid.uuid4)
        content = columns.Text()
        type = columns.Text(primary_key=True)
