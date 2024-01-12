import uuid

from cassandra.cqlengine import columns
from cassandra.cqlengine.models import Model
from data.database import Database

"""
Post schema object corresponds to 'post' Cassandra table 'intel_post' object of CollabEd core (https://github.com/wageeshar/collabed.git)
"""


class Post(Model):
    __table_name__ = 'post'
    __keyspace__ = Database.keyspace()
    __connection__ = Database.conn_name()
    id = columns.UUID(primary_key=True, default=uuid.uuid4)
    content = columns.Text()
    type = columns.Text(primary_key=True)
