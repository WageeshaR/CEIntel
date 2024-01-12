import uuid

from cassandra.cqlengine import columns
from cassandra.cqlengine.models import Model
from data.database import Database

"""
Profile schema object corresponds to 'profile' Cassandra table 'intel_profile' object of CollabEd core (https://github.com/wageeshar/collabed.git)
"""


class Profile(Model):
    __table_name__ = 'profile'
    __keyspace__ = Database.keyspace()
    __connection__ = Database.conn_name()
    id = columns.UUID(primary_key=True, default=uuid.uuid4, partition_key=True)
    user_id = columns.UUID(default=uuid.uuid4)
    is_studying = columns.Boolean(default=False)
    primary_interest = columns.Text()
    secondary_interest = columns.Text()
    tertiary_interest = columns.List(columns.Text)
