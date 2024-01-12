import uuid

from cassandra.cqlengine import columns
from cassandra.cqlengine.models import Model

"""
Profile schema object corresponds to 'profile' Cassandra table 'intel_profile' object of CollabEd core (https://github.com/wageeshar/collabed.git)
"""


class Profile(Model):
    def __init__(self):
        super().__init__()
        __table_name__ = 'intel_profile'
        id = columns.UUID(primary_key=True, default=uuid.uuid4)
        user_id = columns.UUID(default=uuid.uuid4)
        is_studying = columns.Boolean(default=False)
        primary_interest = columns.Text()
        secondary_interest = columns.Text()
        tertiary_interest = columns.List(columns.Text)
