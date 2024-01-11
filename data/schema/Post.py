import uuid

"""
Post schema object corresponds to 'post' Cassandra table 'IntelPost' object of CollabEd core (https://github.com/wageeshar/collabed.git)
"""


class Post:
    def __init__(self, row: object):
        self.id: uuid = row.id
        self.content: str = row.content
        self.type: str = row.type
