"""
This module contains functions related to database connectivity and initialising of session with Cassandra DB.
"""
import logging

from cassandra.cluster import Cluster


class Database:
    """
    self.cluster is the Cassandra cluster representation, later used to connect to using connect() method.
    self.session object holds the connected session and can be used to execute queries against the database.
    """
    def __init__(self, hosts: list, port: int, keyspace: str, auth: dict):
        self.cluster = Cluster(hosts=hosts, port=port)
        self.keyspace = keyspace
        self.session = NotImplemented

    def initdb(self):
        self.session = self.cluster.connect(self.keyspace, wait_for_all_pools=True)

        self.session.execute('USE ' + self.keyspace)

        logging.info(f"Database initialized with session_id: {self.session.session_id}")

    def session(self):
        return self.session
