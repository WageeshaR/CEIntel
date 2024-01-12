"""
This module contains functions related to database connectivity and initialising of session with Cassandra DB.
"""
import logging
import os

from cassandra.cluster import Cluster
from cassandra.cqlengine.connection import register_connection, set_default_connection


class Database:
    """
    self.cluster is the Cassandra cluster representation, later used to connect to using connect() method.
    self.session object holds the connected session and can be used to execute queries against the database.
    """
    def __init__(self, hosts: list, port: int, keyspace: str, auth: dict):
        self.cluster = Cluster(contact_points=hosts, port=port)
        self.keyspace = keyspace
        self.session = NotImplemented

    def initdb(self):
        self.session = self.cluster.connect(self.keyspace, wait_for_all_pools=True)

        self.session.execute('USE ' + self.keyspace)
        register_connection(self.conn_name(), session=self.session)
        set_default_connection(self.conn_name())

        logging.info(f"Database connection initialized with session_id: {self.session.session_id}")

    def session(self):
        return self.session

    @staticmethod
    def keyspace():
        """
        Returns the default cassandra keyspace name
        :return: string
        """
        return os.getenv('CASS_KEYSPACE')

    @staticmethod
    def conn_name():
        """
        Returns the default cassandra connection name
        :return: string
        """
        return os.getenv('DEFAULT_CASS_CONN')
