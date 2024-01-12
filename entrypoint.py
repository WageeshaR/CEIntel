import logging
import os

from cassandra.cluster import Session
from dotenv import load_dotenv
from api.base import base
from data.database import Database
from cassandra import DriverException


def bootstrap():
    load_dotenv()

    database = Database(
        hosts=os.getenv('CASS_HOSTS').split(","),
        port=int(os.getenv('CASS_PORT')),
        keyspace=os.getenv('CASS_KEYSPACE'),
        auth={'username': os.getenv('CASS_USERNAME'), 'password': os.getenv('CASS_PASSWORD')}
    )

    try:
        database.initdb()
    except DriverException as e:
        logging.error(f"Cassandra driver error initializing database: {e}")
        exit(1)
    except Exception as e:
        logging.error(f"Unexpected error while initializing database: {e}")
        exit(1)

    try:
        base.run(host='0.0.0.0', port=os.getenv('FLASK_PORT'))
    except Exception as e:
        logging.error(f"Unexpected error while starting flask server: {e}")
        exit(1)

    logging.info(f"CEIntel application bootstrapped successfully!")


if __name__ == "__main__":
    bootstrap()
