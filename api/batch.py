import logging
import os

from apiflask import APIBlueprint
from flask import request
import global_const

batch = APIBlueprint('batch', __name__)


@batch.route('/update', methods=['POST'])
def update():
    """
    Updates training engine with a new batch of data.
    This request must be sent after the data batch has been persisted into the cassandra database
    :return:
    """
    data = request.get_json()
    req_keys = list(data.keys())

    for key in req_keys:
        if key not in ['entity', 'size', 'train_args']:
            # entity = schema entity, size = batch size, train_args = arguments necessary for training pipeline
            return f"request body is missing key {key}", 400
        if key is None or key == '':
            return f"{key} must not be empty", 400

    if int(data['size']) > int(os.getenv(global_const.IN_MEMORY_PROC_HARD_LIM)):
        logging.info("Dispatching Spark processing queue...")
        # TODO: spark queue implementation
    elif int(data['size']) < int(os.getenv(global_const.IN_MEMORY_PROC_SOFT_LIM)) and data['entity'] in ['post']:
        logging.info("Dispatching Spark processing queue...")
        # TODO: spark queue implementation
    else:
        logging.info("Dispatching in-memory processing queue...")
        # TODO: in-memory processing implementation
