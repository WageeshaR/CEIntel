from apiflask import APIFlask
import psutil

base = APIFlask('CE_INTEL', spec_path='/api.json')
base.config['OPENAPI_VERSION'] = '3.0.2'


@base.route('/health', methods=['GET'])
def health():
    """
    Evaluates the API health status and responds with a status code. Can be used by clients before sending data loads for processing
    :return: one of HEALTH_OK, HEALTH_FAIL, ON_ALERT
    """
    cpu = psutil.cpu_percent(interval=1)
    mem = psutil.virtual_memory().percent

    if cpu > 80 or mem > 80:
        return {"status": "HEALTH_FAIL", "healthy": False}

    if cpu > 60 or mem > 60:
        return {"status": "ON_ALERT", "healthy": True}

    return {"status": "HEALTH_OK", "healthy": True}

