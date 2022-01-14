import logging.config
from os import path
log_file_path = path.join(path.dirname(path.abspath(__file__)), 'logging.conf')
logging.config.fileConfig(log_file_path)


logger_builder = logging.getLogger('builder')
logger_network = logging.getLogger('network')
logger_preprocessor = logging.getLogger('preprocessor')
logger_metrics = logging.getLogger('metrics')

logging.captureWarnings(True)
logger_warnings = logging.getLogger('py.warnings')
