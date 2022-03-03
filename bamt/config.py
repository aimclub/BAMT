import configparser
from os import path

config = configparser.ConfigParser()

CONFIGFILE = path.join(path.dirname(path.abspath(__file__)), 'selbst.ini')
if path.isfile(CONFIGFILE):
    config.read(CONFIGFILE)
else:
    open(CONFIGFILE, 'a').close()
    config['NODES'] = {'models_storage' : path.join(path.dirname(path.abspath(__file__)), 'Nodes_data')}
    config['LOG'] = {"log_conf_loc" : path.join(path.dirname(path.abspath(__file__)), 'logging.conf')}
    with open(CONFIGFILE, 'w') as configfile:
        config.write(configfile)
