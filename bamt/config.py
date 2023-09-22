import configparser
from os import path, stat

config = configparser.ConfigParser()

CONFIGFILE = path.join(path.dirname(path.abspath(__file__)), "selbst.ini")

if path.isfile(CONFIGFILE) and stat(CONFIGFILE).st_size != 0:
    config.read(CONFIGFILE)
else:
    open(CONFIGFILE, "a").close()
    config["LOG"] = {
        "log_conf_loc": path.join(path.dirname(path.abspath(__file__)), "logging.conf")
    }
    with open(CONFIGFILE, "w") as configfile:
        config.write(configfile)
