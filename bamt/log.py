import logging.config
import os
import warnings

from bamt.config import config


class BamtLogger:
    def __init__(self):
        self.file_handler = False
        self.enabled = True
        self.log_file_path = config.get(
            "LOG", "log_conf_loc", fallback="log_conf_path is not defined"
        )
        try:
            logging.config.fileConfig(self.log_file_path)
        except BaseException:
            log_file_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "logging.conf"
            )
            logging.config.fileConfig(log_file_path)
            warnings.warn(
                "Reading log path location from config file failed. Default location will be used instead."
            )

        self.loggers = dict(
            logger_builder=logging.getLogger("builder"),
            logger_network=logging.getLogger("network"),
            logger_preprocessor=logging.getLogger("preprocessor"),
            logger_nodes=logging.getLogger("nodes"),
            logger_display=logging.getLogger("display"),
        )

    def switch_console_out(self, value: bool):
        assert isinstance(value, bool)
        if not value:
            for logger in self.loggers.values():
                for handler in logger.handlers:
                    if handler.__class__.__name__ == "StreamHandler":
                        logger.removeHandler(handler)
                        logger.addHandler(logging.NullHandler())
        else:
            # will be done by release date
            pass

    def switch_file_out(self, value: bool):
        assert isinstance(value, bool)
        if not value:
            pass
        else:
            # will be done by release date. NOT FINISHED!
            pass


bamt_logger = BamtLogger()

bamt_logger.switch_console_out(False)

(
    logger_builder,
    logger_network,
    logger_preprocessor,
    logger_nodes,
    logger_display,
) = list(bamt_logger.loggers.values())
