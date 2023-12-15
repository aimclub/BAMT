import logging
import logging.config
import os


class BamtLogger:
    def __init__(self):
        self.file_handler = False
        self.enabled = True
        self.base = os.path.join(os.path.dirname(os.path.abspath(__file__)))
        self.log_file_path = os.path.join(self.base, "logging.conf")
        logging.config.fileConfig(self.log_file_path)

        self.loggers = dict(
            logger_builder=logging.getLogger("builder"),
            logger_network=logging.getLogger("network"),
            logger_preprocessor=logging.getLogger("preprocessor"),
            logger_nodes=logging.getLogger("nodes"),
            logger_display=logging.getLogger("display"),
        )

    def switch_console_out(self, value: bool):
        """
        Turn off console output from logger.
        By default, all loggers print out messages in console.

        :param value: If False, remove all handlers and pass a NullHandler in order to prevent messages.
        """
        assert isinstance(value, bool)
        if not value:
            for logger in self.loggers.values():
                for handler in logger.handlers:
                    if handler.__class__.__name__ == "StreamHandler":
                        logger.removeHandler(handler)
                        logger.addHandler(logging.NullHandler())
        else:
            for logger in self.loggers.values():
                for handler in logger.handlers:
                    if handler.__class__.__name__ == "NullHandler":
                        logger.removeHandler(handler)
                        logger.addHandler(logging.root.handlers[0])

    def switch_file_out(self, value: bool, log_file: str):
        """
        Send all messages in file
        By default, all loggers print out messages in console.

        :param value: If True, create stdout log file and write messages here as well.
        :param log_file: absolute path to log file, file will be created if not any.

        By default, no log files are created.
        """
        assert isinstance(value, bool)
        if value:
            for logger in self.loggers.values():
                file_handler = logging.FileHandler(log_file, mode="a")
                file_handler.setFormatter(logging.root.handlers[0].formatter)
                logger.addHandler(file_handler)
        else:
            for logger in self.loggers.values():
                for handler in logger.handlers:
                    if handler.__class__.__name__ == "FileHandler":
                        logger.removeHandler(handler)


bamt_logger = BamtLogger()

(
    logger_builder,
    logger_network,
    logger_preprocessor,
    logger_nodes,
    logger_display,
) = list(bamt_logger.loggers.values())
