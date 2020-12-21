# -*- coding: utf-8 -*-
import logging
import os
import datetime
from collections import defaultdict
import time
import numpy as np


class SingletonType(type):
    """
    Singleton class to create a singleton logger
    """

    _instances = {}

    def __call__(cls, *args):
        cls_id = str(cls)
        if len(args) > 0:
            cls_id += str(args[0])
        if cls_id not in cls._instances:
            cls._instances[cls_id] = super(SingletonType, cls).__call__(*args)
        return cls._instances[cls_id]


class Logger(object, metaclass=SingletonType):
    def __init__(self, storage_parent_folder, debug=True):
        self._logger = logging.getLogger("pgdl")
        self.debug_param = debug
        if debug:
            self._logger.setLevel(logging.DEBUG)
        else:
            self._logger.setLevel(logging.WARNING)
        self.storage_parent_folder = storage_parent_folder
        formatter = logging.Formatter(
            "%(asctime)s \t [%(levelname)s | %(filename)s:%(lineno)s] > %(message)s"
        )

        now = datetime.datetime.now()

        if not os.path.isdir(self.storage_parent_folder):
            os.makedirs(self.storage_parent_folder)
        fileHandler = logging.FileHandler(
            os.path.join(
                self.storage_parent_folder, "log_" + now.strftime("%Y-%m-%d") + ".log"
            )
        )

        streamHandler = logging.StreamHandler()

        fileHandler.setFormatter(formatter)
        streamHandler.setFormatter(formatter)
        if not (self._logger.hasHandlers()):
            self._logger.addHandler(fileHandler)
            self._logger.addHandler(streamHandler)

    def get_logger(self):
        return self

    def info(self, message):
        self._logger.info(message)

    def debug(self, message):
        self._logger.debug(message)

    def warning(self, message):
        self._logger.warning(message)

    def error(self, message):
        self._logger.error(message)

    def exception(self, message):
        self._logger.exception("[Exception]" + message)

    def critical(self, message):
        self._logger.critical(message)


# a simple usecase
if __name__ == "__main__":
    logger = Logger.__call__("test").get_logger()
    logger.info("Hello, Logger")
    logger.debug("bug occured")
