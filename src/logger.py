import logging


def get_logger(name):
    return logging.getLogger(name)


class Loggable(object):
    _logger = None

    @property
    def logger(self):
        if self._logger is None:
            name = self.__module__ + '.' + self.__class__.__name__
            self._logger = logging.getLogger(name)
        return self._logger
