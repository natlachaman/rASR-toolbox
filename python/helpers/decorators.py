import functools
import logging


def catch_exception(func):
    """
    A decorator that wraps the passed in function and logs
    exceptions should one occur
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except:
            # log the exception
            err = "There was an exception in  "
            err += func.__name__
            logging.exception(err)
            # re-raise the exception
            raise
    return wrapper
