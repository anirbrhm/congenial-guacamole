import logging

def init_console_logger(logger):
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(levelname)s] %(message)s')
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler("model-logs.log")
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)