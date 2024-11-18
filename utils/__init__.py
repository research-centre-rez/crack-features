import logging


def configure_logger(filename, level=logging.INFO):
    logging.basicConfig(
        level=logging.INFO,
        filename=filename,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )