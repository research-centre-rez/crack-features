import os
import logging

logger = logging.getLogger(__name__)


def prepare_output_path(output_dir_path):
    """
    Creates directory for the output (if it does not exist). Parent directory must exist.
    :param output_dir_path: The path to the directory where output files will be stored.
    :return: None
    """
    if not os.path.exists(output_dir_path):
        if os.path.exists(os.path.dirname(output_dir_path)):
            os.makedirs(output_dir_path)
        else:
            logger.error("Output directory, nor it's parent directory do not exist.")
            raise Exception("Output directory, nor it's parent directory do not exist.")
    else:
        logger.debug("Output directory prepared.")