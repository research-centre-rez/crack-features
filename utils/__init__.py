import logging
import utils.output_dir_generator
import utils.image_logger
import utils.cli_arguments
import utils.config_loader


def configure_logger(filename, level=logging.INFO):
    logging.basicConfig(
        level=logging.INFO,
        filename=filename,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )