import os
import logging

logger = logging.getLogger(__name__)

def add_argparse_argument(parser):
    parser.add_argument(
        "-o",
        "--output-dir-path",
        type=str,
        help="Path to output directory. If directory does not exist, it will be created (but parent directory must exist)."
    )


def prepare_output_path(output_dir_path, sample_name=None):
    """
    Creates directory for the output (if it does not exist). Parent directory must exist.
    When sample name is specified the subdirectory with this name is created as well.

    :param output_dir_path: The path to the directory where output files will be stored.
    :return: output directory path and output files prefix (which is equal to sample name if specified)
    """
    if not os.path.exists(output_dir_path):
        if os.path.exists(os.path.dirname(output_dir_path)):
            os.makedirs(output_dir_path)
        else:
            logger.error("Output directory, nor it's parent directory do not exist.")
            raise Exception("Output directory, nor it's parent directory do not exist.")
    else:
        logger.debug("Output directory prepared.")

    if sample_name is not None:
        output_dir_path = os.path.join(output_dir_path, sample_name)
        os.makedirs(output_dir_path, exist_ok=True)
        output_file_prefix = f"{sample_name}-"
    else:
        output_dir_path = output_dir_path
        output_file_prefix = ""

    return output_dir_path, output_file_prefix
