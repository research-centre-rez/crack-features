import argparse
import imageio.v3 as iio
import os
from utils import output_dir_generator


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""
    According to parameters crops the input phase image.
    """)

    parser.add_argument("image_path", type=str, help="Path to input image.")
    output_dir_generator.add_argparse_argument(parser)
    parser.add_argument("-W", "--width", type=int, help="Width of the cropped phase image.")
    parser.add_argument("-H", "--height", type=int, help="Height of the cropped phase image.")
    parser.add_argument("-D", "--dimensions", type=int, help="Dimensions of the cropped phase image.")

    args = parser.parse_args()

    input_image = iio.imread(args.image_path)
    output_dir_path, _ = output_dir_generator.prepare_output_path(output_dir_path=args.output_dir_path)
    output_file_path = os.path.join(output_dir_path, os.path.basename(args.image_path))

    if len(input_image.shape) == 3:
        iio.imwrite(output_file_path, input_image[:args.height, :args.width, :args.dimensions])
    else:
        iio.imwrite(output_file_path, input_image[:args.height, :args.width])
