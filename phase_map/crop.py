import argparse
import imageio.v3 as iio
import os

import output_generator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""
    According to parameters crops the input phase image.
    """)

    parser.add_argument("image_path", type=str, help="Path to input image.")
    parser.add_argument("-o", "--output_path", type=str,
                        help="Path to output image. If not specified suffix -cropped is added to the input.",
                        required=False, default=None)
    parser.add_argument("-W", "--width", type=int, help="Width of the cropped phase image.")
    parser.add_argument("-H", "--height", type=int, help="Height of the cropped phase image.")
    parser.add_argument("-D", "--dimensions", type=int, help="Dimensions of the cropped phase image.")

    args = parser.parse_args()

    input_image = iio.imread(args.image_path)
    if args.output_path is None:
        output_path = ".".join(args.image_path.split(".")[:-1]) + "_cropped.png"
    else:
        output_generator.prepare_output_path(output_dir_path=os.path.dirname(args.output_path))
        output_path = args.output_path

    if len(input_image.shape) == 3:
        iio.imwrite(output_path, input_image[:args.height, :args.width, :args.dimensions])
    else:
        iio.imwrite(output_path, input_image[:args.height, :args.width])
