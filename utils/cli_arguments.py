def add_sample_name(parser):
    parser.add_argument(
        "--sample-name",
        type=str,
        help="The name of the sample will be used as prefix for all produced outputs."
    )