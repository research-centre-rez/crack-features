import json


def add_argparse_argument_cracks_config_path(parser):
    parser.add_argument(
        "-c",
        "--cracks-config-path",
        type=str,
        help=f"Path to the cracks_config.json. In this file are defined crack image processing parameters."
             f"(@see config/cracks_config.json).",
        default="config/cracks_config.json"
    )


def add_argparse_argument_phase_config_path(parser):
    parser.add_argument(
        "-c",
        "--phase-config-path",
        type=str,
        help=f"Path to the phase_config.json. In this file are defined colors of phases used in phase map "
             f"(@see config/phases_config.json).",
        default="config/phases_config.json"
    )


class Config:
    cracks_config = None
    phases_config = None

    @staticmethod
    def load_config(config_file_path, type):
        with open(config_file_path, 'r') as file:
            if type == 'cracks':
                Config.cracks_config = json.load(file)
                return Config.cracks_config
            elif type == 'phases':
                Config.phases_config = json.load(file)
                return Config.phases_config
            else:
                raise Exception(f"Unknown config type: {type}")

    @staticmethod
    def get_config(type):
        if type == 'cracks':
            config = Config.cracks_config
        elif type == 'phases':
            config = Config.phases_config
        else:
            raise Exception(f"Unknown config type: {type}")

        if config is None:
            raise ValueError("Config not loaded, call `load_config` first.")
        else:
            return config
