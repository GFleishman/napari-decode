from pathlib import Path


class model:
    """ """

    LOGS_FOLDER = str(Path.home()) + '/.napari-decode'
    PARAMS_PREFIX = 'napari-decode-parameters-'
    GATEWAY_URL = "https://raw.githubusercontent.com/TuragaLab/DECODE/master/gateway.yaml"

    def __init__(self):
        """ """

        self.temp_folder = None
        self.write_folder = None
        self.input_parameters_file = None
        self.save_parameters_file = None
        self.parameters = None

        self.sim_train = None
        self.sim_test = None


