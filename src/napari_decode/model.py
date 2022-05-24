import os, tempfile


class Model:
    """ """

    GATEWAY_URL = "https://raw.githubusercontent.com/TuragaLab/DECODE/master/gateway.yaml"

    def __init__(self):
        """ """

        self.temp_folder = None
        self.write_folder = None
        self.input_parameters_file = None
        self.save_parameters_file = None
        self.parameters = None


