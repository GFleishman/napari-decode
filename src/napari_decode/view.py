import napari_decode.controller as controller
import numpy as np
import napari
from typing import List


def parameters(
    write_folder: str = '',
    parameters_file: str = '',
):
    """ """

    write_folder = write_folder or controller.get_write_folder()
    parameters_file = parameters_file or controller.get_save_parameters_file()
    controller.set_temp_folder()
    controller.set_write_folder(write_folder)
    controller.set_parameters(parameters_file)



def simulate() -> List[napari.types.LayerDataTuple]:
    """ """

    frames, spots = controller.simulate()
    return [
        (frames,
        {'name': 'simulated_frames'},),
        (spots,
        {'name': 'simulated_emitters',
         'face_color':'white',
         'edge_color':'white',
         'size':1},
        'points',),
    ]


