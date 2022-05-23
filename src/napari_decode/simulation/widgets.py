import napari
import numpy as np


def simulate(
    size: int = 64,
) -> napari.types.LayerDataTuple:
    """ """

    data = np.arange(size * size, dtype=np.uint16).reshape((size, size))
    return (
        data,
        {"name": "data"},
    )
