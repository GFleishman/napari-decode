import napari_decode.model as model
import napari_decode.controller as controller


__version__ = "0.0.1"
model = model.model()
controller.initialize_parameters(model)

