import napari_decode.controller as controller
import numpy as np
import napari
import glob
from typing import List
from magicgui import magic_factory
import magicgui.widgets as mgwidgets
from superqt import QCollapsible
from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QDoubleSpinBox,
    QCheckBox,
    QLineEdit,
    QSpinBox,
)


# TODO: lots of things should be refactored into controller
# TODO: push to repo
class ParametersWidget(QWidget):
    """ """

    # TODO: get scrolling container to work!
    def __init__(self, napari_viewer, parent=None):
        """ """

        # store viewer, initialize self and vertical layout
        self.viewer = napari_viewer
        super().__init__(parent)
        self.setLayout(QVBoxLayout())

        # add parameter input widgets
        self.parameter_inputs, self.widgets_dict = self.create_all_widgets(
            'Parameter Input Fields', controller.ndm.parameters,
        )
        self.layout().addWidget(self.parameter_inputs)

        # add a load parameters combo box
        self.load_parameters_combo = self.load_parameters_combo()
        self.layout().addWidget(self.load_parameters_combo)

        # add a store parameters button
        self.store_button = QPushButton('Store Parameters', self)
        self.store_button.clicked.connect(self.store_parameters)
        self.layout().addWidget(self.store_button)

        # add a load parameters button
        self.load_button = QPushButton('Load Parameters', self)
        self.load_button.clicked.connect(self.load_parameters)
        self.layout().addWidget(self.load_button)


    # TODO: handle List, List of multitype, spinbox ranges, lists of lists
    def create_all_widgets(self, name, d):
        """ """

        # explicit params first, dicts second, empty params removed
        first, second = [], []
        for k, v in d.items():
            if isinstance(v, dict): second.append((k, v))
            elif v is not None: first.append((k, v))
        d = first + second

        # construct collapsible with all parameters
        widgets_dict = {}
        container = QCollapsible(name)
        for k, v in d:
           if isinstance(v, dict):
               widget, widget_dict = self.create_all_widgets(k, v)
           else:
               widget = self.create_widget(k, v)
               if widget is not None:
                   widget_dict = widget.layout().itemAt(1).widget()
           if widget is not None:
               container.addWidget(widget)
               widgets_dict[k] = widget_dict

        # check if container holds anything
        if container._content.layout().count() > 0:
            return container, widgets_dict
        else:
            return None, None


    def create_widget(self, name, value):
        """ """

        row = QWidget()
        row.setLayout(QHBoxLayout())
        row.layout().setContentsMargins(0, 0, 0, 0)
        label = QLabel(name)
        row.layout().addWidget(label)
        try:
            widget = mgwidgets.create_widget(name=name, value=value).native
            if isinstance(value, float):
                decimals = len(str(value).split('.')[1]) + 1
                widget.setDecimals(decimals)
                widget.setValue(value)
            row.layout().addWidget(widget)
            return row
        except Exception as e:
            print(e)
            print(name, ": ", value, "\n")
            return None


    def load_parameters_combo(self):
        """ """

        files = glob.glob(controller.ndm.LOGS_FOLDER + '/' + controller.ndm.PARAMS_PREFIX + '*')
        files = [f.split('/')[-1] for f in files]
        box = QComboBox(self)
        box.addItems(files)
        return box


    def store_parameters(self):
        """ """

        import decode
        from datetime import datetime
        parameters = self.read_parameters(self.widgets_dict)
        suffix = controller.ndm.PARAMS_PREFIX + 'manual-save-' + \
                 datetime.now().strftime('%d%m%Y%H%M%S') + '.yaml'
        path = controller.ndm.LOGS_FOLDER + '/' + suffix
        decode.utils.param_io.save_params(path, parameters)
        self.load_parameters_combo.addItem(suffix)
        count = self.load_parameters_combo.count()
        self.load_parameters_combo.setCurrentIndex(count-1)


    # TODO: problems reading px_size, chweight_stat, bg_uniform, img_size, img_size
    def read_parameters(self, d):
        """ """

        type_map = {
            QLineEdit:       lambda x: x.text(),
            QDoubleSpinBox:  lambda x: x.value(),
            QCheckBox:       lambda x: x.isChecked(),
            QSpinBox:        lambda x: x.value(),
        }

        parameters = {}
        for k, v in d.items():
            if isinstance(v, dict):
                parameters[k] = self.read_parameters(v)
            else:
                try: parameters[k] = type_map[type(v)](v)
                except Exception as e: print(k, '\n', type(v), '\n', e, '\n')
        return parameters


    def load_parameters(self):
        """ """

        file = controller.ndm.LOGS_FOLDER + '/' + self.load_parameters_combo.currentText()
        import decode
        parameters = decode.utils.param_io.load_params(file).to_dict()
        self.set_parameters(self.widgets_dict, parameters)


    def set_parameters(self, widgets, parameters):
        """ """

        type_map = {
            QLineEdit:       lambda x, y: x.setText(y),
            QDoubleSpinBox:  lambda x, y: x.setValue(y),
            QCheckBox:       lambda x, y: x.setChecked(y),
            QSpinBox:        lambda x, y: x.setValue(y),
        }

        for k, v in parameters.items():
            if k in widgets.keys():
                if isinstance(v, dict):
                    self.set_parameters(widgets[k], v)
                else:
                    widget = widgets[k]
                    try: type_map[type(widget)](widget, v)
                    except Exception as e: print(k, '\n', type(widget), '\n', v, '\n', e, '\n')


@magic_factory(
    z_min={"widget_type": "FloatSlider", "min":-1000, "max":1000},
    z_max={"widget_type": "FloatSlider", "min":-1000, "max":1000},
    learning_rate={"widget_type": "FloatSlider", "min":0.0001, "max":0.001},
)
def simulate(
    temporal_context: bool = True,
    emitter_average: int = 25,
    z_min: float = -800,
    z_max: float = 800,
    learning_rate: float = 0.0006,
) -> List[napari.types.LayerDataTuple]:
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


@magic_factory
def train():
    """ """

    controller.setup_trainer()

