import napari_decode.model as model
import os, tempfile, requests, yaml
from pathlib import Path
import decode
import numpy as np



def set_temp_folder():
    """ """

    if not model.temp_folder:
        model.temp_folder = tempfile.TemporaryDirectory() 

def get_temp_folder():
    """ """

    try:
        return model.temp_folder.name
    except:
        return None


def set_write_folder(write_folder):
    """ """

    write_folder = write_folder or os.getcwd() + '/napari-decode-outputs'
    os.makedirs(write_folder, exist_ok=True)
    model.write_folder = write_folder

def get_write_folder():
    """ """

    return model.write_folder


def set_parameters(parameters_file):
    """ """

    # if no parameter file given, download default parameters
    if not parameters_file:
        gateway_file = get_temp_folder() + '/gateway.yaml'
        with open(gateway_file, 'wb') as f:
            f.write(requests.get(model.GATEWAY_URL).content)
        with open(gateway_file, 'r') as f:
            gateway = yaml.safe_load(f)
        package = gateway['examples']['colab_train_experimental_rc']
        sample_folder = decode.utils.example_helper.load_example_package(
            path=Path(get_temp_folder() + '/' + package['name'] + '.zip'),
            url=package['url'],
            hash=package['hash'],
        )
        parameters_file = get_temp_folder() + '/colab_train_experimental/param_run_in.yaml'

    # TODO: check validity of parameters file
    model.input_parameters_file = parameters_file
    param = decode.utils.param_io.load_params(parameters_file)
    param.Meta.version = decode.utils.bookkeeping.decode_state()

    # XXX: temp params from colab, modified for my macbook
    param.Hardware.num_worker_train = 2
    param.Hardware.torch_threads = 2
    param = decode.utils.param_io.autoset_scaling(param)
    calibration_file = get_temp_folder() + '/colab_train_experimental/spline_calibration_3dcal.mat'
    param.InOut.calibration_file = calibration_file
    model_dir = Path(get_write_folder() + '/network')
    model_dir.mkdir()
    print(f"Created directory, absolute path: {model_dir.resolve()}")
    model_out = model_dir / 'model.pt'
    ckpt_path = model_dir /'decode_ckpt.pt'
    param.InOut.experiment_out = str(model_dir)
    temporal_context = True
    if temporal_context:
      param.HyperParameter.channels_in = 3
    else:
      param.HyperParameter.channels_in = 1
    emitter_average = 25
    param.Simulation.emitter_av = emitter_average
    z_min = -800
    z_max = 800
    param.Simulation.emitter_extent[2] = [z_min, z_max]
    lr = 0.0006
    param.HyperParameter.opt_param.lr = lr
    param.Hardware.device = 'cpu'
    param.Hardware.device_simulation = 'cpu'
    # XXX: end temp params from colab

    # write params
    model.parameters = param
    model.save_parameters_file = get_write_folder() + '/napari-decode-params.yaml'
    decode.utils.param_io.save_params(model.save_parameters_file, model.parameters)


def get_save_parameters_file():
    """ """

    return model.save_parameters_file



def simulate():
    """ """

    sim_train, sim_test = decode.neuralfitter.train.random_simulation.setup_random_simulation(model.parameters)
    tar_em, frames, bg_frames = sim_train.sample()
    # TODO: construct proper set of (frame, x, y) coordinates, look at decode.plot.frame_coord.py
    frame_ix = tar_em.frame_ix.numpy()
    xy_coords = tar_em.xyz.numpy()[:, :2]
    return frames.numpy(), np.concatenate((frame_ix[:, None], xy_coords), axis=1)

