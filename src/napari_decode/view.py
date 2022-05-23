import napari
import numpy as np
import decode


class Model:

    def __init__(self):
        """ """

        self.params = None

    def set_params(self, params):
        """ """

        self.params = params



def parameters():
    """ """

    # Inspect parameters and adjust paths
    param_file = 'sample_data/colab_train_experimental/param_run_in.yaml'
    param = decode.utils.param_io.load_params(param_file)
    
    # set meta
    param.Meta.version = decode.utils.bookkeeping.decode_state()
    
    # the number of workers and threads shall be limited on Colab
    param.Hardware.num_worker_train = 2
    param.Hardware.torch_threads = 2
    
    # compute automatic parameters
    param = decode.utils.param_io.autoset_scaling(param)
    
    # Set the path to the calibration file
    calibration_file = 'sample_data/colab_train_experimental/spline_calibration_3dcal.mat'
    param.InOut.calibration_file = calibration_file
    
    # Set the output directory
    model_dir = 'network'
    
    # Set the directory in which the checkpoints should be saved.
    ckpt_dir = 'network'
    
    model_dir = Path(model_dir)
    
    # check if folder exists, create if at least the parent exists
    if not model_dir.parents[0].is_dir():
      raise FileNotFoundError(f"The path to the directory of 'model_out' (and even its parent folder) could not be found.")
    else:
      if not model_dir.is_dir():
        model_dir.mkdir()
        print(f"Created directory, absolute path: {model_dir.resolve()}")
    
    model_out = Path(model_dir) / 'model.pt'
    ckpt_path = Path(ckpt_dir) /'decode_ckpt.pt'
    param.InOut.experiment_out = str(model_dir)

    # Parameters that have an influence on training
    
    # Training with *Temporal Context* (i.e. 3 input frames)
    temporal_context = True
    
    if temporal_context:
      param.HyperParameter.channels_in = 3
    else:
      param.HyperParameter.channels_in = 1
    
    # Average number of emitters in 40x40 sized frame patch.
    emitter_average = 25
    param.Simulation.emitter_av = emitter_average
    
    # Range of emitters in axial direction [nm]
    z_min = -800
    z_max = 800
    param.Simulation.emitter_extent[2] = [z_min, z_max]
    
    # Learning rate (user lower value if training fails)
    lr = 0.0006
    param.HyperParameter.opt_param.lr = lr
    
    param.Hardware.device = 'cuda:0'
    param.Hardware.device_simulation = 'cuda:0'
    

def simulate(
    size: int = 64,
) -> napari.types.LayerDataTuple:
    """ """

    m = Model()
    print(m.parameters)
    m.set_params( ('a', 'b', 'c') )
    print(m.parameters)
    data = np.arange(size * size, dtype=np.uint16).reshape((size, size))
    return (
        data,
        {"name": "data"},
    )
