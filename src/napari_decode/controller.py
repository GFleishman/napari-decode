import os, tempfile, requests, yaml
from pathlib import Path
import decode
import numpy as np
from decode.neuralfitter.train import live_engine
from decode.neuralfitter.utils import logger as logger_utils
import glob
from datetime import datetime
from collections.abc import MutableMapping


def download_parameters():
    """ """

    temp_folder = ndm.temp_folder.name
    with open(temp_folder + '/gateway.yaml', 'wb') as f:
        f.write(requests.get(ndm.GATEWAY_URL).content)
    with open(temp_folder + '/gateway.yaml', 'r') as f:
        gateway = yaml.safe_load(f)
    package = gateway['examples']['colab_train_experimental_rc']
    sample_folder = decode.utils.example_helper.load_example_package(
        path=Path(temp_folder + '/' + package['name'] + '.zip'),
        url=package['url'],
        hash=package['hash'],
    )
    old_path = temp_folder + '/colab_train_experimental/param_run_in.yaml'
    suffix = 'download-' + datetime.now().strftime('%d%m%Y%H%M%S') + '.yaml'
    new_path = ndm.LOGS_FOLDER + '/' + ndm.PARAMS_PREFIX + suffix
    os.rename(old_path, new_path)
    return [new_path,]


def most_recent_parameters(param_files):
    """ """

    index = 0
    latest = param_files[0].split('-')[-1].split('.')[0]
    latest = datetime.strptime(latest, '%d%m%Y%H%M%S')
    for iii in range(1, len(param_files)):
        dt = param_files[iii].split('-')[-1].split('.')[0]
        dt = datetime.strptime(dt, '%d%m%Y%H%M%S')
        if dt > latest:
            index, latest = iii, dt
    return param_files[index]


def load_parameters(param_file):
    """ """

    rns = decode.utils.param_io.load_params(param_file)
    return rns.to_dict()


def initialize_parameters(model):
    """ """

    global ndm
    ndm = model
    os.makedirs(ndm.LOGS_FOLDER, exist_ok=True)
    ndm.temp_folder = tempfile.TemporaryDirectory()
    param_files = glob.glob(ndm.LOGS_FOLDER + '/' + ndm.PARAMS_PREFIX + '*') or \
        download_parameters()
    param_file = most_recent_parameters(param_files)
    ndm.parameters = load_parameters(param_file)










def set_write_folder(write_folder):
    """ """

    write_folder = write_folder or os.getcwd() + '/napari-decode-outputs'
    os.makedirs(write_folder, exist_ok=True)
    ndm.write_folder = write_folder

def get_write_folder():
    """ """

    return ndm.write_folder


def set_parameters(parameters_file):
    """ """

    # if no parameter file given, download default parameters
    if not parameters_file:
        gateway_file = get_temp_folder() + '/gateway.yaml'
        with open(gateway_file, 'wb') as f:
            f.write(requests.get(ndm.GATEWAY_URL).content)
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
    ndm.input_parameters_file = parameters_file
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
    ndm.parameters = param
    ndm.save_parameters_file = get_write_folder() + '/napari-decode-params.yaml'
    decode.utils.param_io.save_params(ndm.save_parameters_file, ndm.parameters)


def get_save_parameters_file():
    """ """

    return ndm.save_parameters_file


def simulate():
    """ """

    sim_train, sim_test = decode.neuralfitter.train.random_simulation.setup_random_simulation(ndm.parameters)
    ndm.sim_train = sim_train
    ndm.sim_test = sim_test
    tar_em, frames, bg_frames = sim_train.sample()
    # TODO: construct proper set of (frame, x, y) coordinates, look at decode.plot.frame_coord.py
    frame_ix = tar_em.frame_ix.numpy()
    xy_coords = tar_em.xyz.numpy()[:, :2]
    return frames.numpy(), np.concatenate((frame_ix[:, None], xy_coords), axis=1)


def setup_trainer():
    """ """

    # TODO: ensure all params are defined, this function needs inputs
    logger = [logger_utils.SummaryWriter(
                log_dir='logs', 
                filter_keys=["dx_red_mu", "dx_red_sig", 
                             "dy_red_mu", "dy_red_sig", 
                             "dz_red_mu", "dz_red_sig",
                             "dphot_red_mu", "dphot_red_sig",
                             "f1",]
              ), logger_utils.DictLogger()]
    logger = logger_utils.MultiLogger(logger)

    # TODO: check if these are defined first
    X = live_engine.setup_trainer(
        ndm.sim_train,
        ndm.sim_test,
        logger,
        get_write_folder() + '/model.pt',
        get_write_folder() + '/decode_ckpt.pt',
        'cpu',
        ndm.parameters,
    )
    ds_train = X[0]
    ds_test = X[1]
    model = X[2]
    model_ls = X[3]
    optimizer = X[4]
    criterion = X[5]
    lr_scheduler = X[6]
    grad_mod = X[7]
    post_processor = X[8]
    matcher = X[9]
    ckpt = X[10]

    dl_train, dl_test = live_engine.setup_dataloader(ndm.parameters, ds_train, ds_test)

    epoch0 = 0

    # TODO: decide what to do about starting from a checkpoint
#    ckpt_path = get_write_folder + '/decode_ckpt.pt'
#    ckpt = decode.utils.checkpoint.CheckPoint.load(ckpt_path)
#    model.load_state_dict(ckpt.model_state)
#    optimizer.load_state_dict(ckpt.optimizer_state)
#    lr_scheduler.load_state_dict(ckpt.lr_sched_state)
#    epoch0 = ckpt.step + 1
#    model = model.train()

    # TODO: training from scratch
#    converges = False
#    n = 0
#    n_max = param.HyperParameter.auto_restart_param.num_restarts
#    
#    while not converges and n < n_max:
#      n += 1
#      
#      conv_check = decode.neuralfitter.utils.progress.GMMHeuristicCheck(
#          ref_epoch=1,
#          emitter_avg=sim_train.em_sampler.em_avg,
#          threshold=param.HyperParameter.auto_restart_param.restart_treshold,
#      )
#      
#      for i in range(epoch0, param.HyperParameter.epochs):
#        logger.add_scalar('learning/learning_rate', optimizer.param_groups[0]['lr'], i)
#        
#        if i >= 1:
#          train_loss = decode.neuralfitter.train_val_impl.train(
#              model=model,
#              optimizer=optimizer,
#              loss=criterion,
#              dataloader=dl_train,
#              grad_rescale=param.HyperParameter.moeller_gradient_rescale,
#              grad_mod=grad_mod,
#              epoch=i,
#              device=torch.device(param.Hardware.device),
#              logger=logger
#          )
#        
#        val_loss, test_out = decode.neuralfitter.train_val_impl.test(model=model, loss=criterion, dataloader=dl_test,
#                                                                      epoch=i,
#                                                                      device=torch.device(param.Hardware.device))
#    
#        if not conv_check(test_out.loss[:, 0].mean(), i):
#          print(f"The model will be reinitialized and retrained due to a pathological loss. "
#                f"The max. allowed loss per emitter is {conv_check.threshold:.1f} vs."
#                f" {(test_out.loss[:, 0].mean() / conv_check.emitter_avg):.1f} (observed).")
#    
#          ds_train, ds_test, model, model_ls, optimizer, criterion, lr_scheduler, grad_mod, post_processor, matcher, ckpt = \
#              decode.neuralfitter.train.live_engine.setup_trainer(sim_train, sim_test, logger, model_out, ckpt_path, device, param)
#          dl_train, dl_test = decode.neuralfitter.train.live_engine.setup_dataloader(param, ds_train, ds_test)
#          
#          converges = False
#          break
#    
#        else:
#          converges = True
#    
#        """Post-Process and Evaluate"""
#        decode.neuralfitter.train.live_engine.log_train_val_progress.post_process_log_test(loss_cmp=test_out.loss, loss_scalar=val_loss,
#                                                      x=test_out.x, y_out=test_out.y_out, y_tar=test_out.y_tar,
#                                                      weight=test_out.weight, em_tar=ds_test.emitter,
#                                                      px_border=-0.5, px_size=1.,
#                                                      post_processor=post_processor, matcher=matcher, logger=logger,
#                                                      step=i)
#    
#        if i >= 1:
#            if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
#                lr_scheduler.step(val_loss)
#            else:
#                lr_scheduler.step()
#    
#        model_ls.save(model, None)
#        ckpt.dump(model.state_dict(), optimizer.state_dict(), lr_scheduler.state_dict(),
#                          log=logger.logger[1].log_dict, step=i)
#    
#        """Draw new samples Samples"""
#        if param.Simulation.mode in 'acquisition':
#            ds_train.sample(True)
#        elif param.Simulation.mode != 'samples':
#            raise ValueError
#    
#    if converges:
#        print("Training finished after reaching maximum number of epochs.")
#    else:
#        raise ValueError(f"Training aborted after {n_max} restarts. "
#                          "You can try to reduce the learning rate by a factor of 2."
#                          "\nIt is also possible that the simulated data is to challenging. "
#                          "Check if your background and intensity values are correct "
#                          "and possibly lower the average number of emitters.")
