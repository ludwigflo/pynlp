from nlp.utils import read_parameter_file
import nlp.ml_utils as ml_utils
from typing import Union




def initialize_experiment(params: dict) -> Union[tuple, str]:
    """
    Creates a directory for the current experiment and (optionally) initializes tensorboard loggers.

    Parameters
    ----------
    params: Parameters, which define the root directory and indicate, whether to initialize Tensorboard loggers.

    Returns
    -------
    exp_root_dir: Directory, in which logs and results of the Experiment are stored.
    tb_logger_list: (Optionally) A list of tensorboard loggers, which have been initialized within the experiment path.
    """

    # initialize the experiment directory
    tb_log_names: Union[None, list] = params['experiment']['tb_log_names']
    tb_log: bool = params['experiment']['tb_log']
    root_dir = params['experiment']['root_dir']

    # extract the variables, which are created during experiment initialization
    if tb_log and (tb_log_names is not None):
        exp_root_dir, tb_logger_list = ml_utils.init_experiment_dir(root_dir, tb_log, tb_log_names)
        return exp_root_dir, tb_logger_list
    else:
        exp_root_dir = ml_utils.init_experiment_dir(root_dir, tb_log, tb_log_names)
        return exp_root_dir


if __name__ == '__main__':

    # read the parameters
    parameter_file = 'params.yaml'
    p = read_parameter_file(parameter_file)
    initialize_experiment(p)
