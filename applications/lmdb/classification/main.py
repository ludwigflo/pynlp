from nlp.utils import read_parameter_file
import nlp.ml_utils as ml_utils
from typing import Union


if __name__ == '__main__':

    # read the parameters
    parameter_file = 'params.yaml'
    p = read_parameter_file(parameter_file)

    # initialize the experiment directory
    tb_log_names: Union[None, list] = p['experiment']['tb_log_names']
    tb_log: bool = p['experiment']['tb_log']
    root_dir = p['experiment']['root_dir']

    # extract the variables, which are created during experiment initialization
    if tb_log and (tb_log_names is not None):
        exp_root_dir, tb_logger_list = ml_utils.init_experiment_dir(root_dir, tb_log, tb_log_names)
    else:
        exp_root_dir = ml_utils.init_experiment_dir(root_dir, tb_log, tb_log_names)
