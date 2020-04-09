from nlp.text_embeddings.embedding import EmbeddingAlgorithm
from nlp.utils import read_parameter_file
import nlp.ml_utils as ml_utils
from typing import Union, List
from torch.optim import Adam
import numpy as np
import models
import torch
import csv


class DataLoader(ml_utils.data_loader.DataLoaderInterface):

    def __init__(self, total_num_data: int, params: dict, shuffle_data: bool = True):
        """
        Constructor of the DataLoader Class.

        Parameters
        ----------
        total_num_data: Total number of data samples.
        params: Parameters, stored in form of a dictionary.
        shuffle_data: Variable, which determines whether to shuffle the data before splitting or not.
        """

        super(DataLoader, self).__init__(total_num_data, params, shuffle_data)

        # get the directories, where the embeddings and the labels are stored
        embedding_path = params['data_loader']['embedding_path']
        label_path = params['data_loader']['label_path']

        # store the embeddings and the labels
        self.embeddings = EmbeddingAlgorithm.load_object(embedding_path)

        with open(label_path, 'r') as f:
            csv_reader = csv.reader(f)
            self.labels = []
            for line in csv_reader:
                self.labels.append(int(line[1]))
        self.labels = torch.Tensor(self.labels)

    def load_data(self, index: Union[List[int], int]) -> tuple:
        """
        Loading one or multiple data sample by its index or a list of indices.

        Parameters
        ----------
        index: Index or list of indices of documents.

        Returns
        -------
        embeddings: Document embeddings for the queried document indices.
        labels: Corresponding class labels.
        """

        if type(index) == int:
            index = [index]
        embeddings = self.embeddings.embedding(torch.LongTensor(index))
        labels = self.labels[index]
        return embeddings, labels


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


def main(params: dict) -> None:
    """
    Main function for training.
    TODO: Add embedding update to optimizer
    Parameters
    ----------
    params: Training parameters.
    """

    # get some parameters
    num_epochs = params['training']['num_epochs']
    batch_size = params['data_loader']['batch_size']
    num_iterations = params['training']['num_iterations']
    learning_rate = float(params['training']['optimizer']['lr'])
    update_embeddings = params['training']['update_embeddings']
    betas = tuple([float(x) for x in params['training']['optimizer']['betas']])

    # create a new experiment directory
    # root_dir, logger_list = initialize_experiment(params)
    print('\n\nInitialized experiment directory')

    # create a data loader object and extract the train and validation loaders
    data_loader = DataLoader(25000, params, shuffle_data = True)
    data_loader.embeddings.set_embedding_gradient(update_embeddings)
    train_loader = data_loader.train_generator(batch_size = batch_size)

    # initialize a new pytorch model
    feature_size = data_loader.embeddings.embedding.weight.data.size()[1]
    model = models.LinearClassifier(feature_size)

    # create an optimizer
    optim = Adam(model.parameters(), lr = learning_rate, betas = betas)

    # utilize the gpu, if possible
    cuda = torch.cuda.is_available()
    if cuda:
        model = model.cuda()

    # train the model
    for epoch in range(num_epochs):
        for i, (data, label) in enumerate(train_loader):

            # utilize the gpu, if possible
            if cuda:
                data, label = data.float().cuda(), label.float().cuda()

            # compute the model's predictions
            prediction = model(data)

            # compute the loss
            loss = ml_utils.bce_loss_pytorch(prediction, label, logits = False)
            print(epoch, i, np.asscalar(loss.detach().cpu().numpy()))

            # backpropagate the loss and update the weights
            optim.zero_grad()
            loss.backward()
            optim.step()

            # abort criterion
            if i == num_iterations:
                break

if __name__ == '__main__':

    # read the parameters
    parameter_file = 'params.yaml'
    p = read_parameter_file(parameter_file)

    # run the training
    main(p)
