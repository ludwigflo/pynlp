from nlp.text_embeddings.embedding import EmbeddingAlgorithm
from nlp.utils import read_parameter_file
import nlp.ml_utils as ml_utils
from typing import Union, List
from torch.optim import Adam
import models
import torch
import csv
import os


# TODO: Integrate training on multiple GPUs
def evaluate_prediction(prediction: torch.Tensor, target: torch.Tensor) -> tuple:
    """

    Parameters
    ----------
    prediction
    target

    Returns
    -------

    """

    prediction[prediction > 0.5] = 1
    prediction[prediction <= 0.5] = 0

    pos = 0
    num_preds = prediction.size()[0]
    for i in range(num_preds):
        if prediction[i].view(-1) == target[i]:
            pos += 1
    return pos, num_preds


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


def train_epoch(train_loader, model, optim, num_iterations, device) -> tuple:
    """
    Performs one single train epoch.

    Parameters
    ----------
    train_loader: Data loader of the training data.
    model: Pytorch model, which should be trained.
    optim: Optimizer of the model.
    num_iterations: Number of single iterations in the train epoch.
    device: Which device should dbe used for the computation operations.

    Returns
    -------
    avg_loss: Average loss value of the training epoch.
    """

    avg_loss = 0
    total_num = 0
    total_correct = 0
    for i, (data, label) in enumerate(train_loader):

        # utilize the gpu, if possible
        data, label = data.float().to(device), label.float().to(device)

        # compute the model's predictions
        prediction = model(data)

        # compute the loss
        loss = ml_utils.bce_loss_pytorch(prediction, label, logits=False)
        avg_loss += loss.detach().cpu().numpy().item()

        # backpropagate the loss and update the weights
        optim.zero_grad()
        loss.backward()
        optim.step()

        # evaluate the predictions in order to prepare the computation of the accuracy
        correct, num = evaluate_prediction(prediction, label)
        total_correct += correct
        total_num += num

        # abort criterion
        if i == num_iterations:
            break

    # compute and return the average train loss as well as the accuracy
    avg_acc = float(total_correct)/float(total_num)
    avg_loss /= num_iterations
    return avg_loss, avg_acc


def val_epoch(val_loader, model, device) -> tuple:
    """
    Performs one single train epoch.

    Parameters
    ----------
    val_loader: Data loader of the validation data.
    model: Pytorch model, which should be trained.
    device: Which device should dbe used for the computation operations.

    Returns
    -------
    avg_loss: Average loss value of the training epoch.
    """
    with torch.no_grad():
        avg_loss = 0
        num_iter = 1
        total_num = 0
        total_correct = 0
        for i, ((data, label), done) in enumerate(val_loader):

            # utilize the gpu, if possible
            data, label = data.float().to(device), label.float().to(device)

            # compute the model's predictions
            prediction = model(data)

            # compute the loss and update the average loss
            loss = ml_utils.bce_loss_pytorch(prediction, label, logits=False)
            avg_loss += loss.detach().cpu().numpy().item()

            # evaluate the predictions in order to prepare the computation of the accuracy
            correct, num = evaluate_prediction(prediction, label)
            total_correct += correct
            total_num += num

            # abort criterion
            if done:
                num_iter = i + 1
                break

        # compute and return the average train loss as well as the accuracy
        avg_acc = float(total_correct) / float(total_num)
        avg_loss /= num_iter
        return avg_loss, avg_acc


def main(params: dict) -> None:
    """
    Main function for training.

    Parameters
    ----------
    params: Training parameters.
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # get some parameters
    num_epochs = params['training']['num_epochs']
    batch_size = params['data_loader']['batch_size']
    num_iterations = params['training']['num_iterations']
    learning_rate = float(params['training']['optimizer']['lr'])
    update_embeddings = params['training']['update_embeddings']
    betas = tuple([float(x) for x in params['training']['optimizer']['betas']])

    # create a new experiment directory
    root_dir, logger_list = initialize_experiment(params)
    tb_tags = ['BCE Loss', 'Accuracy']
    print('\n\nInitialized experiment directory')

    # show current working directory and copy all required files into the experiment directory
    current_dir = os.getcwd()
    files = [x for x in os.listdir(current_dir) if os.path.isfile(x)]
    ml_utils.copy_files(root_dir+'src/', files)

    # create a data loader object and extract the train and validation loaders
    data_loader = DataLoader(25000, params, shuffle_data = True)
    data_loader.embeddings.set_embedding_gradient(update_embeddings)
    train_loader = data_loader.train_generator(batch_size = batch_size)
    val_loader = data_loader.val_generator(batch_size=1, rand=False)

    # initialize a new pytorch model
    feature_size = data_loader.embeddings.embedding.weight.data.size()[1]
    model = models.LinearClassifier(feature_size).to(device)

    # create an optimizer
    if update_embeddings:
        params = list(model.parameters()) + list(data_loader.embeddings.embedding.parameters())
        optim = Adam(params, lr = learning_rate, betas = betas)
    else:
        optim = Adam(model.parameters(), lr = learning_rate, betas = betas)

    # train the model
    for epoch in range(num_epochs):
        avg_val_loss, avg_val_acc = val_epoch(val_loader, model, device)
        avg_train_loss, avg_train_acc = train_epoch(train_loader, model, optim, num_iterations, device)
        print('Epoch: {0} Train Loss: {1}  Validation Loss: {2}'.format(epoch, avg_train_loss, avg_val_loss))
        print('  Train Accuracy: {0}  Validation Accuracy: {1}'.format(avg_train_acc, avg_val_acc))
        print()

        logger_list[0].log_scalar(tb_tags, [avg_train_loss, avg_train_acc], epoch)
        logger_list[1].log_scalar(tb_tags, [avg_val_loss, avg_val_acc], epoch)

if __name__ == '__main__':

    # read the parameters
    parameter_file = 'params.yaml'
    p = read_parameter_file(parameter_file)

    # run the training
    main(p)
