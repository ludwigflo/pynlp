from .count_based import corpus2cooc_matrix
from torch.autograd import Variable
from torch.optim import Adam
from torch.nn import MSELoss
import numpy as np
import random
import torch


class Glove:

    def __init__(self):
        """
        """

        self.cooc_matrix = None
        self.word_embeddings = None

    def train_glove_model(self, word2index: dict, corpus: list, window_size: int, vector_size: int=300,
                          num_iterations: int = 50, max_count: int=100, alpha: float=0.75) -> None:
        """
        """

        num_words = len(word2index)
        print('--------   Computing Co-occurence Matrix...   --------')
        self.cooc_matrix = corpus2cooc_matrix(word2index, corpus, window_size)
        training_indices = self.get_non_negative_element_indices(self.cooc_matrix)

        # initialize random word vectors
        u = Variable(torch.empty(num_words, vector_size).normal_(mean=0, std=1), requires_grad=True)
        v = Variable(torch.empty(num_words, vector_size).normal_(mean=0, std=1), requires_grad=True)

        # initialize biases with zero
        b_u = Variable(torch.zeros(num_words), requires_grad=True)
        b_v = Variable(torch.zeros(num_words), requires_grad=True)

        # initialize the loss function and the optimizer
        loss_function = MSELoss()
        u_optim = Adam([u, b_u], lr=1e-4)
        v_optim = Adam([v, b_v], lr=1e-4)

        # train the model
        print('--------           Start Training...          --------')
        for epoch in range(num_iterations):
            epoch_loss = 0

            # shuffle the list of training samples
            random.shuffle(training_indices)

            # for each training sample
            for iteration, training_indices in enumerate(training_indices):
                epoch_loss += self.train_iteration(u, v, training_indices, u_optim, v_optim,
                                                   max_count, alpha, loss_function)
            epoch_loss /= num_iterations
            print('Epoch: ' + str(epoch) + 'Average Loss: ' + str((epoch_loss.detach().cpu().numpy().item())))
        self.word_embeddings = u + v

    @staticmethod
    def get_non_negative_element_indices(tensor: np.ndarray) -> list:
        """
        """

        x_dim, y_dim = tensor.shape

        index_list = []

        for x in range(x_dim):
            for y in range(y_dim):
                if tensor[x, y] != 0:
                    index_list.append((x, y))
        return index_list


    @staticmethod
    def weighting_function(max_count: int, alpha: float, occurrence_count: float) -> float:
        """
        """

        if occurrence_count > max_count:
            output = 1
        else:
            output = (occurrence_count/max_count) ** alpha
        return output

    # noinspection PyArgumentList
    def train_iteration(self, u: torch.Tensor, v: torch.Tensor, b_u: torch.Tensor, b_v: torch.Tensor,
                        training_indices: tuple, u_optim: torch.optim.Optimizer, v_optim: torch.optim.Optimizer,
                        max_count: int, alpha: float, loss_function: torch.nn.Module) -> torch.Tensor:
        """
        """

        u_optim.zero_grad()
        v_optim.zero_grad()

        # get the corresponding indices in the u and v matrices
        u_index, v_index = training_indices[0], training_indices[1]

        # get the co-occurrence count values
        occurrence_count = self.cooc_matrix[u_index, v_index]

        # compute the weights for the current sample
        f = self.weighting_function(max_count, alpha, occurrence_count)

        # predict the value of the log co occurrence count with the current word vectors and biases
        prediction = torch.dot(u[u_index, ...], v[v_index, ...]) + b_u[u_index] + b_v[v_index]

        # compute the target
        target = torch.log(occurrence_count)

        # compute the loss value, backpropagate the error and update the word embeddings
        loss = f * loss_function(prediction, target)
        loss.backward()
        u_optim.step()
        v_optim.step()

        return loss
