import matplotlib.pyplot as plt
from typing import Union
import numpy as np
import math


# lambda function for the l2 distance
l2_distance = lambda x, y: np.sqrt(np.sum([(x[i] - y[i])**2 for i in range(x.shape[0])]))


class Neuron:
    def __init__(self, node_number: int, feature_values: np.ndarray) -> None:
        """
        Constructor of the Neuron class.

        Parameters
        ----------
            node_number: Id of the node in order to identify it in the whole network.
            feature_values: Values if the node's features in the input space.
        """

        self.feature_values = feature_values
        self.is_closest_neighbour = False
        self.is_closest_point = False
        self.node_number = node_number
        self.neighbour_list = []
        self.age_list = []
        self.error = 0

    def add_neighbour(self, n: 'Neuron', reverse=True) -> 'Neuron':
        """
        Adds an edge between the current node and the provided node.

        Parameters
        ----------
            n: Neuron, which should be added as neighbour, if it isn't already.
            reverse:

        Returns
        -------
            n: Newly added neighbour neuron.
        """

        # add the provided neuron as neighbour, if it isn't already. Initialize the edge_age to 0.
        edge_age = 0
        if n.node_number not in self.neighbour_list:
            self.neighbour_list.append(n.node_number)
            self.age_list.append(edge_age)

        # if the neighbour already exists, then reset its edge age to 0.
        else:
            # find the index of the node and reset its edge age
            index = self.neighbour_list.index(n.node_number)
            self.age_list[index] = edge_age
        if reverse:
            n.add_neighbour(self, reverse=False)
        return n

    def accumulate_error(self, error: float) -> None:
        """
        Adds an error to the internally stored error variable.
        """
        self.error += error

    def reset_error(self, default_val: float=0.0) -> None:
        """
        Resets the internally stored error variable to a default value.
        """
        self.error = default_val

    def remove_neighbour(self, n: 'Neuron', reverse=True) -> 'Neuron':
        """
        Removes a Neuron from the neighbour list.

        Parameters
        ----------
            n: Neuron, which should be removed as neighbour, if it isn't already.

        Returns
        -------
            n: Newly removed neighbour neuron.
        """

        if n.node_number in self.neighbour_list:
            index = self.neighbour_list.index(n.node_number)
            del self.age_list[index]
            self.neighbour_list.remove(n.node_number)
        if reverse:
            n.remove_neighbour(self, reverse=False)
        return n

    def increment_edge_age(self, node: Union[None, 'Neuron']=None) -> list:
        """
        Increments the edge age for all neighbour nodes or, if provided, for a specific neighbour node.

        Parameters
        ----------
            node: Specific Neuron, to which the edge age should be incremented (Optional).

        Returns
        -------
            output: List, representing the node indices, for which the edge age have been increased.
        """

        # increment the edge age of all neighbour nodes, if no special neighbour was specified
        if node is None:
            self.age_list = [x+1 for x in self.age_list]
            output = self.neighbour_list

        # if a special neighbour node was specified
        else:
            index = self.neighbour_list.index(node.node_number)
            self.age_list[index] += 1
            output = [index]
        return output

    def move_in_feature_direction(self, target: np.ndarray, step_size: float) -> None:
        """
        Moves the features of the neuron into a direction in the feature space.

        Parameters
        ----------
            target: Target, in which the features should be adapted.
            step_size: Value representing, how large the adaptation should be.
        """

        # compute the direction, in which the features are moved
        direction = target - self.feature_values

        # adapt the features in direction of the target with a step size, provided by the user
        self.feature_values += step_size * direction


class GrowingNeuralGas:
    def __init__(self, num_nodes: int=2, feature_dim: int=3, dist_fun=l2_distance, step_size: tuple=(0.2, 0.006),
                 alpha: float=0.5, max_age: int=50, decreasing_constant: float=0.995) -> None:
        """
        Constructor for the class GrowingNeuralGas.

        Parameters
        ----------
            num_nodes: Number of initial nodes.
            feature_dim: Dimension of the feature space, in which the neural gas algorithm operates.
            dist_fun: Function for computing the l2 distance.
            step_size:
            alpha:
        """

        # initialize lists, which represent nodes and edges
        self.node_list = self.init_nodelist(num_nodes, feature_dim)

        # store important properties
        self.alpha = alpha
        self.max_age = max_age
        self.dis_fun = dist_fun
        self.step_size = step_size
        self.num_nodes = num_nodes
        self.decreasing_constant = decreasing_constant

    @staticmethod
    def init_nodelist(num_nodes: int, feature_dim: int) -> list:
        """
        Initializes the node list by adding a specified number of init nodes to a node list (minimal number: 2).

        Parameters
        ----------
            num_nodes: number of nodes, which are used for initializing the node list (minimal number: 2).
            feature_dim: number of features, each node consists of.
        Returns
        -------
            node_list: List of nodes, which have ben initialized randomly (normal distributed).
        """

        node_list = []
        for i in range(num_nodes):
            features = np.random.randn(feature_dim)
            node_list.append(Neuron(i, features))
        return node_list

    # noinspection PyUnboundLocalVariable
    def insert_new_node(self) -> None:
        """
        Inserts a new node to the node list, in order to grow the neural gas.
        """

        # search for the node with the highest error and get its neighbour nodes
        highest_error = -math.inf
        for i, node in enumerate(self.node_list):
            if node.error > highest_error:
                highest_error = node.error
                highest_error_index = i
                highest_error_node = node
        neighbour_list = highest_error_node.neighbour_list

        # search for the neighbour with the highest error
        highest_error = -math.inf
        for neighbour_index in neighbour_list:
            neighbour_node = self.node_list[neighbour_index]
            if neighbour_node.error > highest_error:
                highest_error = neighbour_node.error
                highest_error_neighbour = neighbour_node
                highest_error_neighbour_index = neighbour_index

        # Create a new node
        features = 0.5 * highest_error_node.feature_values + 0.5 * highest_error_neighbour.feature_values
        node_index = self.num_nodes
        self.num_nodes += 1
        new_node = Neuron(node_index, features)

        # insert the node between the highest error node and its highest error neighbour
        highest_error_node = new_node.add_neighbour(highest_error_node)
        highest_error_neighbour = new_node.add_neighbour(highest_error_neighbour)

        # remove the edge connection between the highest error node and its highest error neighbour
        highest_error_neighbour = highest_error_node.remove_neighbour(highest_error_neighbour)

        # decrease the error variable from the highest error node and its neighbour
        highest_error_node.error *= self.alpha
        highest_error_neighbour.error *= self.alpha

        # store the changes in the node list
        self.node_list.append(new_node)
        self.node_list[highest_error_index] = highest_error_node
        self.node_list[highest_error_neighbour_index] = highest_error_neighbour

    def find_nearest_nodes(self, input_val: np.ndarray, num_nodes: int=1) -> list:
        """
        Returns a list of the nearest neighbour nodes, given an input value in the feature space.

        Parameters
        ----------
            input_val: Value, to which the distance of the nodes should be computed.
            num_nodes: Number of nodes, which should be returned from the function.

        Returns
        -------
            nearest_nodes_list: List of the n nearest nodes to the input, where n is provided by the user (default 1).
        """

        # compute the distances for each node from the input value
        distance_list = []
        for node in self.node_list:
            node_id = node.node_number
            features = node.feature_values
            distance = self.dis_fun(features.reshape(-1), input_val.reshape(-1))
            distance_list.append((node_id, distance))

        # sort the distance list and select the nearest nodes
        distance_list = sorted(distance_list, key=lambda x: x[1])

        nearest_nodes_list = [self.node_list[x[0]] for x in distance_list[:num_nodes]]
        return nearest_nodes_list

    def delete_node(self, node) -> None:
        """
        Deletes a single node from the node list.

        Parameters
        ----------
            node: Node, which should be deleted.
        """

        # remove the node
        del self.node_list[node.node_number]
        self.num_nodes -= 1

        # adapt the indices in the node list
        for node_temp in self.node_list[node.node_number:]:
            node_temp.node_number -= 1

        # adapt the indices in the neighbour lists of the nodes
        for node_temp in self.node_list:
            node_temp.neighbour_list = [x - 1 if x > node.node_number else x for x in node_temp.neighbour_list]

    def train_step(self, data_point: np.ndarray, add_node: bool=False):
        """
        One single train iteration in the training loop.

        Parameters
        ----------
            data_point:
            add_node:
        """

        # get the nearest nodes from the provided data point
        nearest_nodes_list = self.find_nearest_nodes(data_point, 2)
        nearest_node, second_nearest_node = nearest_nodes_list[0], nearest_nodes_list[1]
        nearest_node.is_closest_point = True
        second_nearest_node.is_closest_neighbours = True

        # increment the edge ages from all neighbours of the nearest node
        neighbour_list = nearest_node.increment_edge_age()
        for neighbour_index in neighbour_list:
            neighbour = self.node_list[neighbour_index]
            neighbour.increment_edge_age(nearest_node)
            self.node_list[neighbour_index] = neighbour

        # get the distance from the data_point to its nearest node and add this distance as error term
        distance = self.dis_fun(nearest_node.feature_values, data_point)
        nearest_node.error += distance

        # move the node and its topological neighbours in direction of the data point
        nearest_node.move_in_feature_direction(data_point, self.step_size[0])
        for i, neighbour_index in enumerate(neighbour_list):
            neighbour = self.node_list[neighbour_index]
            neighbour.move_in_feature_direction(data_point, self.step_size[1])
            edge_age = nearest_node.age_list[i]
            if edge_age > self.max_age:
                neighbour = nearest_node.remove_neighbour(neighbour)
            self.node_list[neighbour_index] = neighbour

        # add the second nearest node as neighbour to the first one
        second_nearest_node = nearest_node.add_neighbour(second_nearest_node)

        # add the nodes to the node list
        self.node_list[second_nearest_node.node_number] = second_nearest_node
        self.node_list[nearest_node.node_number] = nearest_node

        # if a new node should be inserted
        if add_node:
            self.insert_new_node()

        # remove nodes which don't have any connection decrease the error term for all nodes a little bit
        for node in self.node_list:

            # if there is no edge connection to neighbour nodes
            if len(node.neighbour_list) == 0:

                # remove the node
                self.delete_node(node)



if __name__ == '__main__':

    data_shape = (30000, 2)
    lam = 50
    data_points = np.random.rand(*data_shape) +  3*np.random.randint(0, 2, data_shape)
    data_points[:, 1] = np.sin(data_points[:, 0]) + data_points[:, 1]

    gas = GrowingNeuralGas(feature_dim=data_shape[1])
    for i in range(data_shape[0]):
        print(i)
        data_sample = data_points[i, ...]
        if i%lam==0 and i>0:
            gas.train_step(data_sample, add_node=True)
        else:
            gas.train_step(data_sample, add_node=False)

    # remove nodes, which have never been touched by the algorithm
    for node in gas.node_list:
        if not node.is_closest_point or node.is_closest_neighbour:
            gas.delete_node(node)

    for node in gas.node_list:
        features = node.feature_values
        plt.scatter(features[0], features[1])
    plt.show()