import networkx as nx
from copy import copy
import numpy as np
from typing import Optional

from matplotlib import pyplot as plt


class QVoter:
    """ q_a-voter model with NN influence group. """

    def __init__(self, init_network: nx.Graph):
        self.init_network = init_network
        self.network_size = init_network.size()

        self.operating_network = None
        self.operating_opinion = None
        self.operating_magnetization = []

    def reload_operating_network(self):
        """ Operating network is needed for Monte Carlo trajectories. """
        self.operating_network = copy(self.init_network)

    def reload_operating_opinion(self):
        """ Method initializing opinion of the spinsons to 1. In future this could be changed and improved. """
        self.operating_opinion = {node: 1 for node in self.init_network.nodes}

    def reload_operating_magnetization(self):
        self.operating_magnetization = []

    def influence_choice(self, spinson: int, q: int, type_of_influence: str = 'RND_no_repetitions') -> list:
        """ Method returning spinsons from the network to affect given <spinson (int)> according to given theoretical
            <type_of_influence (int)>
        Args:
            spinson (int):  given spinson.
            q (int): number of people in the influence group
            type_of_influence (str): type of choice of the influence group.
        """

        if type_of_influence == 'RND_no_repetitions':
            # 'q randomly chosen nearest neighbours of the target spinson are in the group. No repetitions.'
            return list(
                set(np.random.choice([neighbour for neighbour in self.operating_network.neighbors(spinson)], q)))
        elif type_of_influence == 'NN':
            # 'q randomly chosen nearest neighbours of the target spinson are in the group.'
            return np.random.choice([neighbour for neighbour in self.operating_network.neighbors(spinson)], q)
        else:
            # in the future there may be other ways of choice implemented as well
            raise NotImplementedError

    def unanimous_check(self, group: list[int]):
        """ Method checking if the group is unanimous.
        Args:
            group (list[int]): Given group"""
        # only if (all are equal to 1) v (all are equal to -1)  <==> abs(sum(group_opinions)) = len(group)
        opinions = [self.operating_opinion[member] for member in group]
        return abs(sum(opinions)) == len(group)

    def single_step(self, p: float, q_a: int, q_c: int, type_of_influence: str = 'RND_no_repetitions'):
        """ Single event. According to the paper: https://www.nature.com/articles/s41598-021-97155-0
        Args:
            p (flaot): 0 <= p <= 1. Probability for spinson to be independent
            q_a (int): number of people in the influence group for independent spinson (anticonformity case)
            q_c (int): number of people in the influence group for not independent spinson (conformity case)
            type_of_influence (str): type of choice of the influence group.
        """
        # (1) 'pick a spinson at random'
        spinson = np.random.choice(self.operating_network.nodes, 1)[0]

        # (2) 'decide with probability p, if the spinson will act as independent
        if np.random.random() < p:
            # (3) if independent, change it's opinion (opposite of the group opinion)
            influence_group = self.influence_choice(spinson, q_a, type_of_influence)
            # only if the q_a-panel is unanimous
            if self.unanimous_check(influence_group):
                self.operating_opinion[spinson] = -1 * self.operating_opinion[list(influence_group)[0]]
        else:
            # (4) if not independent, let the spinson take the opinion of its randomly chosen group of influence.
            influence_group = self.influence_choice(spinson, q_c, type_of_influence)
            # only if the q_a-panel is unanimous
            if self.unanimous_check(influence_group):
                self.operating_opinion[spinson] = self.operating_opinion[list(influence_group)[0]]

    def simulate(self, num_of_events: int, p: float, q_a: int, q_c, type_of_influence: str = 'RND_no_repetitions'):
        """ Method simulating the opinion spread: <num_of_events> steps.
        Args:
            num_of_events: number of iterations (time).
            p (flaot): 0 <= p <= 1. Probability for spinson to be independent
            q_a (int): number of people in the influence group for independent spinson (anticonformity case)
            q_c (int): number of people in the influence group for not independent spinson (conformity case)
            type_of_influence (str): type of choice of the influence group.
        """
        self.initialize_simulation()

        for event in range(num_of_events):
            # single iteration
            self.single_step(p, q_a, q_c, type_of_influence)
            # add current magnetization to the list
            self.update_magnetization_list()

        return self.operating_magnetization

    def simulate_until_stable(self, min_iterations: int, ma_value: int, p: float, q_a: int, q_c,
                              type_of_influence: str = 'RND_no_repetitions',
                              max_iterations: Optional[int] = 10 ** 5) -> tuple[list, int]:
        """ Method simulating the opinion spread. Simulates it until <max_iterations> iterations. From min_iterations
            algorithm compares actual value with MA(<ma_value>) previous values*. MA stands for moving average.
            For optimization purposes, check is done once a <ma_value_iterations>
        * it does that by checking variance of these Values. If variance is equal to 0, loop breaks.

        Args:
            min_iterations (int): minimal number of iterations
            ma_value (int): number of values to compare with actual value
            eps (int): difference between actual value and (moving) average of <ma_value> previous values
            p (flaot): 0 <= p <= 1. Probability for spinson to be independent
            q_a (int): number of people in the influence group for independent spinson (anticonformity case)
            q_c (int): number of people in the influence group for not independent spinson (conformity case)
            type_of_influence (str): type of choice of the influence group.
            min_iterations (Optional[int]): maximal number of iterations
        Returns:
            (list): magnetization
            (int): number of iterations
        """
        self.initialize_simulation()

        num_of_iter = 0
        iter_from_last_check = 0
        while True:
            num_of_iter += 1
            iter_from_last_check += 1
            # single iteration
            self.single_step(p, q_a, q_c, type_of_influence)
            # add current magnetization to the list
            self.update_magnetization_list()

            # check for break
            if num_of_iter > min_iterations:
                if iter_from_last_check > ma_value:
                    if self.operating_magnetization[-1] == self.operating_magnetization[-2] and \
                            np.var(self.operating_magnetization[-1 * ma_value:]) == 0:
                        break
                    iter_from_last_check = 0

            if num_of_iter >= max_iterations:
                break

        return self.operating_magnetization, num_of_iter

    def calculate_magnetization(self):
        """ Method calculating magnetization. """
        return np.mean(list(self.operating_opinion.values()))

    def update_magnetization_list(self):
        """ Method updating magnetization list with current magnetization. """
        self.operating_magnetization.append(self.calculate_magnetization())

    def initialize_simulation(self):
        """ Method initializing operating values, i.e. clearing them. """
        # cleaning operating network
        self.reload_operating_network()
        # cleaning operating opinion
        self.reload_operating_opinion()
        # cleaning magnetization
        self.reload_operating_magnetization()


if __name__ == "__main__":
    """ simple check of methods."""
    n = 100
    m = 2
    network = nx.watts_strogatz_graph(100,4,0.02)
    # print(network)

    q_voter = QVoter(network)

    mag, len_mag = q_voter.simulate_until_stable(min_iterations=1000, max_iterations=100000, ma_value=100, p=0.1, q_a=3, q_c=4)
    print(mag, len_mag)
    plt.scatter(np.linspace(1, len_mag, len_mag), mag, s=10)
    plt.show()