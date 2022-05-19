import networkx as nx
from copy import copy
import numpy as np


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

    def influence_choice(self, spinson: int, q: int, type_of_influence: str = 'NN') -> list:
        """ Method returning spinsons from the network to affect given <spinson (int)> according to given theoretical
            <type_of_influence (int)>
        Args:
            spinson (int):  given spinson.
            q (int): number of people in the influence group
            type_of_influence (str): type of choice of the influence group.
        """
        if type_of_influence == 'NN':
            # 'q_a randomly chosen nearest neighbours of the target spinson are in the group.'
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

    def single_step(self, p: float, q: int, type_of_influence: str = 'NN'):
        """ Single event accroding to the paper.
        Args:
            p (flaot): 0 <= p <= 1. Probability for spinson to be independent
            q (int): number of people in the influence group
            type_of_influence (str): type of choice of the influence group.
        """
        # (1) 'pick a spinson at random'
        spinson = np.random.choice(self.operating_network.nodes, 1)[0]

        # (2) 'decide with probability p, if the spinson will act as independent
        if np.random.random() < p:
            # (3) if independent, change it's opinion with probability 1/2
            if np.random.random() < 0.5:
                opinion = self.operating_opinion[spinson]
                self.operating_opinion[spinson] = -1 * opinion
        else:
            # (4) if not independent, let the spinson take the opinion of its randomly chosen group of influence.
            influence_group = self.influence_choice(spinson, q, type_of_influence)
            # only if the q_a-panel is unanimous
            if self.unanimous_check(influence_group):
                self.operating_opinion[spinson] = self.operating_opinion[list(influence_group)[0]]
            # TODO: there could be also a part from original model, but it's not part of this model:
            #       else: spinson flips it's opinion with probability <eps>.

    def simulate(self, num_of_events: int, p: float, q: int, type_of_influence: str = 'NN'):
        """ Method simulating the opinion spread: <num_of_events> steps.
        Args:
            num_of_events: number of iterations (time).
            p (flaot): 0 <= p <= 1. Probability for spinson to be independent
            q (int): number of people in the influence group
            type_of_influence (str): type of choice of the influence group.
        """
        self.initialize_simulation()

        for event in range(num_of_events):
            # single iteration
            self.single_step(p, q, type_of_influence)
            # add current magnetization to the list
            self.update_magnetization_list()

        return self.operating_magnetization

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
    n = 10
    m = 2
    network = nx.barabasi_albert_graph(n, m)
    # print(network)

    q_voter = QVoter(network)

    mag = q_voter.simulate(num_of_events=10, p=0.2, q=3)
    print(mag)