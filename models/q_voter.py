import networkx as nx
import networkit as nk
from copy import copy
import numpy as np
from typing import Optional

from matplotlib import pyplot as plt
from joblib import Parallel, delayed



class QVoter:
    """ q_a-voter model with NN influence group. """

    def __init__(self, init_network: nk.Graph):
        self.init_network = init_network
        self.network_size = init_network.numberOfNodes()

        self.operating_network = None
        self.operating_opinion = None
        self.operating_magnetization = []

    def reload_operating_network(self):
        """ Operating network is needed for Monte Carlo trajectories. """
        self.operating_network = copy(self.init_network)

    def reload_operating_opinion(self, init_type: str = "disordered_exact_fraction", *args, **kwargs):
        """ Method initializing opinion of the spinsons to 1. In future this could be changed and improved. """
        if init_type == "disordered_exact_fraction":
            self.operating_opinion = self.create_opinion_exact_fraction(*args, **kwargs)
        elif init_type == "p_for_positive":
            self.operating_opinion = self.create_opinion_according_to_p(*args, **kwargs)
        elif init_type == "all_positive":
            self.operating_opinion = {node: 1 for node in self.init_network.iterNodes()}
        elif init_type == "all_negative":
            self.operating_opinion = {node: -1 for node in self.init_network.iterNodes()}
        else:
            raise NotImplementedError

    def create_opinion_according_to_p(self, c: float = 0.5) -> dict:
        """ Function creating vector of opinions according to p for 1 """
        return {node: np.random.choice((-1, 1), p=(c, 1-c)) for node in self.init_network.iterNodes()}

    def create_opinion_exact_fraction(self, c: float = 0.5) -> dict:
        """ Function creating vector of opinions according to p for 1 """
        frac = round(self.network_size * c)

        positive_nodes = np.random.choice(list(self.init_network.iterNodes()), frac, replace=False)

        positive_opinions = {node: 1 for node in positive_nodes}
        negative_opinions = {node: -1 for node in self.init_network.iterNodes() if node not in positive_nodes}

        # FYI from Python 3.9 '|' operator merges 2 dictionaries. Doesn't work for lists :/
        return positive_opinions | negative_opinions

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
            neighbours = [neighbour for neighbour in self.operating_network.iterNeighbors(spinson)]
            if len(neighbours) < q:
                return neighbours
            else:
                return np.random.choice(neighbours, q, replace=False)
        elif type_of_influence == 'NN':
            # 'q randomly chosen nearest neighbours of the target spinson are in the group.'
            return np.random.choice([neighbour for neighbour in self.operating_network.iterNeighbors(spinson)], q)
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
        spinson = np.random.choice(list(self.operating_network.iterNodes()), 1)[0]

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
                              max_iterations: Optional[int] = 10 ** 5, opinion_init: str = "disordered_exact_fraction",
                              *args, **kwargs) -> tuple[list, int, float]:
        """ Method simulating the opinion spread. Simulates it until <max_iterations> iterations. From min_iterations
            algorithm compares actual value with MA(<ma_value>) previous values*. MA stands for moving average.
            For optimization purposes, check is done once a <ma_value_iterations>
        * it does that by checking variance of these Values. If variance is equal to 0, loop breaks.

        Args:
            min_iterations (int): minimal number of iterations
            ma_value (int): number of values to compare with actual value
            eps (int): difference between actual value and (moving) average of <ma_value> previous values
            p (float): 0 <= p <= 1. Probability for spinson to be independent
            q_a (int): number of people in the influence group for independent spinson (anticonformity case)
            q_c (int): number of people in the influence group for not independent spinson (conformity case)
            type_of_influence (str): type of choice of the influence group.
            min_iterations (Optional[int]): maximal number of iterations
        Returns:
            (list): magnetization
            (int): number of iterations
            (float): global concentration
        """
        self.initialize_simulation(opinion_init, *args, **kwargs)

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
        concentration = self.calculate_global_concentration()
        return self.operating_magnetization, num_of_iter, concentration

    def calculate_global_concentration(self):
        """ Method calculating global concentration. Positive/all"""
        return len([opinion for opinion in self.operating_opinion.values() if opinion == 1])/len(self.operating_opinion)

    def calculate_magnetization(self):
        """ Method calculating magnetization. """
        return np.mean(list(self.operating_opinion.values()))

    def update_magnetization_list(self):
        """ Method updating magnetization list with current magnetization. """
        self.operating_magnetization.append(self.calculate_magnetization())

    def initialize_simulation(self, opinion_init: str = "disordered_exact_fraction", *args, **kwargs):
        """ Method initializing operating values, i.e. clearing them. """
        # cleaning operating network
        self.reload_operating_network()
        # cleaning operating opinion
        self.reload_operating_opinion(opinion_init, *args, **kwargs)
        # cleaning magnetization
        self.reload_operating_magnetization()

    def monte_carlo_for_given_p(self, prob, mc_steps=10, q_a=4, q_c=10, c_0=1):
        """Method calculating concentration for given p.

        Args:
            prob: probability for given step
            mc_steps (int, optional): number of Monte Carlo iterations. Defaults to 10.

        Returns:
            float: mean value of concentration for given p
        """
        c = []
<<<<<<< HEAD
        with open('c50_4_10_05_ws.txt', 'a') as c_file:  
            for i in range(mc_steps):
                mag, len_mag, concentration = self.simulate_until_stable(min_iterations=1000, max_iterations=100000, ma_value=1000, p=prob, q_a=4, q_c=10, c=0.5)
=======
        with open('c50_5.txt', 'a') as c_file:  
            for i in range(mc_steps):
                _, len_mag, concentration = self.simulate_until_stable(min_iterations=700000, max_iterations=1000000, ma_value=1000, p=prob, q_a=q_a, q_c=q_c, c=c_0)
>>>>>>> 05d4137cb36a6dfbcfe2e71dbaa64abd9911735c
                c.append(concentration)

                c_file.write(str(prob) + ' ' + str(len_mag) + ' ' + str(concentration) + '\n')

                print(prob, len_mag, concentration)

        return np.mean(c)


if __name__ == "__main__":
    # number of nodes
    n = 10000
    # average degree
    k = 150
    # probability. For Erdos Renyi <k> = np
    p = k/n
    # group of anticonformity
    q_a = 13
    q_c = 10
    # initial concentration of opinion
    c_0=0.99

    network = nk.generators.WattsStrogatzGenerator(n, k/2, 0.5)
    network = network.generate()

    # as in the article
<<<<<<< HEAD
    probs = np.linspace(0, 0.21, 72)
=======
    probs = np.linspace(0.85, 0.755, 24)
>>>>>>> 05d4137cb36a6dfbcfe2e71dbaa64abd9911735c

    q_voter = QVoter(network)

    r = Parallel(n_jobs=8, verbose=10)(delayed(q_voter.monte_carlo_for_given_p)(p, q_a=q_a, q_c=q_c, c_0=c_0) for p in probs)

    # """ simple check of methods."""
    # n = 100000
    # p = 150/n
    # network = nx.fast_gnp_random_graph(n, p)

    # mc_steps = 10
    # ps = np.linspace(0, 1, 25)

    # q_voter = QVoter(network)

    # final_concentrations = []
    # final_magnetizations = []

    # for prob in ps:
    #     print(prob)
    #     c = []
    #     m = []
    #     for i in range(mc_steps):
    #         print(i)
    #         mag, len_mag, concentration = q_voter.simulate_until_stable(min_iterations=1000, 
    #                              max_iterations=1000000, ma_value=1000, 
    #                               p=prob, q_a=3, q_c=4, c=1)
    #         c.append(concentration)
    #         m.append(mag)

    #     print(np.mean(c))
    #     print(np.mean(m))

    #     final_concentrations.append(np.mean(c))
    #     final_magnetizations.append(np.mean(m))

    # final_concentration_array = np.array(final_concentrations)
    # final_magnetizations_array = np.array(final_magnetizations)

    # np.savetxt('concentration.txt', final_concentration_array)
    # np.savetxt('magnetization.txt', final_magnetizations)

    # plt.scatter(ps, final_concentrations)
    # plt.xlabel("p")
    # plt.ylabel("C(s, t)")
    # plt.title("C_0 = 0")
    # plt.show()

    # print(f'len_mag = {len_mag}, concentration = {concentration}')
    # plt.scatter(np.linspace(1, len_mag, len_mag), mag, s=10)
    # plt.xlabel('number of iteration')
    # plt.ylabel('magnetization')
    # plt.show()