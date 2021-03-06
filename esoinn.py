# Copyright (c) 2017 Gangchen Hua
# This software is released under the MIT License.
# http://opensource.org/licenses/mit-license.php

"""
E-SOINN in Python 3
Version 1.0
"""
from __future__ import division 
from typing import overload
import numpy as np
from scipy.sparse import dok_matrix
from sklearn.base import BaseEstimator, ClusterMixin
from random import choice
import matplotlib.pyplot as plt
from numpy import *
import operator


class ESoinn(BaseEstimator, ClusterMixin):

    INITIAL_LABEL = -1

    def __init__(self, dim=2, max_edge_age=20, iteration_threshold=200, c1=0.0001, c2=0.8):
        self.dim = dim
        self.iteration_threshold = iteration_threshold
        self.c1 = c1
        self.c2 = c2
        self.max_edge_age = max_edge_age
        self.num_signal = 0
        self._reset_state()
        # self.fig = plt.figure()

    def _reset_state(self):
        self.nodesByclass = []
        self.nodes = np.array([], dtype=np.float64)
        self.winning_times = []
        self.density = []
        self.N = []
        # if active
        self.won = []
        self.total_loop = 1
        self.s = []
        self.adjacent_mat = dok_matrix((0, 0), dtype=np.float64)
        self.node_labels = []
        self.labels_ = []

    def fitAgain(self, X , rangeNum):
        """
        train data in batch manner
        :param X: array-like or ndarray
        """
        # self._reset_state()
        # choose 50000 signals randomly
        # xtemp = X[1:2400,:]
        for x in range(rangeNum):

            self.input_signal(choice(X))
        # self.labels_ = self.__label_samples(X)
        self.__classify()
        return self

    def fit(self, X, rangeNum):
        """
        train data in batch manner
        :param X: array-like or ndarray
        """
        self._reset_state()
        # plt.ion()
        # plt.xlim(-5, 30)
        # plt.ylim(-5, 20)
        for x in range(rangeNum):
            self.input_signal(choice(X))
            
        # self.labels_ = self.__label_samples(X)
        self.__classify()

        return self

    def findNearestWinner(self, signal):
        winner, dists = self.__find_nearest_nodes(2, signal) 
        # print(signal,self.nodes[winner[0]])
        # input('dddddddddd')
        return self.nodes[winner[0]], self.node_labels[winner[0]]


    def input_signal(self, signal: np.ndarray):
        """
        Input a new signal one by one, which means training in online manner.
        fit() calls __init__() before training, which means resetting the
        state. So the function does batch training.
        :param signal: A new input signal
        :return:
        """
        # Algorithm 3.4 (2)
        signal = self.__check_signal(signal)
        # plt.plot(signal[0], signal[1], '.')
        # plt.pause(1)
        # plt.ioff()
        self.num_signal += 1

        # Algorithm 3.4 (1)
        if len(self.nodes) < 2:
            self.__add_node(signal)
            return

        # Algorithm 3.4 (3)
        winner, dists = self.__find_nearest_nodes(2, signal)  #??????winner?????????????????????

        sim_thresholds = self.__calculate_similarity_thresholds(winner)

        # print(sim_thresholds)
        # input('shu ru ..............')
        if dists[0] > sim_thresholds[0] or dists[1] > sim_thresholds[1]:
            self.__add_node(signal)
            # plt.plot(signal[0], signal[1], 'ro')
        else:
            # Algorithm 3.4 (4)
            self.__increment_edge_ages(winner[0])   #?????????????????????winner?????????????????????+1
            # Algorithm 3.4 (5)
            need_add_edge, need_combine = self.__need_add_edge(winner)  #??????????????????????????????winner
            if need_add_edge:
                # print("add edge")
                # Algorithm 3.4 (5)(a)
                self.__add_edge(winner)  # ???????????????????????????winner?????????
            else:
                # Algorithm 3.4 (5)(b)
                self.__remove_edge_from_adjacent_mat(winner)   #???????????????????????????????????????
            # Algorithm 3.4 (5)(a) need combine subclasses
            if need_combine:
                self.__combine_subclass(winner)
            # Algorithm 3.4 (6) checked, maybe has problem
            self.__update_density(winner[0])  # ??????winner?????????
            # Algorithm 3.4 (7)(8)
            self.__update_winner(winner[0], signal) # add 1 to a local accumulated number of signals Ma1
            # Algorithm 3.4 (8)
            self.__update_adjacent_nodes(winner[0], signal)  #??????winner???weight vectors

        # Algorithm 3.4 (9)
        self.__remove_old_edges()  #

        # Algorithm 3.4 (10)
        if self.num_signal % self.iteration_threshold == 0:
            for i in range(len(self.won)):
                if self.won[i]:
                    self.N[i] += 1
            print("Input signal amount:", self.num_signal, "nodes amount", len(self.nodes))
            self.__separate_subclass()
            self.__delete_noise_nodes()
            self.total_loop += 1

    # checked
    def __combine_subclass(self, winner):
        if self.node_labels[winner[0]] == self.node_labels[winner[1]]:
            raise ValueError
        class_id = self.node_labels[winner[0]]
        node_belong_to_class_1 = self.find_all_index(self.node_labels, self.node_labels[winner[1]])
        for i in node_belong_to_class_1:
            self.node_labels[i] = class_id

    # checked
    def __remove_old_edges(self):
        for i in list(self.adjacent_mat.keys()):
            if self.adjacent_mat[i] > self.max_edge_age + 1:
                # print("Edge removed")
                self.adjacent_mat.pop((i[0], i[1]))

    # checked
    def __remove_edge_from_adjacent_mat(self, ids):
        if (ids[0], ids[1]) in self.adjacent_mat and (ids[1], ids[0]) in self.adjacent_mat:
            self.adjacent_mat.pop((ids[0], ids[1]))
            self.adjacent_mat.pop((ids[1], ids[0]))

    def __separate_subclass(self):
        # find all local apex
        density_dict = {}
        density = list(self.density)
        for i in range(len(self.density)):
            density_dict[i] = density[i]
        class_id = 0
        while len(density_dict) > 0:
            apex = max(density_dict, key=lambda x: density_dict[x])
            # print("len", len(density_dict))
            # print("apex", apex)
            ids = []
            ids.append(apex)
            self.__get_nodes_by_apex(apex, ids, density_dict)
            for i in set(ids):
                if i not in density_dict:
                    raise ValueError
                self.node_labels[i] = class_id
                density_dict.pop(i)
            class_id += 1

        # print("class_id", class_id)

    def __get_nodes_by_apex(self, apex, ids, density_dict):
        new_ids = []
        pals = self.adjacent_mat[apex]
        for k in pals.keys():
            i = k[1]
            if self.density[i] <= self.density[apex] and i in density_dict and i not in ids:
                ids.append(i)
                new_ids.append(i)
        if len(new_ids) != 0:
            for i in new_ids:
                self.__get_nodes_by_apex(i, ids, density_dict)
        else:
            return

    # Algorithm 3.2, checked
    def __need_add_edge(self, winner):
        if self.node_labels[winner[0]] == self.INITIAL_LABEL or \
                        self.node_labels[winner[1]] == self.INITIAL_LABEL:  #?????????winner?????????????????????
            return True, False  # ?????????????????????????????? ???????????????????????????
        elif self.node_labels[winner[0]] == self.node_labels[winner[1]]: #??????winner???????????????
            return True, False  # ??????
        else:  # ??????winner???????????????????????????7,8???
            mean_density_0, max_density_0 = self.__mean_max_density(self.node_labels[winner[0]])
            mean_density_1, max_density_1 = self.__mean_max_density(self.node_labels[winner[1]])
            alpha_0 = self.calculate_alpha(mean_density_0, max_density_0)
            alpha_1 = self.calculate_alpha(mean_density_1, max_density_1)
            min_density = min([self.density[winner[0]], self.density[winner[1]]])
            # print(self.density[winner[0]], self.density[winner[1]])
            # print(mean_density_0, max_density_0, mean_density_1, max_density_1, alpha_0, alpha_1, min_density)
            if alpha_0 * max_density_0 < min_density or alpha_1 * max_density_1 < min_density:  # (7),(8)
                # print("True")
                return True, True  # ??????
            else:
                return False, False  # ????????????

    @staticmethod
    def calculate_alpha(mean_density, max_density):
        if max_density > 3.0 * mean_density:
            return 1.0
        elif 2.0 * mean_density < max_density <= 3.0 * mean_density:
            return 0.5
        else:
            return 0.0

    @staticmethod
    def find_all_index(ob, item):
        return [i for i, a in enumerate(ob) if a == item]

    # checked
    def __mean_max_density(self, class_id):
        node_belong_to_class = self.find_all_index(self.node_labels, class_id)
        avg_density = 0.0
        max_density = 0.0
        for i in node_belong_to_class:
            avg_density += self.density[i]
            if self.density[i] > max_density:
                max_density = self.density[i]
        avg_density /= len(node_belong_to_class)
        return avg_density, max_density

    @overload
    def __check_signal(self, signal: list) -> None:
        ...

    def __check_signal(self, signal: np.ndarray):
        """
        check type and dimensionality of an input signal.
        If signal is the first input signal, set the dimension of it as
        self.dim. So, this method have to be called before calling functions
        that use self.dim.
        :param signal: an input signal
        """
        if isinstance(signal, list):
            signal = np.array(signal)
        if not (isinstance(signal, np.ndarray)):
            print("1")
            raise TypeError()
        if len(signal.shape) != 1:
            print("2")
            raise TypeError()
        self.dim = signal.shape[0]
        if not (hasattr(self, 'dim')):
            self.dim = signal.shape[0]
        else:
            if signal.shape[0] != self.dim:
                print("3")
                raise TypeError()
        return signal

    # checked
    def __add_node(self, signal: np.ndarray):
        n = self.nodes.shape[0]
        self.nodes.resize((n + 1, self.dim), refcheck=False)
        self.nodes[-1, :] = signal
        self.winning_times.append(1)
        self.adjacent_mat.resize((n + 1, n + 1))  # ??
        self.N.append(1)
        self.density.append(0)
        self.s.append(0)
        self.won.append(False)
        self.node_labels.append(self.INITIAL_LABEL)

    # checked
    def __find_nearest_nodes(self, num: int, signal: np.ndarray):
        n = self.nodes.shape[0]  #self.nodes ???????????????
        indexes = [0] * num

        sq_dists = [0.0] * num

        D = np.sum((self.nodes - np.array([signal] * n)) ** 2, 1)
        # print("D", D)
        for i in range(num):
            indexes[i] = np.nanargmin(D)
            sq_dists[i] = D[indexes[i]]
            D[indexes[i]] = float('nan')
        return indexes, sq_dists

    # checked
    def __calculate_similarity_thresholds(self, node_indexes):
        sim_thresholds = []
        for i in node_indexes:
            pals = self.adjacent_mat[i, :]
            if len(pals) == 0: #???neighbour
                # ????????????????????????????????????
                idx, sq_dists = self.__find_nearest_nodes(2, self.nodes[i, :])
                sim_thresholds.append(sq_dists[1])
            else:  #???neighbour
                pal_indexes = []  
                for k in pals.keys():
                    pal_indexes.append(k[1])
                sq_dists = np.sum((self.nodes[pal_indexes] - np.array([self.nodes[i]] * len(pal_indexes))) ** 2, 1)
                sim_thresholds.append(np.max(sq_dists))
        return sim_thresholds

    # checked
    def __add_edge(self, node_indexes): 
        self.__set_edge_weight(node_indexes, 1)

    # checked
    def __increment_edge_ages(self, winner_index):
        for k, v in self.adjacent_mat[winner_index, :].items():
            self.__set_edge_weight((winner_index, k[1]), v + 1)  #k[1] ??????????????????????????? v+1???age

    # checked
    def __set_edge_weight(self, index, weight):
        self.adjacent_mat[index[0], index[1]] = weight
        self.adjacent_mat[index[1], index[0]] = weight

    # checked
    def __update_winner(self, winner_index, signal):
        w = self.nodes[winner_index]
        self.nodes[winner_index] = w + (signal - w) / self.winning_times[winner_index]

    # checked, maybe has problem
    def __update_density(self, winner_index):
        self.winning_times[winner_index] += 1
        if self.N[winner_index] == 0:
            raise ValueError
        # print(self.N[winner_index])
        pals = self.adjacent_mat[winner_index]
        pal_indexes = []
        for k in pals.keys():
            pal_indexes.append(k[1])
        if len(pal_indexes) != 0:
            # print(len(pal_indexes))
            sq_dists = np.sum((self.nodes[pal_indexes] - np.array([self.nodes[winner_index]]*len(pal_indexes)))**2, 1)
            # print(sq_dists)
            mean_adjacent_density = np.mean(np.sqrt(sq_dists))
            p = 1.0/((1.0 + mean_adjacent_density) ** 2)
            self.s[winner_index] += p
            self.density[winner_index] = self.s[winner_index]/self.total_loop

        if self.s[winner_index] > 0:
            self.won[winner_index] = True

    # checked
    def __update_adjacent_nodes(self, winner_index, signal):
        pals = self.adjacent_mat[winner_index]
        for k in pals.keys():
            i = k[1]
            w = self.nodes[i]
            self.nodes[i] = w + (signal - w) / (100 * self.winning_times[i])

    # checked
    def __delete_nodes(self, indexes):
        if not indexes:
            return
        n = len(self.winning_times)
        self.nodes = np.delete(self.nodes, indexes, 0)
        remained_indexes = list(set([i for i in range(n)]) - set(indexes))
        self.winning_times = [self.winning_times[i] for i in remained_indexes]
        self.N = [self.N[i] for i in remained_indexes]
        self.density = [self.density[i] for i in remained_indexes]
        self.node_labels = [self.node_labels[i] for i in remained_indexes]
        self.won = [self.won[i] for i in remained_indexes]
        self.s = [self.s[i] for i in remained_indexes]
        self.__delete_nodes_from_adjacent_mat(indexes, n, len(remained_indexes))

    # checked
    def __delete_nodes_from_adjacent_mat(self, indexes, prev_n, next_n):
        while indexes:
            next_adjacent_mat = dok_matrix((prev_n, prev_n))
            for key1, key2 in self.adjacent_mat.keys():
                if key1 == indexes[0] or key2 == indexes[0]:
                    continue
                if key1 > indexes[0]:
                    new_key1 = key1 - 1
                else:
                    new_key1 = key1
                if key2 > indexes[0]:
                    new_key2 = key2 - 1
                else:
                    new_key2 = key2
                # Because dok_matrix.__getitem__ is slow,
                # access as dictionary.
                next_adjacent_mat[new_key1, new_key2] = super(dok_matrix, self.adjacent_mat).__getitem__((key1, key2))
            self.adjacent_mat = next_adjacent_mat.copy()
            indexes = [i - 1 for i in indexes]
            indexes.pop(0)
        self.adjacent_mat.resize((next_n, next_n))

    # checked
    def __delete_noise_nodes(self):
        n = len(self.winning_times)
        # print(n)
        noise_indexes = []
        mean_density_all = np.mean(self.density)
        # print(mean_density_all)
        for i in range(n):
            if len(self.adjacent_mat[i, :]) == 2 and self.density[i] < self.c1 * mean_density_all:
                noise_indexes.append(i)
            elif len(self.adjacent_mat[i, :]) == 1 and self.density[i] < self.c2 * mean_density_all:
                noise_indexes.append(i)
            elif len(self.adjacent_mat[i, :]) == 0:
                noise_indexes.append(i)
        print("Removed noise node:", len(noise_indexes))
        self.__delete_nodes(noise_indexes)

    def __get_connected_node(self, index, indexes):
        new_ids = []
        pals = self.adjacent_mat[index]
        for k in pals.keys():
            i = k[1]
            if i not in indexes:
                indexes.append(i)
                new_ids.append(i)

        if len(new_ids) != 0:
            for i in new_ids:
                self.__get_connected_node(i, indexes)
        else:
            return

    # Algorithm 3.3
    def __classify(self):
        need_classified = list(range(len(self.node_labels)))
        for i in range(len(self.node_labels)):
            self.node_labels[i] = self.INITIAL_LABEL
        class_id = 0
        self.nodesByclass = []
        self.classCenter = []
# pie = []
# a = np.empty(shape=[3])

# pie.append(a)
# pie[0] = np.append(pie[0], [1,2,3], axis=0)
# # print(pie[0].append([1,2,3]))
# a = np.empty(shape=[0, 3])
# pie.append(a)
# print(len(pie))
        while len(need_classified) > 0:
            oneClassNodes = []
            indexes = []
            index = choice(need_classified)
            indexes.append(index)
            self.__get_connected_node(index, indexes)
            for i in indexes:
                self.node_labels[i] = class_id
                oneClassNodes.append(self.nodes[i])
                need_classified.remove(i)
            class_id += 1
            self.nodesByclass.append(oneClassNodes)

        for i in range(class_id):
            
            for j in range(len(self.nodesByclass[i])):
                if j==0:
                    a = self.nodesByclass[i][j]
                else:
                    a = list(map(operator.add, a, self.nodesByclass[i][j]))
                # print(self.nodesByclass[i][j])

            averageNum =  np.array(a)
            for j in range(len(a)):
                bb = np.float64(len(self.nodesByclass[i]))
                averageNum[j] =  a[j] / bb
                # print(a[j])
                # print(len(self.nodesByclass[i]))
                # print(averageNum[j])
                # input(3333334444444)

            
            self.classCenter.append(averageNum)

        # print(class_id, len(self.classCenter))
        # input(2233333)
        # print(len(self.node_labels))
        # print()
        print("Number of classes???", class_id)


    def findNearestCenter(self, signal):
        n = len(self.classCenter)  #self.nodes ???????????????
        indexes = [0] * 1

        sq_dists = [0.0] * 1

        D = np.sum((np.array(self.classCenter) - np.array([signal] * n)) ** 2, 1)
        indexes = np.nanargmin(D)

        # print(self.classCenter[indexes], indexes)
        # input(33333333333)
        # print(D)
        # input(33333333333)


        # sq_dists[0] = D[indexes[0]]
        # D[indexes[0]] = float('nan')

        # winner, dists = self.__find_nearest_nodes(2, signal) 
        # print(signal,self.nodes[winner[0]])
        # input('dddddddddd')
        return indexes, indexes