import random

from utils import *
from data import ExperimentData
from utils import dis
import numpy as np

def kmeanscenter(center_list, k):
    cnt = 0
    select = [np.random.randint(len(center_list)),]
    n = len(center_list)
    prob = np.zeros(n)
    while (len(select) < k):
        totdis = 0
        for i, center in enumerate(center_list):
            mindis = 2**100
            mincenter = 0
            for j in select:
                if dis(center, center_list[j]) < mindis:
                    mindis = dis(center, center_list[j])
                    mincenter= j
            prob[i] = mindis
            totdis += mindis
        prob = prob / totdis
        cnt += 1
        choice = np.random.choice(n, 1, p=prob)
        select.append(choice[0])
    return select


def local_capture(client_list, k, rho=1.0, max_iter=100, all_centers = []):
    '''
    This function runs the Local Capture Algorithm.
    :param client_list: All available clients
    :param k: Number of centers we want to find
    :param rho: Rho Proportionality Measure used for local search
    :param max_iter: Maximum number of iterations for local_capture
    :param all_centers: All available centers to choose
    :return: a list of k centers chosen by Local Capture Algorithm
    '''
    dist = np.zeros((len(client_list), len(all_centers)))
    for i, client in enumerate(client_list):
        for j, center in enumerate(all_centers):
            dist[i, j] = dis(client, center)

    num = len(client_list)
    flag = True
    temp = kmeanscenter(all_centers, k)
    iter_cnt = 0
    while flag and iter_cnt < max_iter:
        iter_cnt += 1
        flag = False
        # Find each client's closest center
        assignment = dict()
        for i, client in enumerate(client_list):
            min = 2**30
            for center in temp:
                if (dist[i][center] < min):
                    min = dist[i][center]
                    mincenter = center
            assignment[client] = mincenter
        # Find the number of clients each center attracts
        client_num = dict()
        for center in temp:
            client_num[center] = 0
        for client in client_list:
            client_num[assignment[client]] += 1

        next_close_center_client = 2**100
        next_close_center_index = 0

        # Find the center attracting the least number of clients
        for i, center_index in enumerate(temp):
            if (client_num[center_index] < next_close_center_client):
                next_close_center_client = client_num[center_index]
                next_close_center_index = i

        # Find a new center attracts at least n/k clients in the current setting
        for j, potential_center in enumerate(all_centers):
            cnt = 0
            for i, client in enumerate(client_list):
                if rho * dist[i][j] <= dist[i][assignment[client]]:
                    cnt += 1
            if (cnt >= num / k and  j not in set(temp)):
                temp[next_close_center_index] = j
                flag = True
                break

    if iter_cnt == max_iter:
        print('Did not converge %d' % (rho))

    # Calculate KMeans Objective
    kmeansobj = 0
    for i, client in enumerate(client_list):
        min = 2 ** 30
        for center in temp:
            if (dist[i][center] < min):
                min = dist[i][center]
                mincenter = center
        kmeansobj += dist[i, mincenter]
    center_result = []
    for i, center_index in enumerate(temp):
        center_result.append(all_centers[center_index])
    return center_result
