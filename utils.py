import math
from data_parser import data_pt
import numpy as np
from sklearn.cluster import KMeans


def calc_kmeans_obj(client_list, center_list, k):
    assignment = dict()
    for client in client_list:
        min_dis = 2**200
        min_j = 0
        for center in center_list:
            if dis(client, center) < min_dis:
                min_dis = dis(client, center)
                min_j = center
        assignment[client] = min_j

    ans = 0
    for client in client_list:
        dist = dis(client, assignment[client]) ** 2
        ans += dist
    return ans


def kmeansinitialization(client_list, k):
    kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=1).fit(np.array(client_list))
    kmeans_center = []
    for center in kmeans.cluster_centers_:
        kmeans_center.append(data_pt(center, 'kmeans'))
    return kmeans_center, kmeans.cluster_centers_


def calc_rho(client_list, center_list, k):
    """
    Calculate the smallest beta value in the solution (figuratively speaking, the maximum distance any
    blocking coalition is allowed to deviate).
    
    Rho is defined as:
        For all clients in a blocking coalition (if it exists), rho * new_dist < old_dist.
            where  new_dist is distance from the client to the potential new center,
            and    old_dist is original distance from the client to the nearest center in center_list.

    Note: client_list is a list of actual clients; center_list is a list of indexes referencing to 
        client_list.
    """
    num = len(client_list)

    # Compute current assignments (nearest center for each client)
    assignment = dict()
    dist = np.zeros((len(client_list), len(center_list)))
    for i, client in enumerate(client_list):
        min = 2**100
        for j, center in enumerate(center_list):
            dist[client][center] = dis(client, center)
            if dist[i,j] < min:
                min = dist[i, j]
                mincenter = j
        assignment[client] = mincenter
    max_rho = 1.0
    for j, potential_center in enumerate(center_list):
        # For each potential center, find its rho
        # Get a list of betas for all clients
        rho_list = []
        for i, client in enumerate(client_list):
            if dist[i][assignment[client]] <= 0 or dist[i][j] <= 0: # old_dist is already 0, can't improve
                continue
            ratio = float(dist[i][assignment[client]]) / dist[i][j]
            rho_list.append(ratio)

        if (len(rho_list) < num / k): # Insufficient number of deviating clients
            continue

        # Calculate smallest beta - formed by the first (alpha * num / k) clients with smallest beta
        rho_list.sort(reverse=True)
        rho = rho_list[int(math.ceil(num / k)) - 1]
        max_rho = rho if rho > max_rho else max_rho
    return max_rho


def calc_rho_proportionality(all_clients, k_centers, k, audit_centers=[]):
    """
    Calculate Rho-Proportionality of a clustering
    :param all_clients: the list of all clients
    :param k_centers: the list of centers picked
    :param k: the number of centers picked
    :param audit_centers: the list of all available centers
    :return: Rho-Proportionality of k_centers
    """
    num = len(all_clients)
    # Compute the nearest center in k_centers for each client
    assignment = {}
    for client in all_clients:
        min_dis = dis(k_centers[0], client)
        min_center = k_centers[0]
        for center in k_centers:
            if dis(center, client) < min_dis:
                min_dis = dis(center, client)
                min_center = center
        assignment[client] = min_center

    max_rho = 1.0
    for potential_center in audit_centers:
        # Get a list of betas for all clients
        rho_list = []
        for client in all_clients:
            if dis(client, potential_center) <= 0: # old_dist is already 0, can't improve
                continue
            ratio = float(dis(client, assignment[client])) / dis(client, potential_center)
            rho_list.append(ratio)
        assert (len(rho_list) >= num / k)
        # Calculate smallest beta - formed by the first (alpha * num / k) clients with smallest beta
        rho_list.sort(reverse=True)
        rho = rho_list[int(math.ceil(num / k)) - 1]
        max_rho = rho if rho > max_rho else max_rho
    return max_rho


def calc_distances(client_list):
    """
    Calculate the distance between every pair of points.
    :param client_list: list of clients
    :return: nested dict giving distances between every pair of points
    """
    distances = {}
    for x in client_list:
        distances[x] = {}
        for y in client_list:
            distances[x][y] = dis(x, y)
    return distances


def dis(pt1, pt2):
    sum = 0.0
    for i in range(pt1.dim):
        a = float(pt1.data[i])
        b = float(pt2.data[i])
        sum = float(sum + (a - b) * (a - b))
    return math.sqrt(sum)


def add(pt1, pt2):
    sum = 0
    temp = [0] * (pt1.dim + 1)
    for i in range(pt1.dim + 1):
        a = float(pt1.raw_data[i])
        b = float(pt2.raw_data[i])
        temp[i] = a + b
    return data_pt(temp, 'raw')


def div(pt1, ratio):
    temp = [0] * (pt1.dim + 1)
    for i in range(pt1.dim):
        temp[i] = pt1.raw_data[i]/ratio
    return data_pt(temp, 'raw')


def printData(output_file, kmeans_obj, rho, label = "Default", EOL = False):
    '''
    Output the experiment data to file
    :param output_file: The name of the file it outputs to
    :param kmeans_obj:
    :param rho:
    :param label:
    :param EOL: Whether this is the end of a line
    '''
    print ("%s Algorithm finishes with KMeans Objective %f, and Rho Fairness Measure %f" % (label, kmeans_obj, rho))
    if EOL:
        f = open(output_file, "a")
        f.write(str(kmeans_obj) + " " + str(rho) + "\n")
        f.close()
    else:
        f = open(output_file, "a")
        f.write(str(kmeans_obj) + " " + str(rho) + " ")
        f.close()
