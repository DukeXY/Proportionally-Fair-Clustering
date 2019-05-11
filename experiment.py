from local_search_capture import *
from ball_growing import *
from data_parser import parse_data, random_sample
from utils import *
from sklearn.cluster import KMeans
from utils import calc_rho_proportionality
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', action="store_true", dest = "sample_type", default = False)
    parser.add_argument('--sample_clients', action="store", dest="sample_num", type=int, default=5000)
    parser.add_argument('--sample_centers', action="store", dest="center_num", type=int, default=400)
    parser.add_argument('--file_name', action="store", dest = "file_name", default = 'iris')
    parser.add_argument('--rho', action="store", type=float, default=1.01)

    option = parser.parse_args()
    file_name = "./data/" + option.file_name
    output_file = "./result/" + option.file_name + "_result.txt"
    rho_value = option.rho

    min_k = 2
    max_k = 10
    k_step = 1
    sample_num = option.sample_num
    center_num = option.center_num
    sample_type = "Random" if option.sample_type else "Full" # Choose from Full, Random
    tot_exp = 1  # Number of Experiments ran for each k

    print('Working on %s, Randomly select %d samples, k from %d to %d' % (file_name, sample_num, min_k, max_k))
    parsed_data, kmeans_parsed_data = parse_data(file_name)
    dim = len(kmeans_parsed_data[0])

    # Assertion makes sure we get identical copy of data in two formats
    assert (len(kmeans_parsed_data[0]) == parsed_data[0].dim)
    assert (len(parsed_data) == len(kmeans_parsed_data))
    print('Succeed in Parsing Data')

    if sample_type == "Random":
        all_clients, reverse_map, _ = random_sample(parsed_data, sample_num)
        print('Succeed in Sampling %d Clients' % len(all_clients))
        all_centers, original_centers = kmeansinitialization(kmeans_parsed_data, center_num)
        print('Succeed in Sampling %d Centers' % len(all_centers))
    else:
        all_centers = parsed_data
        all_clients = parsed_data

    for rho in [rho_value]:
        print(rho)
        k_values = range(min_k, max_k + 1, k_step)
        for k in k_values:
            print('k = %d' % k)

            # Find Audit Centers for Rho Proportionality Calculation
            if sample_type == "Random":
                audit_centers, _ = kmeansinitialization(kmeans_parsed_data, center_num)
            else:
                audit_centers = all_centers

            # Local Capture Algorithm Part
            print("Start Local Search")
            local_capture_centers = local_capture(all_clients, k, rho=rho, all_centers=all_centers)
            assert (len(local_capture_centers) == k)
            local_capture_kmeans = calc_kmeans_obj(parsed_data, local_capture_centers, k)
            local_capture_rho = calc_rho_proportionality(all_clients, local_capture_centers, k,
                                                         audit_centers=audit_centers)
            printData(output_file, local_capture_kmeans, local_capture_rho, label="Local Search")

            # KMeans++ Algorithm Part
            print("Start Kmeans++")
            kmeans = KMeans(n_clusters=k, random_state=0, init='k-means++').fit(np.array(kmeans_parsed_data))
            kmeans_centers = []
            for center in kmeans.cluster_centers_:
                kmeans_centers.append(data_pt(center, 'kmeans'))
            kmeansobj_kmeans = calc_kmeans_obj(parsed_data, kmeans_centers, k)
            assert (abs(kmeansobj_kmeans - kmeans.inertia_) < 1000)
            kmeans_rho = calc_rho_proportionality(all_clients, kmeans_centers, k, audit_centers=audit_centers)
            printData(output_file, kmeans.inertia_, kmeans_rho, label="KMeans++")

            if sample_type == "Random":
                # KMeans Heuristic Part
                print("Start Center Reduction Heuristic")
                remain_centers = local_capture_centers + kmeans_centers
                k2_kmeans = calc_kmeans_obj(parsed_data, remain_centers, k)
                k2_rho = calc_rho_proportionality(all_clients, remain_centers, k, audit_centers=audit_centers)
                printData(output_file, k2_kmeans, k2_rho, label="2KCenters")

                # Reduce as many centers as possible
                flag = True
                while (len(remain_centers) > k and flag):
                    flag = False
                    for next_close_center in remain_centers:
                        remain_centers.remove(next_close_center)
                        temp_kmeans = calc_kmeans_obj(parsed_data, remain_centers, k)
                        temp_rho = calc_rho_proportionality(all_clients, remain_centers, k, audit_centers=audit_centers)
                        if (temp_rho <= 1.2 * local_capture_rho and temp_kmeans <= 1.5 * kmeansobj_kmeans):
                            flag = True
                            break
                        remain_centers.append(next_close_center)

                remain_centers_kmeans = calc_kmeans_obj(parsed_data, remain_centers, k)
                remain_centers_rho = calc_rho_proportionality(all_clients, remain_centers, k,
                                                              audit_centers=audit_centers)
                print("Hybrid Heuristic finish with %d centers, Kmeans Objective %d, Rho Objective %f" % (
                len(remain_centers), remain_centers_kmeans, remain_centers_rho))
                f = open(output_file, "a")
                f.write(
                    str(remain_centers_kmeans) + " " + str(remain_centers_rho) + " " + str(len(remain_centers)) + "\n")
                f.close()
            else:
                # Greedy Capture Algorithm Part
                print("Start Greedy Capture Algorithm")
                greedy_center = ball_growing_repeated(all_clients, k, alpha=1, distances=None)
                assert (len(greedy_center) == k)
                kmeansobj_greedy = calc_kmeans_obj(parsed_data, greedy_center, k)
                kmeans_rho_greedy = calc_rho_proportionality(all_clients, greedy_center, k, audit_centers=audit_centers)
                printData(output_file, kmeansobj_greedy, kmeans_rho_greedy, label="Greedy Ball Growing", EOL=True)
