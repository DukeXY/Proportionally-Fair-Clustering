import random
import numpy as np

def represent_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

class data_pt:
    def __init__(self, content, flag):
        if flag == 'raw':
            self.data = content[: -1]
            self.dim = len(content) - 1
            self.result = content[-1]
            self.raw_data = content
        if flag == 'kmeans':
            self.data = content
            self.dim = len(content)
            self.raw_data = content


    def __str__(self):
        return ','.join(self.raw_data)


def error_check(data):
    for entry in data:
        if entry == '?':
            return False
    return True

def convert_to_num(line_data, convert_map):
    # Input: convert_map[i] - list of dict, each dict is for one single coordinate
    # Output: modified_data - list of float, each coordinate converts to float
    modified_data = list()
    for i in range(len(line_data)):
        if represent_float(line_data[i]):
            modified_data.append(float(line_data[i]))
            continue
        if not line_data[i] in convert_map[i]:
            convert_map[i][line_data[i]] = float(len(convert_map[i]))
        modified_data.append(convert_map[i][line_data[i]])
    return modified_data

def parse_data(file_name, max_item = 100000, scale = False):
    # Parse the data in .csv file into two data formats
    # 1. data_list: list of data_pt
    # 2. kmeans_fmt_data_list: list of coordinates (i.e. a double list
    data_list = []
    convert_map = []
    with open(file_name) as f:
        content = f.readlines()
        kmeans_fmt_data_list = []
        m = len(content[0].split(','))
        for i in range(m):
            convert_map.append(dict())
        # maxv - max value in each coordinate
        # minv - min value in each coordinate
        # maxv and minv are used to scale each dimension
        maxv = np.zeros(m)
        minv = np.zeros(m)
        for content_line in content[:max_item]:
            line_data = content_line.split(',')
            line_data = convert_to_num(line_data, convert_map)
            maxv = np.maximum(maxv, line_data)
            minv = np.minimum(minv, line_data)

        for content_line in content[:max_item]:
            line_data = content_line.split(',')
            line_data = convert_to_num(line_data, convert_map)
            if (scale):
                line_data = [line_data[i]/(maxv[i]-minv[i]) if maxv[i]-minv[i] != 0 else 0 for i in range(len(line_data))]

            if error_check(line_data):
                kmeans_fmt_data_list.append(line_data[:-1])
                # Using 'raw' since last bit is result
                data_list.append(data_pt(line_data, 'raw'))
        print(len(content))
    return data_list, kmeans_fmt_data_list


def random_sample(parsed_data, num):

    s = set()
    data_list = []
    reverse_map = dict()
    positive_map = dict()
    cnt = 0
    while (len(s) < num):
        t = random.randint(0, len(parsed_data) - 1)
        if not parsed_data[t] in s:
            s.add(parsed_data[t])
            data_list.append(parsed_data[t])
            reverse_map[cnt] = t
            positive_map[t] = cnt
            cnt += 1
    return data_list, reverse_map, positive_map



if __name__ == '__main__':
    file_name = 'dataset_1'
    sample_num = 10
    parsed_data = parse_data(file_name)
    print('Succeed')
    sampled_data = random_sample(parsed_data, sample_num)
