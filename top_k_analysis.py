import numpy as np
from pandas import DataFrame

input_path = 'data/lstm-label.txt'
attributes = ['floral', 'striped', 'plaid', 'leopard', 'camo', 'graphic',
              'crew neck', 'square neck', 'v-neck',
              'maxi dress', 'midi dress', 'mini dress',
              'denim', 'knit', 'faux leather', 'cotton', 'chiffon', 'satin',
              'mesh', 'ruched', 'cutout', 'lace', 'frayed', 'wrap',
              'tropical', 'peasant', 'swim', 'bikini', 'active', 'cargo']


class TopKGenerator():
    def __init__(self, input_path, attributes):
        super().__init__()
        self.data = None
        self.K = 10
        self.load_data(input_path, attributes)

    def load_data(self, input_path, attributes):
        raw_data = []
        date_list = []
        with open(input_path, 'r') as f:
            lines = f.readlines()[:120]
            for line in lines:
                date_list.append(line.rstrip().split(': ')[0])
                attrs_num = line.rstrip().split(': ')[1].split(' ')
                atrrs_num = [int(x) for x in attrs_num]
                raw_data.append(attrs_num)
        all_data = np.array(raw_data, dtype='float32')
        all_data = all_data.T
        self.data = DataFrame(all_data, index=attributes, columns=date_list)

    def generate_top_k(self, ):
        pass


top_k_generator = TopKGenerator(input_path, attributes)
