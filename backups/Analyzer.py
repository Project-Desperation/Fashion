import os
# from datetime import timedelta
from dateutil import parser
from dateutil.relativedelta import relativedelta
import numpy as np
from pandas import DataFrame, Series

input_path = 'data/lstm-label.txt'
attributes = ['floral', 'striped', 'plaid', 'leopard', 'camo', 'graphic',
              'crew neck', 'square neck', 'v-neck',
              'maxi dress', 'midi dress', 'mini dress',
              'denim', 'knit', 'faux leather', 'cotton', 'chiffon', 'satin',
              'mesh', 'ruched', 'cutout', 'lace', 'frayed', 'wrap',
              'tropical', 'peasant', 'swim', 'bikini', 'active', 'cargo']

data_path = 'data'


# ----------------------------------------------------------------------------------------------------------------------
# newly released analysis
class Analyzer:
    def __init__(self, data_path, attributes):
        super().__init__()
        self.data_path = data_path
        self.attributes = attributes

    def get_newly_released(self, start_date, mode, step):
        """
        获取某个时间段内新发售商品的数据
        """
        if mode == 'daily':
            start_date = parser.parse(start_date)
            end_date = start_date + relativedelta(days=step)
        elif mode == 'monthly':
            start_date = parser.parse(start_date[:-2] + '01')
            end_date = start_date + relativedelta(months=step)
        elif mode == 'seasonly':
            start_date = start_date.split('-')
            start_date[2] = "01"
            start_date[1] = "{0:02d}".format(int((int(start_date[1]) - 1) / 3) * 3 + 1)
            start_date = parser.parse('-'.join(start_date))
            end_date = start_date + relativedelta(months=step * 3)
        else:
            raise ValueError("_(:3 」∠)_")

        date_list = []
        today = start_date
        while today <= end_date:
            date_list.append(today.strftime('%Y-%m-%d'))
            today += relativedelta(days=1)

        with open(os.path.join(self.data_path, 'first_appearance.txt'), 'r') as f:
            first_appearance = Series(eval(f.read()))

        newly_released_attr_dic = {}
        for attr in attributes:
            newly_released_attr_dic[attr] = 0

        for date in date_list:
            pids = first_appearance[first_appearance.values == date].index
            if not os.path.exists(os.path.join(self.data_path, f'text/{date}')):
                raise ValueError("Missing data of {}".format(date))
            for root, dirs, files in os.walk(os.path.join(self.data_path, f'text/{date}')):
                for filename in files:
                    if filename[:-4] in pids:
                        with open(os.path.join(root, filename), 'r', encoding='utf8') as f:
                            detail = eval(f.read())

                        description = ''
                        for key in ['name', 'details', 'Details']:
                            if key in detail.keys():
                                description = ' '.join([description, str(detail[key])])

                        for attr in self.attributes:
                            if attr in description:
                                newly_released_attr_dic[attr] += 1

        return newly_released_attr_dic


# ----------------------------------------------------------------------------------------------------------------------
# class TopKGenerator():
#     def __init__(self, input_path, attributes):
#         super().__init__()
#         self.data = None
#         self.K = 10
#         self.load_data(input_path, attributes)
#
#     def load_data(self, input_path, attributes):
#         raw_data = []
#         date_list = []
#         with open(input_path, 'r') as f:
#             lines = f.readlines()[:120]
#             for line in lines:
#                 date_list.append(line.rstrip().split(': ')[0])
#                 attrs_num = line.rstrip().split(': ')[1].split(' ')
#                 atrrs_num = [int(x) for x in attrs_num]
#                 raw_data.append(attrs_num)
#         all_data = np.array(raw_data, dtype='float32')
#         all_data = all_data.T
#         self.data = DataFrame(all_data, index=attributes, columns=date_list)
#
#     def generate_top_k(self):
#         pass


# ----------------------------------------------------------------------------------------------------------------------
# top_k_generator = TopKGenerator(input_path, attributes)
self = Analyzer(data_path, attributes)
test = self.get_newly_released('2021-05-21', 'daily', 0)
print(test)
