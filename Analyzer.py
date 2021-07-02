import os
# from datetime import timedelta
import pymongo
import datetime
from dateutil import parser
from dateutil.relativedelta import relativedelta
import numpy as np
from pandas import DataFrame, Series


# ----------------------------------------------------------------------------------------------------------------------
# newly released analysis
# 202.120.37.116
class Analyzer:
    def __init__(self, data_path, attributes):
        super().__init__()
        self.db = pymongo.MongoClient("mongodb://qyli:qyli2233@202.120.37.116:2233/admin")
        self.data_path = data_path
        self.attributes = attributes

    @staticmethod
    def get_lstm_data(attributes):
        db = pymongo.MongoClient("mongodb://qyli:qyli2233@202.120.37.116:2233/admin")
        lstm_labels = db.fashion.lstm_labels
        attr_selector = {}
        for attribute in attributes:
            attr_selector[attribute] = 1
        attr_selector['_id'] = 0
        lstm_labels = lstm_labels.find({}, attr_selector)
        raw_data = []
        for i in lstm_labels:
            item = Series(i).loc[attributes]
            raw_data.append(item.values)
        all_data = np.array(raw_data, dtype='float32').T
        return all_data

        # raw_data = []
        # with open(input_path, 'r') as f:
        #     lines = f.readlines()[:120]
        #     for line in lines:
        #         attrs_num = line.rstrip().split(': ')[1].split(' ')
        #         atrrs_num = [int(x) for x in attrs_num]
        #         raw_data.append(attrs_num)
        # all_data = np.array(raw_data, dtype='float32')
        # all_data = all_data.T

    @staticmethod
    def get_date_list(start_date, mode='daily', step=1):
        if mode == 'daily':
            if step >= 1:
                step -= 1
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

        return start_date, end_date

    def attribute_counter(self, goods):
        count_result = {}
        for attr in self.attributes:
            count_result[attr] = 0

        for detail in goods:

            description = ''
            for key in ['name', 'Content + Care', 'Details']:
                if key in detail.keys():
                    description = ' '.join([description, str(detail[key])])

            for attr in self.attributes:
                if attr in description:
                    count_result[attr] += 1

        return count_result

    def get_newly_released(self, start_date, mode='daily', step=1):
        """
        获取某个时间段内新发售商品的数据
        """
        start_date, end_date = self.get_date_list(start_date, mode, step)

        goods_info = self.db.fashion.goods_info
        goods = goods_info.find({'first_appearance': {'$gte': start_date, '$lte': end_date}})

        newly_released_attr_dic = self.attribute_counter(goods)

        return newly_released_attr_dic

    def get_newly_released_relative(self, start_date, mode='monthly', step=1):
        date_list = self.get_date_list(start_date, mode, step)
        newly_released_attr_dic = self.get_newly_released(date_list)

        goods = set()
        for date in date_list:
            if not os.path.exists(os.path.join(self.data_path, f'text/{date}')):
                raise ValueError('Missing {}'.format(os.path.join(self.data_path, f'text/{date}')))
            for root, dirs, files in os.walk(os.path.join(self.data_path, f'text/{date}')):
                for file in files:
                    goods.add(file)

        total_attr_dic = self.attribute_counter(goods)

        newly_released_relative = {}
        for key in newly_released_attr_dic.keys():
            if total_attr_dic[key] != 0:
                newly_released_relative[key] = newly_released_attr_dic[key] / total_attr_dic[key]
            else:
                newly_released_relative[key] = 0

        return newly_released_relative

    def calculate_ma(self, days, date=None):
        if not date:
            end_time = parser.parse(datetime.datetime.today().strftime("%Y-%m-%d"))
        elif isinstance(date, str):
            end_time = parser.parse(date)
        else:
            end_time = date
        start_time = end_time - relativedelta(days=days)
        goods_info = self.db.fashion.goods_info
        goods = goods_info.find({'first_appearance': {'$gt': start_time, '$lte': end_time}}, {'_id': 0, 'attributes': 1})
        results = None
        for good in goods:
            if not isinstance(results, Series):
                results = Series(good['attributes'])
            else:
                results += Series(good['attributes'])
        results /= days
        return results


if __name__ == '__main__':
    input_path = 'data/lstm-label.txt'
    attributes = ['floral', 'striped', 'plaid', 'leopard', 'camo', 'graphic',
                  'crew neck', 'square neck', 'v-neck',
                  'maxi dress', 'midi dress', 'mini dress',
                  'denim', 'knit', 'faux leather', 'cotton', 'chiffon', 'satin',
                  'mesh', 'ruched', 'cutout', 'lace', 'frayed', 'wrap',
                  'tropical', 'peasant', 'swim', 'bikini', 'active', 'cargo']

    data_path = 'spider_foever21/data'

    self = Analyzer(data_path, attributes)
    # test = self.get_newly_released('2021-05-01', 'monthly', 1)
    test = self.calculate_ma(60)
    print(test)

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
# top_k_generator = TopKGenerator(input_path, attributes)

# for root, dirs, files in os.walk(os.path.join(self.data_path, f'text_unique')):
#     for filename in files:
#         if filename[:-4] in pids:
#             with open(os.path.join(root, filename), 'r', encoding='utf8') as f:
#                 detail = eval(f.read())
#
#             description = ''
#             for key in ['name', 'details', 'Details']:
#                 if key in detail.keys():
#                     description = ' '.join([description, str(detail[key])])
#
#             for attr in self.attributes:
#                 if attr in description:
#                     newly_released_attr_dic[attr] += 1
