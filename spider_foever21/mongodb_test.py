import os
import pymongo
from dateutil import parser
import json

myclient = pymongo.MongoClient("mongodb://qyli:qyli2233@202.120.37.116:2233/admin")

goods_info = myclient.fashion.goods_info
goods = goods_info.find({}, {'_id': 1, 'name': 1, 'Details': 1, 'Content + Care': 1})
for good in goods:
    pid = good['_id']
    detail = ' '.join([str(i).lower() for i in good.values()])
    anno_dict = {
        'floral': 0, 'striped': 0, 'plaid': 0, 'leopard': 0, 'camo': 0, 'graphic': 0, 'crew neck': 0, 'square neck': 0,
        'v-neck': 0, 'maxi dress': 0, 'midi dress': 0, 'mini dress': 0, 'denim': 0, 'knit': 0, 'faux leather': 0,
        'cotton': 0, 'chiffon': 0, 'satin': 0, 'mesh': 0, 'ruched': 0, 'cutout': 0, 'lace': 0, 'frayed': 0, 'wrap': 0,
        'tropical': 0, 'peasant': 0, 'swim': 0, 'bikini': 0, 'active': 0, 'cargo': 0
    }
    for anno in anno_dict.keys():
        if anno in detail:
            anno_dict[anno] = 1
    goods_info.update_one({'_id': pid}, {'$set': {'attributes': anno_dict}})


def generate_goods_info(myclient):
    goods_info = myclient.fashion.goods_info

    with open('data/first_appearance.txt', 'r') as f:
        first_appearance = eval(f.read())

    data = []
    for key, value in first_appearance.items():
        data.append({"_id": key, "first_appearance": parser.parse(value)})

    goods_info.insert_many(data)

    valid_keys = ['name', 'Details', 'Content + Care', 'Size + Fit', 'price', 'img_URL']
    for root, dirs, files in os.walk('data/text_unique'):
        for file in files:
            with open(os.path.join(root, file), 'r', encoding='utf8') as f:
                data = f.read()
            # if '<name>' in data:
            #     continue
            data = eval(data)
            tmp = {}
            for key in valid_keys:
                if key in data.keys():
                    tmp[key] = data[key]
            data = tmp
            data['_id'] = file.split('.')[0]
            goods_info.update_many({'_id': data['_id']}, {"$set": data}, upsert=True)


def insert_index():
    index = myclient.fashion.index
    for root, dirs, files in os.walk('data/index'):
        for file in files:
            date = parser.parse(file.split('.')[0])
            with open(os.path.join(root, file), 'r', encoding='utf8') as f:
                data = eval(f.read())
            data['_id'] = date
            index.update_one({'_id': date}, {"$set": data}, upsert=True)


def generate_lstm_labels():
    index = myclient.fashion.index
    goods_info = myclient.fashion.goods_info
    lstm_labels = myclient.fashion.lstm_labels
    daily_records = index.find()
    for daily_record in daily_records:
        today = daily_record['_id']
        pids = []
        for key in ['women', 'mens', 'girls']:
            pids.extend(daily_record[key])

        goods = goods_info.find({'_id': {'$in': pids}}, {'_id': 0, 'name': 1, 'Details': 1, 'Content + Care': 1})
        anno_dict = {
            'floral': 0, 'striped': 0, 'plaid': 0, 'leopard': 0, 'camo': 0, 'graphic': 0, 'crew neck': 0, 'square neck': 0,
            'v-neck': 0, 'maxi dress': 0, 'midi dress': 0, 'mini dress': 0, 'denim': 0, 'knit': 0, 'faux leather': 0,
            'cotton': 0, 'chiffon': 0, 'satin': 0, 'mesh': 0, 'ruched': 0, 'cutout': 0, 'lace': 0, 'frayed': 0, 'wrap': 0,
            'tropical': 0, 'peasant': 0, 'swim': 0, 'bikini': 0, 'active': 0, 'cargo': 0
        }
        for good in goods:
            detail = ' '.join([str(i).lower() for i in good.values()])
            for anno in anno_dict.keys():
                if anno in detail:
                    anno_dict[anno] += 1
        anno_dict['_id'] = today
        lstm_labels.insert_one(anno_dict)

def generate_raw_text():
    raw_text = myclient.fashion.raw_text
    for root, dirs, files in os.walk(f'data/text'):
        for file in files:
            current_path = os.path.join(root, file)
            text = {'pid': file.split('.')[0], 'date': parser.parse(current_path.split(os.sep)[-3])}
            selector = text
            if not raw_text.find_one(selector):
                with open(current_path, 'r', encoding='utf8') as f:
                    text['text'] = f.read()
                raw_text.insert_one(text)

# def fill_missing_detail():
#     from selenium import webdriver
#
#     browser = webdriver.Chrome()
#     from bs4 import BeautifulSoup
#
#     goods_info = myclient.fashion.goods_info
#     records = goods_info.find({'Details': {'$type': 'array'}}, {'_id': 1})
#     for record in records:
#         pid = record['_id']
#         browser.get(f'https://www.forever21.com/us/{pid}.html')
#         soup = BeautifulSoup(browser.page_source, features="html.parser")
#         sections = soup.find_all('section', attrs={'class': "d_wrapper"})
#         for section in sections:
#             section_key = section.h3.text.strip('\n')
#             if section_key == 'Details':
#                 Detail = {'Details': section.div.text.strip('\n')}
#                 goods_info.update_one({'_id': pid}, {'$set': Detail})
#                 print(Detail)
#                 break
