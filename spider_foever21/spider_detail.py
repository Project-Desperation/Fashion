# import requests
from bs4 import BeautifulSoup
from datetime import date
import pymongo
import os
from dateutil import parser
import time
import re

from selenium import webdriver

browser = webdriver.Chrome()
# database = pymongo.MongoClient("mongodb://qyli:qyli2233@202.120.37.116:2233/admin")
# goods_info = database.fashion.goods_info
# valid_keys = ['name', 'Details', 'Content + Care', 'Size + Fit', 'price', 'img_URL']

today = date.today().strftime("%Y-%m-%d")
# today = '2021-06-09'
data_dir = f'data'
update_first_appearance = True

with open(os.path.join(data_dir, f'index/{today}.txt'), 'r') as f:
    data_pid = eval(f.read())
    f.close()

failure = 0
for key in data_pid.keys():
    print('working on {} details, total {}.'.format(key, len(data_pid[key])))
    count = 0
    current_text_path = os.path.join(data_dir, f'text/{today}/{key}')
    if not os.path.exists(current_text_path):
        os.makedirs(current_text_path)
    for pid in data_pid[key]:
        if count % 100 == 0:
            print("{} detail {} / {}, failure {}.".format(key, count, len(data_pid[key]), failure))
        count += 1
        current_img_path = os.path.join(data_dir, f'image/{today}/{key}/{pid}')
        soup = None

        # # image
        # if not os.path.exists(current_img_path):
        #     try:
        #         URL = f'https://www.forever21.com/us/{pid}.html'
        #         html_doc = requests.get(URL)
        #         if html_doc.status_code != 200:
        #             raise ValueError("URL: {} 请求失败，错误码： {}".format(URL, html_doc.status_code))
        #         soup = BeautifulSoup(html_doc.content, features="lxml")
        #     except:
        #         failure += 1
        #         continue
        #
        #     os.makedirs(current_img_path)
        #     img_list = soup.find_all('img', attrs={'class': re.compile("product-thumb__item__img.*")})
        #     for img in img_list:
        #         if img.get('src'):
        #             img_URL = img.get('src').split('?')[0]
        #             img_title = img_URL.split('/')[-2] + '.jpg'
        #             img_content = requests.get(img_URL).content
        #             with open(os.path.join(current_img_path, img_title), 'wb') as f:
        #                 f.write(img_content)
        #                 f.close()

        # text
        if not os.path.exists(os.path.join(current_text_path, f'{pid}.txt')):
            try:
                if not soup:
                    URL = f'https://www.forever21.com/us/{pid}.html'
                    browser.get(URL)
                    # if html_doc.status_code != 200:
                    #     raise ValueError("URL: {} 请求失败，错误码： {}".format(URL, html_doc.status_code))
                    soup = BeautifulSoup(browser.page_source, features="html.parser")

                attr_dic = {}
                attr_dic['name'] = soup.find_all('h1', attrs={'itemprop': "name"})[0].text.strip('\n')
                # description
                try:
                    sections = soup.find_all('section', attrs={'class': "d_wrapper"})
                    for section in sections:
                        section_key = section.h3.text.strip('\n')
                        if section_key not in attr_dic.keys():
                            if section_key == 'Details':
                                attr_dic[section_key] = section.div.text.strip('\n')
                            elif section.div.p:
                                attr_dic[section_key] = []
                                for para in section.div.find_all('p'):
                                    attr_dic[section_key].append(para.text.strip('\n'))
                            else:
                                attr_dic[section_key] = section.div.text.strip('\n')
                except:
                    pass
                # price
                try:
                    attr_dic['price'] = soup.find('span', attrs={'itemprop': 'price'}).text.strip('\n')
                except:
                    pass
                # contents and care
                try:
                    pass
                except:
                    pass
                # img URLs
                try:
                    attr_dic['img_URL'] = []
                    img_list = soup.find_all('img', attrs={'class': re.compile("product-thumb__item__img.*")})
                    for img in img_list:
                        if img.get('src'):
                            img_URL = img.get('src').split('?')[0]
                            attr_dic['img_URL'].append(img_URL)
                except:
                    pass

                time.sleep(0.5)

            except:
                failure += 1
                continue

            with open(os.path.join(current_text_path, f'{pid}.txt'), 'w', encoding='utf-8') as f:
                if os.path.exists(os.path.join(data_dir, f'text_unique')):
                    with open(os.path.join(data_dir, f'text_unique/{pid}.txt'), 'w', encoding='utf-8') as desf:
                        desf.write(str(attr_dic))
                else:
                    raise ValueError('Missing text_unique')
                f.write(str(attr_dic))
                f.close()

    print("Finished {}! failure: {}".format(key, failure))
# ----------------------------------------------------------------------------------------------------------------------
# update first_appearance
if update_first_appearance:
    print("update first_appearance")
    first_appearance_path = os.path.join(data_dir, 'first_appearance.txt')

    with open(first_appearance_path, 'r') as f:
        first_appearance = eval(f.read())

    count = 0
    for root, dirs, files in os.walk(os.path.join(data_dir, 'text', today)):
        for file in files:
            pid = file.split('.')[0]
            if pid not in first_appearance.keys():
                first_appearance[pid] = today
                count += 1

    with open(first_appearance_path, 'w') as f:
        f.write(str(first_appearance))

    print("first_appearance updated! added {} new goods".format(count))


def update_mongo(index_path, text_path, today):
    # Connect
    print("Connect to mongodb...")
    myclient = pymongo.MongoClient("mongodb://qyli:qyli2233@202.120.37.116:2233/admin")
    print("Connected!")

    # update index
    print("Insert index")
    today = parser.parse(today)
    index = myclient.fashion.index
    if not index.find_one({'_id': today}):
        with open(index_path, 'r') as f:
            data = eval(f.read())
        data['_id'] = today
        index.insert_one(data)
    else:
        print("index of {} already exist.".format(today.strftime('%Y-%m-%d')))

    # update goods_info and raw_text
    print("Update goods_info")
    valid_keys = ['name', 'Details', 'Content + Care', 'Size + Fit', 'price', 'img_URL']
    goods_info = myclient.fashion.goods_info
    raw_text = myclient.fashion.raw_text
    count = 0
    pids = []
    for root, dirs, files in os.walk(text_path):
        for file in files:
            current_path = os.path.join(root, file)
            text = {'pid': file.split('.')[0], 'date': today}
            selector = text
            if not raw_text.find_one(selector):
                with open(current_path, 'r', encoding='utf8') as f:
                    text['text'] = f.read()
                raw_text.insert_one(text)

            pid = file.split('.')[0]
            pids.append(pid)
            if not goods_info.find_one({'_id': pid}, {}):
                attrs = {'_id': pid, 'first_appearance': today}
                with open(os.path.join(root, file), 'r', encoding='utf8') as f:
                    raw_attrs = eval(f.read())
                description = ''
                for valid_key in valid_keys:
                    if valid_key in raw_attrs.keys():
                        attrs[valid_key] = raw_attrs[valid_key]
                        if valid_key in ['name', 'Content + Care', 'Details']:
                            description = ' '.join([description, str(raw_attrs[valid_key])])

                anno_dict = {
                    'floral': 0, 'striped': 0, 'plaid': 0, 'leopard': 0, 'camo': 0,
                    'graphic': 0, 'crew neck': 0, 'square neck': 0,
                    'v-neck': 0, 'maxi dress': 0, 'midi dress': 0, 'mini dress': 0, 'denim': 0,
                    'knit': 0, 'faux leather': 0,
                    'cotton': 0, 'chiffon': 0, 'satin': 0, 'mesh': 0, 'ruched': 0, 'cutout': 0,
                    'lace': 0, 'frayed': 0, 'wrap': 0,
                    'tropical': 0, 'peasant': 0, 'swim': 0, 'bikini': 0, 'active': 0, 'cargo': 0
                }
                for anno in anno_dict.keys():
                    if anno in description:
                        anno_dict[anno] = 1
                attrs['attributes'] = anno_dict

                goods_info.insert_one(attrs)
                count += 1
    print("Added {} new goods.".format(count))

    # update lstm_labels
    print('Insert lstm_labels')
    lstm_labels = myclient.fashion.lstm_labels
    if not lstm_labels.find_one({'_id': today}):
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
    else:
        print("lstm_labels of {} already exist.".format(today.strftime('%Y-%m-%d')))



update_mongo(os.path.join(data_dir, f'index/{today}.txt'), os.path.join(data_dir, f'text/{today}'), today)
# ----------------------------------------------------------------------------------------------------------------------
print("work done!")

# # update database
# data = {}
# upsert = False
# for valid_key in valid_keys:
#     if valid_key in attr_dic.keys():
#         data[valid_key] = attr_dic[valid_key]
# if not goods_info.find_one({'_id': pid}, {}):
#     data['_id'] = pid
#     data['first_appearance'] = parser.parse(today)
#     upsert = True
# goods_info.update_one({'_id': pid}, {"$set": data}, upsert=upsert)