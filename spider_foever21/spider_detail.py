# import requests
from bs4 import BeautifulSoup
from datetime import date
import os
import time
import re

from selenium import webdriver

browser = webdriver.Chrome()

today = date.today().strftime("%Y-%m-%d")
# today = '2021-04-15'
data_dir = f'data'

with open(os.path.join(data_dir, f'index/{today}.txt'), 'r') as f:
    data_pid = eval(f.read())
    f.close()

for key in data_pid.keys():
    print('working on {} details, total {}.'.format(key, len(data_pid[key])))
    count = 0
    failure = 0
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
                            if section.div.p:
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
                f.write(str(attr_dic))
                f.close()
