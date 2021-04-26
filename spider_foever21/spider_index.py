# import requests
from bs4 import BeautifulSoup
from datetime import date
import re
import os
import time

from selenium import webdriver

target_URLs = {'women': 'https://www.forever21.com/us/shop/catalog/category/f21/app-main',
               'mens': 'https://www.forever21.com/us/shop/catalog/category/f21/mens-clothing',
               'girls': 'https://www.forever21.com/us/shop/catalog/category/f21/girls-clothing'}

data_dir = f'data'

today = date.today().strftime("%Y-%m-%d")
index_dir = os.path.join(data_dir, f'index')
if not os.path.exists(index_dir):
    os.makedirs(index_dir)

data_pid = {}
for key in target_URLs.keys():
    data_pid[key] = []
    URL = target_URLs[key]
    # html_doc = requests.get(URL)
    browser = webdriver.Chrome()
    browser.get(URL)
    # if html_doc.status_code != 200:
    #     raise ValueError("URL: {} 请求失败，错误码： {}".format(URL, html_doc.status_code))
    soup = BeautifulSoup(browser.page_source, features="html.parser")
    pages = int(soup.find_all('button', attrs={'aria-label': re.compile('View Page.*')})[-1].text)
    for page in range(pages):
        if page % 5 == 0:
            print('working on {}, page {} of {}.'.format(key, page, pages))
        browser.get(URL + "?start=" + str(page * 32) + "&sz=32")
        soup = BeautifulSoup(browser.page_source, features="html.parser")
        products = soup.find_all('div', attrs={'class': "product"})
        for product in products:
            data_pid[key].append(product.div['data-pid'])
        time.sleep(0.5)
with open(os.path.join(index_dir, f'{today}.txt'), 'w') as f:
    f.write(str(data_pid))
    f.close()

print("done!")