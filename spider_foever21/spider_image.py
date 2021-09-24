import os
import time
from datetime import date

import requests

today = date.today().strftime("%Y-%m-%d")
text_path = f'data/text/{today}'
# text_path = f'data/text/2021-08-21'
# text_path = f'data/text'
img_path = 'data/image'

for root, dirs, files in os.walk(text_path):
    for name in files:
        pid = name.split('.')[0]
        current_path = os.path.join(img_path, pid)
        current_text = os.path.join(root, name)

        try:
            with open(current_text, 'r', encoding='utf8') as f:
                attri_dic = eval(f.read())
        except:
            continue

        if (not os.path.exists(current_path)) and ('img_URL' in attri_dic.keys()):
            print("Work on {}".format(pid))
            os.makedirs(current_path)
            for img_URL in attri_dic['img_URL']:
                img_title = img_URL.split('/')[-2] + '.jpg'
                destiny = os.path.join(current_path, img_title)
                if not os.path.exists(destiny):
                    resp = requests.get(img_URL)
                    if resp.status_code == 200:
                        with open(destiny, 'wb') as f:
                            f.write(resp.content)
                        time.sleep(0.5)

