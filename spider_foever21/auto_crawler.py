'''
每天指定时间自动爬取forever21数据
'''
import pymongo
import datetime
import time

wait_count = 0
while True:
    if not datetime.time(1, 0) <= datetime.datetime.now().time() <= datetime.time(2, 0):
        wait_count += 1
        if wait_count % 4 == 0:
            print("Waiting... time: {}".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            wait_count = 0
        time.sleep(15 * 60)
    else:
        try:
            myclient = pymongo.MongoClient("mongodb://qyli:qyli2233@202.120.37.116:2233/admin")
            myclient.close()
        except:
            raise ValueError("pymongo connection error!")

        # index
        with open('spider_index.py', 'r', encoding='utf8') as f:
            exec(f.read())

        # detail
        failure = 0
        with open('spider_detail.py', 'r', encoding='utf8') as f:
            exec(f.read())
        if failure > 0:
            with open('spider_detail.py', 'r', encoding='utf8') as f:
                exec(f.read())
        if failure > 100:
            raise ValueError("too much failure")

        # image
        with open('spider_image.py', 'r', encoding='utf8') as f:
            exec(f.read())
    # break
