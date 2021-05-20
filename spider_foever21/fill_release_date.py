import os
from datetime import timedelta
from dateutil import parser

root_dir = 'data/text'
start_date = '2021-04-20'

today = parser.parse(start_date)
current_path = os.path.join(root_dir, today.strftime('%Y-%m-%d'))
first_appearance = {}
while os.path.exists(current_path):
    for root, dirs, files in os.walk(current_path):
        for file in files:
            pid = file.split('.')[0]
            if pid not in first_appearance.keys():
                first_appearance[pid] = today.strftime('%Y-%m-%d')
    today = today + timedelta(days=1)
    current_path = os.path.join(root_dir, today.strftime('%Y-%m-%d'))

with open('data/first_appearance.txt', 'w') as f:
    f.write(str(first_appearance))