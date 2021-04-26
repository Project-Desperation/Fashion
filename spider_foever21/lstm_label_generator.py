import os
import copy

anno_dict = {
    'floral': 0, 'striped': 0, 'plaid': 0, 'leopard': 0, 'camo': 0, 'graphic': 0, 'crew neck': 0, 'square neck': 0,
    'v-neck': 0, 'maxi dress': 0, 'midi dress': 0, 'mini dress': 0, 'denim': 0, 'knit': 0, 'faux leather': 0,
    'cotton': 0, 'chiffon': 0, 'satin': 0, 'mesh': 0, 'ruched': 0, 'cutout': 0, 'lace': 0, 'frayed': 0, 'wrap': 0,
    'tropical': 0, 'peasant': 0, 'swim': 0, 'bikini': 0, 'active': 0, 'cargo': 0
}

annotations = ['floral', 'striped', 'plaid', 'leopard', 'camo', 'graphic', 'crew neck', 'square neck', 'v-neck',
             'maxi dress', 'midi dress', 'mini dress', 'denim', 'knit', 'faux leather', 'cotton', 'chiffon', 'satin',
             'mesh', 'ruched', 'cutout', 'lace', 'frayed', 'wrap', 'tropical', 'peasant', 'swim', 'bikini', 'active', 'cargo']


def dict_to_string(dic):
    array = []
    for _, v in dic.items():
        array.append(str(v))
    return ' '.join(array)


root_dir = 'data/text'

if os.path.exists('lstm_labels.txt'):
    with open('lstm_labels.txt', 'r') as f:
        exist_labels = f.read()
else:
    exist_labels = ''

for day in sorted(os.listdir(root_dir)):
    if day in exist_labels:
        continue
    day_path = os.path.join(root_dir, day)
    tmp = copy.copy(anno_dict)
    for p, _, f in os.walk(day_path):
        for ff in f:
            filename = os.path.join(p, ff)
            with open(filename, encoding='utf-8') as fi:
                text = fi.read().lower()
                for anno in annotations:
                    if anno in text:
                        tmp[anno] = tmp[anno] + 1
    print(f'day: {day}, statistics: {tmp}')
    with open('lstm_labels.txt', 'a') as a:
        a.write(f'{day}: {dict_to_string(tmp)}\n')
