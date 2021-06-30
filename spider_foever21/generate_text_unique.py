import os

def generator():
    text_dir = f'data/text'
    output_dir = f'data/text_unique'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if False : # os.path.exists(os.path.join(output_dir, f'latest_update.txt')):
        pass
    else:
        # process all data
        date_lis = []
        root_depth = len(text_dir.split(os.path.sep))
        for root, dirs, files in os.walk(text_dir):
            for dir in dirs:
                current_path = os.path.join(root, dir)
                if len(current_path.split(os.path.sep)) == root_depth + 1:
                    date_lis.append(dir)

    for date in date_lis:
        for root, dirs, files in os.walk(os.path.join(text_dir, date)):
            for file in files:
                with open(os.path.join(output_dir, file), 'wb') as desf:
                    with open(os.path.join(root, file), 'rb') as srcf:
                        desf.write(srcf.read())

    with open(os.path.join(output_dir, f'latest_update.txt'), 'w') as f:
        f.write(date_lis[-1])

for root, dirs, files in os.walk(f'data/text_unique'):
    for file in files:
        with open(os.path.join(root, file), 'r', encoding='utf8') as f:
            text = f.read()

        try:
            data = eval(text)
        except:
            data = {}
            data['name'] = text.split('<name>')[1].split('</name>')[0].strip('\n')
            data['Details'] = text.split('<description>')[1].split('</description>')[0].strip('\n')
            with open(os.path.join(root, file), 'w', encoding='utf8') as f:
                f.write(str(data))
            continue

        if 'Details' not in data.keys():
            for key in ['description', 'details']:
                if key in data.keys():
                    data['Details'] = data[key]
                    break
            with open(os.path.join(root, file), 'w', encoding='utf8') as f:
                f.write(str(data))
            print(data)
