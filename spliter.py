import csv
import re

data_positive = []
data_negative = []

with open('all.csv', 'r', encoding='gbk') as f:
    reader = csv.reader(f)
    for row in reader:
        if row[1] == '积极':
            data_positive.append(row)
        elif row[1] == '消极':
            data_negative.append(row)

f.close()

print(len(data_positive))
print(len(data_negative))


with open('datasets/train.csv', 'w', encoding='utf-8') as f:
    writer = csv.writer(f)
    for row in data_positive[:35000]:
        writer.writerow(row)

    for row in data_negative[:35000]:
        writer.writerow(row)

f.close()

with open('datasets/valid.csv', 'w', encoding='utf-8') as f:
    writer = csv.writer(f)
    for row in data_positive[40000:42000]:
        writer.writerow(row)

    for row in data_negative[40000:42000]:
        writer.writerow(row)

f.close()

with open('datasets/test.csv', 'w', encoding='utf-8') as f:
    writer = csv.writer(f)
    for row in data_positive[-5000:]:
        writer.writerow(row)

    for row in data_negative[-5000:]:
        writer.writerow(row)

f.close()
