import csv
import random

header = ['label'] + [f'pixel{i}' for i in range(1, 577)]  # 576 pixels (24x24)
rows = []

for _ in range(1000):
    label = random.randint(0, 9)
    pixels = [random.randint(0, 255) for _ in range(576)]
    rows.append([label] + pixels)

with open('D:\poc\sample_dataset.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)