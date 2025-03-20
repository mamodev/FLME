import pandas as pd
import os
from matplotlib import pyplot as plt


csv_files = []

csv_folder = 'arxiv_res/'

if not os.path.exists(csv_folder):
    exit(1)

for file in os.listdir(csv_folder):
    if file.endswith('.csv'):
        csv_files.append(file)


columns = ["Round", "MeanNorm", "MaxNorm", "Loss"]

metrics = ['MeanNorm', 'MaxNorm', 'Loss', 'MaxNorm:MeanNorm']

for f in csv_files:
    #THE FISRT ROW IS THE HEADER
    df = pd.read_csv(csv_folder + f, names=columns, header=0)
    df['MaxNorm:MeanNorm'] = df['MaxNorm'] / df['MeanNorm']

    base_name = os.path.basename(f).split('.')[0]
    for metric in metrics:
        df.plot(x='Round', y=metric, title=base_name + ' ' + metric)
        plt.savefig('arxiv_res/' + base_name + '_' + metric + '.png')
        plt.close()



   