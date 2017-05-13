import numpy as np
import numpy.random as rnd
import os
import seaborn as sns

sns.set_style('whitegrid')

MOVIE_PATH = "Dataset"

import pandas as pd

def load_data(path=MOVIE_PATH):
    csv_path = os.path.join(path, "movie_metadata.csv")
    return pd.read_csv(csv_path)

raw_data = load_data()


import matplotlib
matplotlib.use('tkagg')
matplotlib.interactive(1)
import matplotlib.pyplot as plt


raw_data.hist(bins=50, figsize=(11,8))


corr_matrix = raw_data.corr()
yticks = raw_data.index

plt.figure(figsize=(15,15))
plt.xticks(rotation = 90)
plt.yticks(rotation = 0)
sns.heatmap(corr_matrix, vmax=1, square=True,annot=True,cmap='cubehelix')


from pandas.tools.plotting import scatter_matrix

attributes = ["imdb_score","actor_3_facebook_likes", "duration", "cast_total_facebook_likes", "num_user_for_reviews"]
scatter_matrix(raw_data[attributes], figsize=(11, 8))

plt.show()
