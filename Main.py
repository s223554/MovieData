import numpy as np
import numpy.random as rnd
import os
import matplotlib
matplotlib.use('tkagg')
matplotlib.interactive(1)
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')

MOVIE_PATH = "Dataset"

import pandas as pd

def load_data(path=MOVIE_PATH):
    csv_path = os.path.join(path, "movie_metadata.csv")
    return pd.read_csv(csv_path)

raw_data = load_data()





raw_data.hist(bins=50, figsize=(11,8))


corr_matrix = raw_data.corr()
yticks = raw_data.index

plt.figure(figsize=(15,15))
plt.xticks(rotation = 90)
plt.yticks(rotation = 0)
sns.heatmap(corr_matrix, vmax=1, square=True,annot=True,cmap='cubehelix')


from pandas.tools.plotting import scatter_matrix

attributes = ["imdb_score", "duration", "cast_total_facebook_likes", "num_user_for_reviews"]
scatter_matrix(raw_data[attributes], figsize=(11, 8))
attri_train = ["duration", "cast_total_facebook_likes", "num_user_for_reviews"]

from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from sklearn.base import BaseEstimator, TransformerMixin
#from sklearn_pandas import DataFrameMapper

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


num_pipeline = Pipeline([
#    ('selector',DataFrameMapper(attributes))
    ('selector',DataFrameSelector(attri_train)),
    ('imputer',Imputer(strategy='median')),
#    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
])

data_prepared = num_pipeline.fit_transform(raw_data)

from sklearn.linear_model import LinearRegression

data_labels = raw_data['imdb_score']
lin_reg = LinearRegression()
lin_reg.fit(data_prepared, data_labels)

lin_reg.predict(data_prepared[1:10])
