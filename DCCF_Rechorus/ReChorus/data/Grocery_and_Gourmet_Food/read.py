import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import scipy.sparse as sp

df = pd.read_csv("data\\Grocery_and_Gourmet_Food\\train.csv", sep="\t").reset_index(drop=True).sort_values(by = ['user_id','time'])
# print(df)
mat = {}
n_user, n_item = df["user_id"].max() + 1, df["item_id"].max() + 1
matrix = np.zeros((n_user, n_item))
print(matrix.shape)
df = df.apply(lambda x: tuple(x), axis=1).values.tolist()
for instance in df:
    matrix[instance[0], instance[1]] = 1.0
data_mat = sp.coo_matrix(matrix)
mat['train'] = data_mat

#df = df.apply(lambda x: tuple(x), axis=1).values.tolist()
print(mat["train"].todok())
