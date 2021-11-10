import os
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None

from random import seed
RANDOM_SEED = 42
seed(RANDOM_SEED) # set the random seed so that the random permutations can be reproduced again
np.random.seed(RANDOM_SEED)


def load_sba(shift=False):                                                                      
    orig_file = os.path.join(os.path.dirname(__file__), 'processed_data/sba_processed_.csv')           
    processed_file = os.path.join(os.path.dirname(__file__), 'processed_data/sba_processed_shift_.csv')
    read_file = orig_file if not shift else processed_file                                               
                                                                                                         
    df = pd.read_csv(read_file)
    for i in range(df.shape[0]):
        if df.iloc[i, 5] > 1:
            df.iloc[i, 5] = 1.0
    # df.iloc[392, 4] = 1.0
    df['New'] += 1
    df['RealEstate'] += 1
    df['Recession'] += 1
    return df.astype('float64')




# import numpy as np
# import pandas as pd

# import loadData

# from random import seed
# RANDOM_SEED = 54321
# seed(RANDOM_SEED) # set the random seed so that the random permutations can be reproduced again
# np.random.seed(RANDOM_SEED)

# from sklearn.linear_model import LogisticRegression
# from sklearn.linear_model import LinearRegression
# from sklearn.linear_model import Lasso

# dataset_obj = loadData.loadDataset('german', return_one_hot = False, load_from_cache = False)
# df = dataset_obj.data_frame_kurz

# # See Figure 3 in paper

# # Credit
# X_train = df[['x0', 'x1']]
# y_train = df[['x2']]
# model_pretrain = LinearRegression()
# # model_pretrain = Lasso()
# model_trained = model_pretrain.fit(X_train, y_train)
# print(model_trained.coef_)
# print(model_trained.intercept_)

# # Loan duration
# X_train = df[['x2']]
# y_train = df[['x3']]
# model_pretrain = LinearRegression()
# model_trained = model_pretrain.fit(X_train, y_train)
# print(model_trained.coef_)
# print(model_trained.intercept_)




