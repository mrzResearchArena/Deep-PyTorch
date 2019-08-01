'''
Singular Value Decomposition (SVD)
'''

import numpy as np

A = np.array([
        [1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 1, 0],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 1],
        [0, 1, 0, 1],
        [0, 1, 0, 0],
        [0, 1, 0, 1],
        [0, 1, 0, 1],
        [0, 1, 0, 0],
        [0, 1, 0, 1],
        [0, 1, 0, 0],
        [0, 1, 0, 1],
        [0, 1, 0, 0],
        [0, 1, 0, 1],
        [0, 1, 0, 0],
        [0, 1, 0, 1],
        [0, 1, 0, 0],
        [0, 1, 0, 1],
        [0, 1, 0, 0],
        [0, 1, 0, 1],
        [0, 1, 0, 1],
        [0, 1, 0, 0],
        ])

#%%
feature = [0] * A.shape[0] # --> Upto the #row
#%%

u, s, v = np.linalg.svd(A)
#A      --> 25 x 4
#u      --> 25 x 25
#s      --> 4
#v      --> 4 x 4
#feature--> 25
#%%

for i in range(len(s)):
    feature = feature + (u[i] * s[i]) / A.shape[0]

#%%

print(feature)
