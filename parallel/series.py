import numpy as np
from scipy.spatial import distance
import time

xyz = np.load("data.npy")
start = time.time()
r = distance.cdist(xyz,xyz)
print(time.time()-start) # 0.001206



