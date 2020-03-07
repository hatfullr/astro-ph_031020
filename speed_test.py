import numpy as np
import time
import random

N = 1000
x = np.random.rand(N).tolist()
y = np.random.rand(N).tolist()
z = np.random.rand(N).tolist()

#####################
# Allocate array during calculation
total = 0
for t in range(0,3):
    r = [] # Create array
    start = time.time()
    for i in range(0,N):
        for j in range(0,N): # Append the array
            r.append(((x[i]-x[j])**2.+(y[i]-y[j])**2.+(z[i]-z[j])**2.)**0.5)
    total += time.time()-start
print(total/3.) # 0.507244984309

#####################
# Allocate array before calculation
total = 0
for t in range(0,3):
    r = [None]*N*N # Allocate array
    start = time.time()
    for i in range(0,N):
        p = i*N    # Store position in array r
        xi = x[i]  # Store x[i]
        yi = y[i]  # Store y[i]
        zi = z[i]  # Store z[i]
        for j in range(0,N):
            r[p + j] = ((xi-x[j])**2.+(yi-y[j])**2.+(zi-z[j])**2.)**0.5
    total += time.time()-start
print(total/3.) # 0.420340220133

#####################
# Python list comprehension
total = 0
for t in range(0,3):
    start = time.time()
    r = [((x[i]-x[j])**2.+(y[i]-y[j])**2.+(z[i]-z[j])**2.)**0.5 for i in range(0,N) for j in range(0,N)]
    total += time.time()-start
print(total/3.) # 0.402298688889

#####################
# NumPy
import numpy as np
x = np.asarray(x) # Convert to NumPy arrays
y = np.asarray(y)
z = np.asarray(z)

total = 0
for t in range(0,3):
    r = np.zeros(N*N) # Create array
    start = time.time()
    for i in range(0,N): # Subtract each value at i from all other values in each array
        r[i*N:(i+1)*N] = np.sqrt((x[i]-x)**2. + (y[i]-y)**2. + (z[i]-z)**2.)
    total += time.time()-start
print(total/3.) # 0.0125865936279

#####################
# SciPy
from scipy.spatial import distance
xyz = np.zeros(shape=(N,3)) # Create one big array
xyz[:,0] = x
xyz[:,1] = y
xyz[:,2] = z
total = 0
for t in range(0,3):
    start = time.time()
    r = distance.cdist(xyz,xyz) # Use .flatten() to get a 1D array
    total += time.time() - start
print(total/3.) # 0.00238434473674
