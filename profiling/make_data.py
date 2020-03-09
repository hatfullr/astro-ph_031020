import numpy as np

N = 500
x = np.random.rand(N)
y = np.random.rand(N)
z = np.random.rand(N)

data = np.zeros(shape=(N,3))
data[:,0] = x
data[:,1] = y
data[:,2] = z

np.save("data.npy",data)
