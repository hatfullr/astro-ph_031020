import numpy as np
data = np.load("data.npy")
x = data[:,0] # Use NumPy arrays
y = data[:,1]
z = data[:,2]
N = len(x) # x, y, and z all same length

@profile
def main():
    r = np.zeros(N*N)
    for i in range(0,N):
        for j in range(0,N):
            dx2 = (x[i]-x[j])**2.
            dy2 = (y[i]-y[j])**2.
            dz2 = (z[i]-z[j])**2.
            dr = (dx2 + dy2 + dz2)**0.5
            r[i*N+j] = dr
main()


