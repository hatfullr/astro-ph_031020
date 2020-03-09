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
        dx2 = (x[i]-x)**2.
        dy2 = (y[i]-y)**2.
        dz2 = (z[i]-z)**2.
        dr = (dx2 + dy2 + dz2)**0.5
        r[i*N:(i+1)*N] = dr
main()


