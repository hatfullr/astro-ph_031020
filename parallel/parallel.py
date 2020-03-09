import numpy as np
from scipy.spatial import distance
from multiprocessing import Pool
import time

# The function each process will run
def get_distance(xyz,idxs):
    return distance.cdist(xyz[idxs],xyz), idxs

# Prevents subprocesses from spawning subsubprocesses etc.
if __name__ == "__main__":
    xyz = np.load("data.npy")
    N = len(xyz)
    
    pool = Pool() # Create processes
    nprocs = pool._processes

    # Make an array of indices and split it up among
    # the processes
    idxs = np.arange(N)
    idxs = np.array_split(idxs,nprocs)

    # Assign the processes a job to do
    procs = []
    for i in range(0,nprocs):
        P = pool.apply_async(get_distance,args=(xyz,idxs[i]))
        procs.append(P)

    r = np.zeros(shape=(N,N))
    
    # Start the processes and collect the results as they come
    start = time.time()
    for i in range(0,nprocs):
        rtemp, idx = procs[i].get()
        r[idx] = rtemp
    print(time.time()-start) # 0.00565
    
    pool.close() # End the processes


    

