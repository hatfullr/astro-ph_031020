def f(x,ID):
    # Some function that requires lots of time to compute per iteration
    ret = np.zeros(len(ID))
    for i in range(0,len(ID)):
        this_x = x[ID[i]]
        for j in range(0,len(ID)):
            if this_x > 0:
                ret[j] = 0.
            elif this_x > 1.:
                ret[j] = 1.
            else:
                ret[j] = 0.
    return ret, ID

from multiprocessing import Pool
import numpy as np
import time

# Prevents spawned processes from recursing.
if __name__ == "__main__":
    # Generate some dummy data.
    N = 5000
    x = np.linspace(0,10,N)
    
    # Spawn a number of processes equal to the number of processors
    # you have (detected automatically). Doing Pool(5) spawns 5.
    pool = Pool()
    nproc = pool._processes # Get number of processes that spawned.

    # np.array_split divides an array into nproc "even" parts.
    ID = np.arange(N)
    ID_chunks = np.array_split(ID,nproc) # Array of indices
    
    # Assign each processes a task. We will use as input our full
    # data array x and the indices of x for f to compute.
    proc = [None]*nproc
    for i in range(0,nproc):
        proc[i] = pool.apply_async(f,args=(x,ID_chunks[i]))
        
    # Tell the processes to run their tasks. Processes can finish
    # out of order, so we carefully store the results by having f
    # return the indices over which it operated to get its result.
    y = np.zeros(N)    
    start = time.time()
    for i in range(0,nproc):
        ytemp, ID_chunk = proc[i].get()
        y[ID_chunk] = ytemp
    print(time.time()-start) # 0.186653852463
    pool.close() # Close the processes

    ID = np.arange(N)
    start = time.time()
    y_slow,dummy = f(x,ID)
    print(time.time()-start) # 4.84067487717 (~25x slower)

    print(np.array_equal(y,y_slow)) # True


    
