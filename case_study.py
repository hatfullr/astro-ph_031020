import numpy as np
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool
from scipy.integrate import simps

#def density(x,y,z):
    

def create_gas_cloud(Nx,Ny,Nz):
    data = np.zeros(shape=(Nx,Ny,Nz,5))
    
    data[:,:,:,0] = np.reshape(np.repeat(np.repeat(np.linspace(-1.,1.,Nx),Ny),Nz),(Nx,Ny,Nz))
    data[:,:,:,1] = np.reshape(np.repeat(np.tile(np.linspace(-1.,1.,Ny),Nx),Nz),(Nx,Ny,Nz))
    data[:,:,:,2] = np.reshape(np.tile(np.tile(np.linspace(-1.,1.,Nz),Ny),Nx),(Nx,Ny,Nz))
    data[:,:,:,3] = np.sum(data[:,:,:,0:4]**2.,axis=-1)

    idx = np.where(data[:,:,:,3] <= 1.)
    data[:,:,:,4][idx] = data[:,:,:,3][idx]**(-2.)

    return data

def coldens(z,rho,x_idxs,Ny):
    # x_idxs is a 1D array of the indices in the x array to calculate for
    result = np.zeros(shape=(len(x_idxs),Ny))
    for i in range(0,len(x_idxs)):
        for j in range(0,Ny):
            result[i][j] = simps(rho[x_idxs[i],j,:],x=z[x_idxs[i],j,:])
    return result,x_idxs

def parallel(data,nprocs):
    x = data[:,:,:,0]
    y = data[:,:,:,1]
    z = data[:,:,:,2]
    rho = data[:,:,:,4]

    x_uniq = np.unique(x)
    y_uniq = np.unique(y)

    Nx = len(x_uniq)
    Ny = len(y_uniq)

    colden = np.zeros(shape=(Nx,Ny))

    xidxs = np.arange(Nx)

    pool = Pool(nprocs)

    # Just split up processes for the x direction

    x_chunks = np.array_split(xidxs,nprocs)

    procs = [None]*nprocs
    for i in range(0,nprocs):
        procs[i] = pool.apply_async(coldens,args=(z,rho,x_chunks[i],Ny))

    start = time.time()
    for i in range(0,nprocs):
        rho_temp, idxs = procs[i].get()
        colden[idxs] = rho_temp
    finish = time.time() - start
    pool.close()


    """ Uncomment to prove the method works
    if Nx > 90:
        xmin = np.amin(x)
        xmax = np.amax(x)
        ymin = np.amin(y)
        ymax = np.amax(y)
        plt.imshow(np.log10(colden),extent=(xmin,xmax,ymin,ymax))
        plt.subplots_adjust(left=0.05,right=0.95,top=0.95,bottom=0.05)
        ax = plt.gca()
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.set_title("log column density")
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        plt.savefig("gas_cloud.png")
        plt.show()
        quit()
    """
    
    return finish

    

if __name__ == "__main__":
    #start = 100 # Gives communication times scaling image
    #stop = 500
    #step = 100
    start = 10 # Gives standard scaling image
    stop = 110
    step = 10
    Ns = np.zeros(shape=((stop-start)/step,3),dtype=int)
    Ns[:,0] = np.arange(start,stop,step,dtype=int)
    Ns[:,1] = Ns[:,0]
    Ns[:,2] = Ns[:,0]

    nprocs = [ 1,2,3,4,5,6,7,8 ]
    resolutions = Ns[:,0]*Ns[:,1]*Ns[:,2]

    datas = [None]*len(Ns)
    for i in range(0,len(Ns)):
        datas[i] = create_gas_cloud(Ns[i][0],Ns[i][1],Ns[i][2])
    
    for j in range(0,len(nprocs)):
        print("nprocs = ",nprocs[j])
        times = [None]*len(Ns)
        for i in range(0,len(Ns)):
            times[i] = parallel(datas[i],nprocs[j])

        plt.plot(np.log10(times),np.log10(resolutions),label="$N_P$="+str(nprocs[j]))
        
    plt.legend()
    plt.ylabel("log$_{10}$ Grid Resolution")
    plt.xlabel("log$_{10}$ Time Elapsed / s")
    plt.subplots_adjust(top=0.95,right=0.95)
    plt.savefig("Scaling.png")
    plt.show()
