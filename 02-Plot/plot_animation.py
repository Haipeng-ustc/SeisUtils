import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def plot_wavefield(wavefield):
    ''' Plot wavefield

    Parameters
    ----------
    wavefield : 3D numpy array
        Wavefield to plot (nt, nx, ny)
        nt = number of time steps
        nx = number of grid points in x-direction
        ny = number of grid points in y-direction
    '''

    fig = plt.figure()
    ims = []

    for i in range(wavefield.shape[0]):
        caxis = wavefield.max() * 0.2
        im = plt.imshow(wavefield[i].T, vmin = -caxis, vmax = caxis, aspect = 1, cmap='seismic', animated=False)
        ims.append([im])
    
    ani = animation.ArtistAnimation(fig, ims, interval=500, blit=True,repeat=True, repeat_delay=0)
    plt.close()

    return ani


## Use plot_wavefield in Jupyter notebook
from IPython.display import HTML

wavefield = np.random.rand(100, 100, 100)
wavefield_ani = plot_wavefield(wavefield)
HTML(wavefield_ani.to_jshtml())
