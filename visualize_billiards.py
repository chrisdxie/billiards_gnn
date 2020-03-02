import os
import matplotlib.pyplot as plt
import numpy as np
import imageio



def plot_positions(xy, window_size=10):
    """ 

        @param xy: a [1 x n x 4] numpy array of object states
    """

    # default window_size=10, default ball_size=270.
    # Scale ball_size down by ratio**2 (matplotlib markersize is given in points**2)
    ratio = window_size / 10 
    ball_size = int(270/(ratio**2))

    fig_num = len(xy)
    n = len(xy[0])
    mydpi = 100
    fig = plt.figure(figsize=(128/mydpi, 128/mydpi))
    plt.xlim(0, window_size)
    plt.ylim(0, window_size)
    plt.xticks([])
    plt.yticks([])

    color = ['r', 'g', 'b', 'k', 'y', 'm', 'c']
    for i in range(fig_num):
        for j in range(n):
            plt.scatter(xy[i, j, 0], xy[i, j, 1],
                        c=color[j % len(color)], s=ball_size, alpha=(i+1)/fig_num)

    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return image


def animate(states, img_folder, prefix, window_size=10):
    """ Animate into a GIF

        @param states: a [T x n x d] numpy array (or torch tensor) of object states
        @param img_folder: where to write the GIFs
        @param prefix: filename prefix
    """
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)

    images = []
    for i in range(len(states)):
        images.append(plot_positions(states[i:i + 1], window_size=window_size))
    imageio.mimsave(img_folder+prefix+'.gif', images, fps=24)