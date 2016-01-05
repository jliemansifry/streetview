import matplotlib
import numpy as np


def make_cmap(colors):
    '''
    INPUT:  (1) list of RGB tuples to make the colormap from
    OUTPUT: (1) matplotlib colormap
    '''
    position = np.linspace(0, 1, len(colors))
    colors = [(color[0] / 255., color[1] / 255., color[2] / 255.)
              for color in colors]
    cdict = {'red': [], 'green': [], 'blue': []}
    for pos, color in zip(position, colors):
        cdict['red'].append((pos, color[0], color[0]))
        cdict['green'].append((pos, color[1], color[1]))
        cdict['blue'].append((pos, color[2], color[2]))
    cmap = matplotlib.colors.LinearSegmentedColormap('my_colormap', cdict, 256)
    return cmap


def make_cmyk_greyscale_continuous_cmap():
    '''
    INPUT:  None
    OUTPUT: (1) matplotlib colormap

    This series of colors makes a nice colormap that is good for
    'heatmap' style plots. It is continuous in cmyk, rgb, and greyscale.
    The new matplotlib has colormaps that are continuous, but they all
    start from a dark color. The colors below start from white.
    '''
    colors = [(255, 255, 255), (210, 253, 255), (184, 252, 255),
              (192, 244, 204), (155, 255, 145), (210, 200, 12),
              (230, 180, 7), (236, 124, 13), (233, 100, 25),
              (228, 30, 45), (198, 0, 46), (103, 0, 51)]
    cont_cmap = make_cmap(colors)
    return cont_cmap
