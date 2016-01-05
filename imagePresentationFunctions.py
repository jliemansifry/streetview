import matplotlib
import sys
import numpy as np


def make_cmap(colors, position=None, bit=False):
    '''
    INPUT:  (1) list of RGB tuples to make the colormap from
    OUTPUT: (1) matplotlib colormap
    '''
    bit_rgb = np.linspace(0, 1, 256)
    if position is None:
        position = np.linspace(0, 1, len(colors))
    else:
        if len(position) != len(colors):
            sys.exit("position length must be the same as colors")
        elif position[0] != 0 or position[-1] != 1:
            sys.exit('position must start with 0 and end with 1')
    if bit:
        for i in range(len(colors)):
            colors[i] = (bit_rgb[colors[i][0]],
                         bit_rgb[colors[i][1]], bit_rgb[colors[i][2]])
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
    cont_cmap = make_cmap(colors, bit=True)
    return cont_cmap
