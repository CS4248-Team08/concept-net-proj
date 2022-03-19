import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def plot_heatmap(attentions):
    num_layers = len(attentions)

    cols = 2
    rows = int(num_layers/cols)

    fig, axes = plt.subplots(rows,cols, figsize = (14,30))
    axes = axes.flat
    
    for layer_index, att in enumerate(attentions):
        sns.heatmap(att, vmin = 0, vmax = 1,ax = axes[layer_index])
        axes[layer_index].set_title(f'layer - {layer_index} ' )