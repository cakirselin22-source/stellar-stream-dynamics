"""
Shared plotting utilities: consistent savefig behavior, colormap/norm helpers.
"""
import matplotlib.pyplot as plt


def savefig(fig, outfile, dpi=200, show=False):
    """
    Consistent save behavior across all plotting functions.
    """
    fig.savefig(outfile, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    return outfile



