import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interpn


def plot_hist2d(x, y, bins=40, mean_absolute_error=None, rmse=None, res=None, **kwargs):
    fig, ax = plt.subplots()

    from matplotlib.colors import LogNorm
    plt.hist2d(x, y, bins=bins, norm=LogNorm())

    plt.tick_params(labelsize=15)
    plt.xticks(size=20, family='Times New Roman')
    plt.yticks(size=20, family='Times New Roman')
    cbar = plt.colorbar(shrink=1)
    cbar.ax.set_ylabel('Antal datapunkter', size=20, family='Times New Roman', position=(1, 1), rotation='vertical')

    ax.set_aspect('equal', 'box')
    plt.xlim([0, 2.5])
    plt.ylim([0, 2.5])
    plt.xlabel("Målvärde [m]", size=20, family='Times New Roman')
    plt.ylabel("Prediktion [m]", size=20, family='Times New Roman')

    data, x_e, y_e = np.histogram2d(x, y, bins=bins)
    z = interpn((0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])), data, np.vstack([x, y]).T,
                method="splinef2d", bounds_error=False)

    # To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    x_eq_y = np.linspace(0, x.max())
    plt.plot(x_eq_y, x_eq_y, color='Orange', label='x=y')
    sorted_pairs = sorted((i, j) for i, j in zip(x, y))
    x_sorted = []
    y_sorted = []
    for i, j in sorted_pairs:
        x_sorted.append(i)
        y_sorted.append(j)

    # change this to e.g 3 to get a polynomial of degree 3 to fit the curve
    order_of_the_fitted_polynomial = 1
    p30 = np.poly1d(np.polyfit(x_sorted, y_sorted, order_of_the_fitted_polynomial))
    plt.plot(x_sorted, p30(x_sorted), color='Red', label='linjär anpassning')
    if mean_absolute_error is not None:
        fig_text = f"MAE={mean_absolute_error:.3f}m"
        plt.plot([], [], ' ', label=fig_text)
    if rmse is not None:
        fig_text = f"RMSE={rmse:.3f}m"
        plt.plot([], [], ' ', label=fig_text)
    if res is not None:
        fig_text = f"R={res:.3f}"
        plt.plot([], [], ' ', label=fig_text)

    ax.legend()
    plt.show()
