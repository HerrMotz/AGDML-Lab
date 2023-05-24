import numpy as np
import matplotlib.pyplot as plt


def plot_dec_boundary(X : np.ndarray, y : np.ndarray, estimator : callable, ax : plt.axes = None):
    '''
    Convenience function for plotting decision boundary of binary classificator.

    @Params:
        X... features (shape n_samples x feature_dim)
        y... labels (shape n_samples)
        estimator... instance of a class with .fit() and .predict() for binary classification
        ax... matplotlib axis on which to plot
    '''
    if ax is None:
        ax = plt.gca()

    # make these smaller to increase the resolution
    dx, dy = 0.01, 0.01
    # generate grids + labels
    x1, x2 = np.mgrid[
        slice(np.min(X[:, -2]), np.max(X[:, -2]) + dy, dy),
        slice(np.min(X[:, -1]), np.max(X[:, -1]) + dx, dx),
    ]
    points = np.stack([np.ones(np.prod(x1.shape)), x1.flatten(), x2.flatten()]).T
    if X.shape[1] == 2:
        points = points[:,1:]
    labels = estimator.predict(points).reshape(x1.shape)

    # plot points + areas
    cmap = plt.get_cmap("bwr")
    ax.contourf(
        x1,
        x2,
        labels,
        cmap=cmap,
        alpha=0.4,
        vmin=0,
        vmax=1
    )
    #plt.colorbar()
    ax.scatter(X[:, -2], X[:, -1], c=y, cmap="bwr", s=8)