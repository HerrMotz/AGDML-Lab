import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st



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



def plot_density(X : np.ndarray, xmax : np.ndarray = None, xmin : np.ndarray = None, ax : plt.axes = None):
    '''
    Convenience function for plotting density of samples.

    @Params:
        X... features (shape n_samples x 2)
        xmax... array of length 2 with maximum values for each dimension
        xmin... array of length 2 with minimum values for each dimension
        ax... matplotlib axis on which to plot
    '''
    
    if ax is None:
        ax = plt.gca()

    # Extract x and y
    x1 = X[:, 0]
    x2 = X[:, 1]# Define the borders

    if xmax is None:
        xmax = np.max(X, axis = 0)
    
    if xmin is None:
        xmin = np.min(X, axis = 0)


    deltaX1 = (xmax[0] - xmin[0])/10
    deltaX2 = (xmax[1] - xmin[1])/10
    x1min = xmin[0] - deltaX1
    x1max = xmax[0] + deltaX1
    x2min = xmin[1] - deltaX2
    x2max = xmax[1] + deltaX2
    xx1, xx2 = np.mgrid[x1min:x1max:100j, x2min:x2max:100j]
    positions = np.vstack([xx1.ravel(), xx2.ravel()])
    values = np.vstack([x1, x2])
    kernel = st.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx1.shape)
    
    ax.set_xlim(x1min, x1max)
    ax.set_ylim(x2min, x2max)
    
    ax.contourf(xx1, xx2, f, cmap='Reds', alpha = 1.0, vmin = 0.0)

    ax.set_xticks([])
    ax.set_yticks([])