import numpy as np


def sim_calculate(objs: list, method: str = 'pearson') -> np.ndarray:
    """
    Calculate similarity matrix between multiple tasks.

    Similarity is a correlation between the objective values that the same set
    of decision vectors attains on each pair of tasks; it therefore assumes the
    tasks were evaluated on a shared set of points (see the ``the_same=True``
    initialization).

    Parameters
    ----------
    objs : list
        List of objective values for each task, length is nt (number of tasks).
        objs[i] is a 2D array with shape (n, 1), representing objective values
        of n samples on the i-th task.
    method : {'pearson', 'spearman'}, optional
        Correlation type (default: 'pearson'). ``'spearman'`` correlates the
        objective *ranks* instead of the raw values, making it scale-invariant
        across tasks with very different objective magnitudes (as used by
        RAMTEA).

    Returns
    -------
    sim : np.ndarray
        Similarity matrix, shape (nt, nt). Diagonal entries are 1.0.
    """
    nt = len(objs)

    if method == 'spearman':
        # Correlate rank vectors (Pearson of ranks == Spearman correlation)
        cols = [np.argsort(np.argsort(objs[i].flatten())).astype(float) for i in range(nt)]
        feat_matrix = np.column_stack(cols)  # shape: (n, nt)
    elif method == 'pearson':
        feat_matrix = np.hstack([objs[i] for i in range(nt)])  # shape: (n, nt)
    else:
        raise ValueError(f"Unknown similarity method: {method!r}")

    sim = np.zeros((nt, nt))
    for i in range(nt):
        for j in range(nt):
            if i == j:
                sim[i, j] = 1.0
            else:
                corr = np.corrcoef(feat_matrix[:, i], feat_matrix[:, j])[0, 1]
                sim[i, j] = 0.0 if np.isnan(corr) else corr

    return sim
