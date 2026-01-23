import numpy as np

def sim_calculate(objs: list) -> np.ndarray:
    """
    Calculate similarity matrix between multiple tasks.

    Parameters
    ----------
    objs : list
        List of objective values for each task, length is nt (number of tasks).
        objs[i] is a 2D array with shape (n, 1), representing objective values
        of n samples on the i-th task.

    Returns
    -------
    sim : np.ndarray
        Similarity matrix, shape (nt, nt)
    """
    nt = len(objs)
    n = objs[0].shape[0]

    # Convert list to n*nt matrix for easier computation
    # Each column represents all sample values of one task
    obj_matrix = np.hstack([objs[i] for i in range(nt)])  # shape: (n, nt)

    # Initialize similarity matrix
    sim = np.zeros((nt, nt))

    # Calculate similarity between each pair of tasks
    for i in range(nt):
        for j in range(nt):
            if i == j:
                sim[i, j] = 1.0
            else:
                # Use Pearson correlation coefficient as similarity
                corr = np.corrcoef(obj_matrix[:, i], obj_matrix[:, j])[0, 1]
                sim[i, j] = corr

    return sim