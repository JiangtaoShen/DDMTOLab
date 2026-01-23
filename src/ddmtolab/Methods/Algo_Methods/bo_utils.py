import torch
import numpy as np
import gpytorch
from botorch.models import SingleTaskGP, MultiTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.models.transforms.outcome import Standardize
from botorch.acquisition.objective import ScalarizedPosteriorTransform
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import UpperConfidenceBound



def gp_build(
    decs: np.ndarray,
    objs: np.ndarray,
    data_type: torch.dtype = torch.float
) -> SingleTaskGP:
    """
    Build and fit a Single-Task Gaussian Process model.

    Parameters
    ----------
    decs : np.ndarray
        Historical decision variables, shape: (n_samples, dim)
    objs : np.ndarray
        Historical objective function values, shape: (n_samples,) or (n_samples, 1)
    data_type : torch.dtype, optional
        Data type for tensors (default: torch.float)

    Returns
    -------
    gp : SingleTaskGP
        Fitted Gaussian Process model
    """
    # Prepare training data
    train_X = torch.tensor(decs, dtype=data_type)
    train_Y = torch.tensor(-objs, dtype=data_type)
    if train_Y.dim() == 1:
        train_Y = train_Y.unsqueeze(-1)

    # Build and fit Gaussian Process model
    gp = SingleTaskGP(
        train_X=train_X,
        train_Y=train_Y,
        outcome_transform=Standardize(m=1)
    )
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)

    return gp


def gp_predict(
    gp: SingleTaskGP,
    test_X: np.ndarray,
    data_type: torch.dtype = torch.float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Predict objectives and uncertainties using a trained Gaussian Process model.

    Parameters
    ----------
    gp : SingleTaskGP
        Trained Gaussian Process model
    test_X : np.ndarray
        Test decision variables, shape: (n_candidates, dim)
    data_type : torch.dtype, optional
        Data type for tensors (default: torch.float)

    Returns
    -------
    pred_objs : np.ndarray
        Predicted objective values, shape: (n_candidates, 1)
    pred_std : np.ndarray
        Predicted standard deviations, shape: (n_candidates, 1)
    """
    # Convert to tensor
    test_X_tensor = torch.tensor(test_X, dtype=data_type)

    # Predict using the trained model
    gp.eval()
    with torch.no_grad():
        posterior = gp.posterior(test_X_tensor)
        pred_mean = posterior.mean
        pred_var = posterior.variance

    # Convert to numpy and negate for minimization
    pred_objs = -pred_mean.cpu().numpy()
    pred_std = torch.sqrt(pred_var).cpu().numpy()

    return pred_objs, pred_std


def mo_gp_build(decs, objs, data_type=torch.float):
    """
    Build Gaussian Process models for each objective in multi-objective optimization.

    Parameters
    ----------
    decs : np.ndarray
        Decision variables, shape (N, D) where N is the number of samples
        and D is the dimension of decision space.
    objs : np.ndarray
        Objective values, shape (N, M) where M is the number of objectives.
    data_type : torch.dtype, optional
        Data type for GP models (default: torch.float).

    Returns
    -------
    models : list
        List of trained GP models, one for each objective.
    """
    M = objs.shape[1]
    models = []
    for j in range(M):
        model = gp_build(decs, objs[:, j:j + 1], data_type)
        models.append(model)
    return models


def mo_gp_predict(models, x, data_type=torch.float, mse=False):
    """
    Predict objectives using trained GP models for multi-objective optimization.

    Parameters
    ----------
    models : list
        List of trained GP models (one per objective), as returned by mo_gp_build.
    x : np.ndarray
        Decision variables to predict, shape (N, D) where N is the number of
        samples and D is the dimension of decision space.
    data_type : torch.dtype, optional
        Data type for GP prediction (default: torch.float).
    mse : bool, optional
        If True, also return the Mean Squared Error (variance) of predictions.
        If False, only return predicted objective values (default: False).

    Returns
    -------
    pred_objs : np.ndarray
        Predicted objective values, shape (N, M) where M is the number of objectives.
    pred_mse : np.ndarray, optional
        Predicted MSE (variance) for each objective, shape (N, M).
        Only returned if mse=True.
    """
    N = x.shape[0]
    M = len(models)
    pred_objs = np.zeros((N, M))

    if mse:
        pred_mse = np.zeros((N, M))
        for j in range(M):
            pred, std = gp_predict(models[j], x, data_type)
            pred_objs[:, j] = pred.flatten()
            pred_mse[:, j] = (std ** 2).flatten()
        return pred_objs, pred_mse
    else:
        for j in range(M):
            pred, _ = gp_predict(models[j], x, data_type)
            pred_objs[:, j] = pred.flatten()
        return pred_objs


def bo_next_point(
    dim_i: int,
    decs_i: np.ndarray,
    objs_i: np.ndarray,
    data_type: torch.dtype = torch.float
) -> np.ndarray:
    """
    Get the next sampling point using Bayesian Optimization

    Parameters
    ----------
    dim_i : int
        Dimension of decision variables
    decs_i : np.ndarray
        Historical decision variables, shape: (n_samples, dim_i)
    objs_i : np.ndarray
        Historical objective function values, shape: (n_samples,) or (n_samples, 1)
    data_type : torch.dtype, optional
        Data type, default is torch.float

    Returns
    -------
    candidate_np : np.ndarray
        Next sampling point, shape: (1, dim_i)
    """
    # Define search bounds [0, 1]^dim_i
    bounds = torch.stack([
        torch.zeros(dim_i, dtype=data_type),
        torch.ones(dim_i, dtype=data_type)
    ], dim=0)

    # Prepare training data for Gaussian Process
    train_X = torch.tensor(decs_i, dtype=data_type)
    train_Y = torch.tensor(-objs_i, dtype=data_type)
    if train_Y.dim() == 1:
        train_Y = train_Y.unsqueeze(-1)

    # Build and fit Gaussian Process model
    gp = SingleTaskGP(
        train_X=train_X,
        train_Y=train_Y,
        # outcome_transform=Standardize(m=1)
    )
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)

    # Optimize Log Expected Improvement acquisition function
    best_f = train_Y.max()
    logEI = LogExpectedImprovement(model=gp, best_f=best_f)
    candidate, _ = optimize_acqf(
        logEI,
        bounds=bounds,
        q=1,
        num_restarts=5,
        raw_samples=20
    )

    # Convert to numpy array and return
    candidate_np = candidate.detach().cpu().numpy()

    return candidate_np


def bo_next_point_lcb(
    dim_i: int,
    decs_i: np.ndarray,
    objs_i: np.ndarray,
    data_type: torch.dtype = torch.float,
    kappa: float = 2.0
) -> tuple:
    """
    Get the next sampling point using Bayesian Optimization with LCB acquisition

    Parameters
    ----------
    dim_i : int
        Dimension of decision variables
    decs_i : np.ndarray
        Historical decision variables, shape: (n_samples, dim_i)
    objs_i : np.ndarray
        Historical objective function values, shape: (n_samples,) or (n_samples, 1)
    data_type : torch.dtype, optional
        Data type, default is torch.float
    kappa : float, optional
        Exploration weight for LCB, default is 2.0

    Returns
    -------
    candidate_np : np.ndarray
        Next sampling point, shape: (1, dim_i)
    gp : SingleTaskGP
        Fitted Gaussian Process model
    """

    # Define search bounds [0, 1]^dim_i
    bounds = torch.stack([
        torch.zeros(dim_i, dtype=data_type),
        torch.ones(dim_i, dtype=data_type)
    ], dim=0)

    # Prepare training data for Gaussian Process
    train_X = torch.tensor(decs_i, dtype=data_type)
    train_Y = torch.tensor(-objs_i, dtype=data_type)  # 取负以最大化
    if train_Y.dim() == 1:
        train_Y = train_Y.unsqueeze(-1)

    # Build and fit Gaussian Process model
    gp = SingleTaskGP(train_X=train_X, train_Y=train_Y)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)

    # Optimize Upper Confidence Bound (UCB) acquisition function
    UCB = UpperConfidenceBound(model=gp, beta=kappa**2)  # beta = kappa^2
    candidate, _ = optimize_acqf(
        UCB,
        bounds=bounds,
        q=1,
        num_restarts=5,
        raw_samples=20
    )

    # Convert to numpy array and return
    candidate_np = candidate.detach().cpu().numpy()

    return candidate_np, gp


def mtgp_build(
        decs: list[np.ndarray],
        objs: list[np.ndarray],
        dims: list[int],
        data_type: torch.dtype = torch.float
) -> MultiTaskGP:
    """
    Build a Multi-Task Gaussian Process model.

    Parameters
    ----------
    decs : list[np.ndarray]
        List of decision variable matrices for each task
    objs : list[np.ndarray]
        List of objective value matrices for each task
    dims : list[int]
        List of dimensionalities for each task
    std_params : list[dict] | None
        Standardization parameters for each task. If None, objectives are not standardized.
    data_type : torch.dtype
        Data type for tensors

    Returns
    -------
    mtgp : MultiTaskGP
        Fitted Multi-Task GP model
    """
    nt = len(decs)
    max_dim = max(dims)
    train_X_list = []
    train_Y_list = []
    train_i_list = []

    task_range = torch.linspace(0, 1, nt).unsqueeze(-1)

    for i in range(nt):
        task_data = torch.tensor(decs[i], dtype=data_type)
        # Use negative objectives for maximization
        task_obj = torch.tensor(-objs[i], dtype=data_type)
        task_idx = torch.full(
            (task_data.shape[0], 1),
            task_range[i].item(),
            dtype=data_type
        )

        # Pad with zeros if current task has fewer dimensions
        if dims[i] < max_dim:
            padding = torch.zeros(task_data.shape[0], max_dim - dims[i], dtype=data_type)
            task_data = torch.cat([task_data, padding], dim=1)

        train_X_list.append(task_data)
        train_Y_list.append(task_obj)
        train_i_list.append(task_idx)

    train_X = torch.cat(train_X_list, dim=0)
    train_Y = torch.cat(train_Y_list, dim=0)
    train_i = torch.cat(train_i_list, dim=0)
    train_X_with_task = torch.cat([train_X, train_i], dim=1)

    mtgp = MultiTaskGP(
        train_X=train_X_with_task,
        train_Y=train_Y,
        task_feature=-1,
    )

    mll = ExactMarginalLogLikelihood(mtgp.likelihood, mtgp)
    fit_gpytorch_mll(mll)

    return mtgp


def mtgp_predict(
    mtgp: MultiTaskGP,
    off_decs: np.ndarray,
    task_id: int,
    dims: list[int],
    nt: int,
    obj_min_vals: list[float] | None = None,
    obj_max_vals: list[float] | None = None,
    data_type: torch.dtype = torch.float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Use Multi-Task GP to predict objectives and uncertainties for candidate solutions.

    Parameters
    ----------
    mtgp : MultiTaskGP
        Trained Multi-Task Gaussian Process model
    off_decs : np.ndarray
        Candidate decision variables, shape (n_candidates, dim)
    task_id : int
        Task index for prediction
    dims : list[int]
        List of dimensionalities for each task
    nt : int
        Total number of tasks
    obj_min_vals : list[float] | None
        Minimum objective values for each task (for denormalization)
    obj_max_vals : list[float] | None
        Maximum objective values for each task (for denormalization)
    data_type : torch.dtype, optional
        Data type for tensors (default: torch.float)

    Returns
    -------
    pred_objs : np.ndarray
        Predicted objective values, shape (n_candidates, 1)
    pred_std : np.ndarray
        Predicted standard deviations, shape (n_candidates, 1)
    """
    # Convert to tensor and pad dimensions if necessary
    test_X = torch.tensor(off_decs, dtype=data_type)
    max_dim = max(dims)
    if dims[task_id] < max_dim:
        padding = torch.rand(test_X.shape[0], max_dim - dims[task_id], dtype=data_type)
        test_X = torch.cat([test_X, padding], dim=1)

    # Append task index as the last feature
    task_range = torch.linspace(0, 1, nt)
    task_idx = torch.full((test_X.shape[0], 1), task_range[task_id].item(), dtype=data_type)
    test_X_with_task = torch.cat([test_X, task_idx], dim=1)

    # Predict using the trained model
    mtgp.eval()
    with torch.no_grad():
        posterior = mtgp.posterior(test_X_with_task)
        pred_mean = posterior.mean
        pred_var = posterior.variance

    # Convert to numpy and negate for minimization
    pred_objs = -pred_mean.cpu().numpy()
    pred_std = torch.sqrt(pred_var).cpu().numpy()

    # Denormalize if min/max values are provided
    if obj_min_vals is not None and obj_max_vals is not None:
        min_val = obj_min_vals[task_id]
        max_val = obj_max_vals[task_id]
        range_val = max_val - min_val
        if range_val < 1e-10:
            range_val = 1.0
        pred_objs = pred_objs * range_val + min_val
        pred_std = pred_std * range_val

    return pred_objs, pred_std


def mtgp_task_corr(
    mtgp: MultiTaskGP
) -> np.ndarray:
    """
    Extract task correlation matrix from multi-task Gaussian process model.

    Parameters
    ----------
    mtgp : MultiTaskGP
        Trained Multi-Task Gaussian Process model

    Returns
    -------
    task_corr : np.ndarray
        Task correlation matrix (normalized covariance matrix)
    """
    # Set model to evaluation mode
    mtgp.eval()

    # Find the IndexKernel from the covariance module
    index_kernel = None
    for kernel in mtgp.covar_module.kernels:
        if isinstance(kernel, gpytorch.kernels.IndexKernel):
            index_kernel = kernel
            break

    # Extract task correlations without gradient computation
    with torch.no_grad():
        covar_factor = index_kernel.covar_factor
        task_covar = (covar_factor @ covar_factor.T).cpu().numpy()

    # Normalize covariance matrix to get correlation matrix
    diag = np.sqrt(np.diag(task_covar))
    task_corr = task_covar / np.outer(diag, diag)

    return task_corr


def mtbo_next_point(
    mtgp: MultiTaskGP,
    task_id: int,
    objs: list[np.ndarray],
    dims: list[int],
    nt: int,
    data_type: torch.dtype = torch.float
) -> np.ndarray:
    """
    Get the next sampling point using Multi-Task Bayesian Optimization.

    Parameters
    ----------
    mtgp : MultiTaskGP
        Trained Multi-Task Gaussian Process model
    task_id : int
        Task index for which to find the next point
    objs : list[np.ndarray]
        List of objective value matrices for each task
    dims : list[int]
        List of dimensionalities for each task
    nt : int
        Total number of tasks
    data_type : torch.dtype, optional
        Data type for tensors (default: torch.float)

    Returns
    -------
    candidate_np : np.ndarray
        Next sampling point, shape: (1, dims[task_id])
    """
    # Define search bounds [0, 1]^max_dim with fixed task index
    max_dim = max(dims)
    lower_bound = torch.zeros(max_dim + 1, dtype=data_type)
    upper_bound = torch.ones(max_dim + 1, dtype=data_type)
    task_range = torch.linspace(0, 1, nt)
    lower_bound[-1] = task_range[task_id].item()
    upper_bound[-1] = task_range[task_id].item()
    bounds = torch.stack([lower_bound, upper_bound], dim=0)

    # Compute the best observed value for the current task
    best_f = torch.tensor(-objs[task_id], dtype=data_type).squeeze().max()

    # Build Log Expected Improvement acquisition function
    posterior_transform = ScalarizedPosteriorTransform(weights=torch.tensor([1.0], dtype=data_type))
    logEI = LogExpectedImprovement(
        model=mtgp,
        best_f=best_f,
        posterior_transform=posterior_transform
    )

    # Optimize acquisition function and extract decision variables
    candidate, _ = optimize_acqf(
        logEI,
        bounds=bounds,
        q=1,
        num_restarts=5,
        raw_samples=20
    )
    candidate_np = candidate[:, :dims[task_id]].detach().cpu().numpy()

    return candidate_np


