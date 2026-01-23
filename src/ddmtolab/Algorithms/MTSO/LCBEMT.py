"""
Lower Confidence Bound-based Evolutionary Multi-Tasking (LCB-EMT)

This module implements LCB-EMT for multi-task optimization with knowledge transfer
via Transfer Gaussian Process and similarity-based lower confidence bound.

References
----------
    [1] Wang, Zhenzhong, et al. "Evolutionary multitask optimization with lower confidence \
        bound-based solution selection strategy." IEEE Transactions on Evolutionary Computation (2024).

Notes
-----
Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2025.12.19
Version: 1.0
"""
from tqdm import tqdm
import time
import torch
from ddmtolab.Methods.Algo_Methods.tgp import TGPRegression
from ddmtolab.Methods.Algo_Methods.algo_utils import *


def tgp_slcb_knowledge_transfer(target_task_idx, source_task_idx, decs, objs, dims, n_transfer):
    """
    Perform knowledge transfer from source task to target task using TGP and SLCB.

    This function implements the Transfer Gaussian Process (TGP) with Similarity-based
    Lower Confidence Bound (SLCB) strategy for selecting promising solutions to transfer
    between tasks in multi-task optimization.

    Parameters
    ----------
    target_task_idx : int
        Index of the target task (receiving knowledge)
    source_task_idx : int
        Index of the source task (providing knowledge)
    decs : List[np.ndarray]
        Decision variables for all tasks. Each element is a 2D array of shape (n_solutions, n_dims)
    objs : List[np.ndarray]
        Objective values for all tasks. Each element is a 1D or 2D array of objective values
    dims : List[int]
        Dimensions for all tasks
    n_transfer : int
        Number of solutions to transfer

    Returns
    -------
    np.ndarray
        Transferred solutions (decision variables) to add to target task offspring.
        Shape: (n_transfer, target_dim)

    Notes
    -----
    The function performs the following steps:
    1. Dimension alignment between source and target tasks
    2. Objective normalization for each task
    3. TGP model training with task indicators
    4. SLCB computation for solution selection
    5. Probabilistic selection of transfer solutions
    """
    target_dim = dims[target_task_idx]
    source_dim = dims[source_task_idx]

    # Step 1: Dimension alignment - adjust source task dimensions to match target task
    adjusted_source_decs = decs[source_task_idx].copy()

    if source_dim < target_dim:
        # Zero-padding for smaller dimensions
        padding = np.zeros((len(adjusted_source_decs), target_dim - source_dim))
        adjusted_source_decs = np.hstack([adjusted_source_decs, padding])
    elif source_dim > target_dim:
        # Truncation for larger dimensions
        adjusted_source_decs = adjusted_source_decs[:, :target_dim]

    # Step 2: Prepare training data
    target_decs = decs[target_task_idx]
    target_objs = objs[target_task_idx]
    source_objs = objs[source_task_idx]

    # Step 3: Normalize objectives for each task based on their own population
    # Normalize target task objectives
    target_objs_min = np.min(target_objs)
    target_objs_max = np.max(target_objs)
    if target_objs_max - target_objs_min > 1e-10:  # Avoid division by zero
        target_objs_norm = (target_objs - target_objs_min) / (target_objs_max - target_objs_min)
    else:
        target_objs_norm = np.zeros_like(target_objs)

    # Normalize source task objectives
    source_objs_min = np.min(source_objs)
    source_objs_max = np.max(source_objs)
    if source_objs_max - source_objs_min > 1e-10:  # Avoid division by zero
        source_objs_norm = (source_objs - source_objs_min) / (source_objs_max - source_objs_min)
    else:
        source_objs_norm = np.zeros_like(source_objs)

    # Step 4: Add task indicators (0 for target task, 1 for source task)
    target_task_ind = np.zeros((len(target_decs), 1))
    source_task_ind = np.ones((len(adjusted_source_decs), 1))

    # Step 5: Combine target and source data
    X_train = np.vstack([
        np.hstack([target_decs, target_task_ind]),
        np.hstack([adjusted_source_decs, source_task_ind])
    ])
    y_train = np.concatenate([target_objs_norm, source_objs_norm])

    # Step 6: Convert to PyTorch tensors
    X_train_torch = torch.tensor(X_train, dtype=torch.float32)
    y_train_torch = torch.tensor(y_train.flatten(), dtype=torch.float32)

    # Step 7: Train Transfer Gaussian Process model
    tgp_model = TGPRegression()
    tgp_model.train(X_train_torch, y_train_torch, n_restarts=3, verbose=False)

    # Step 8: Get learned inter-task similarity (lambda)
    lambda_gen = tgp_model.get_task_correlation()

    # Step 9: Prepare candidate solutions for transfer
    # Solution set S = {x' | x' = M(x), x in source task}
    # Here we use identity mapping M(x) = x
    S = adjusted_source_decs.copy()

    # Step 10: Predict mean (f) and uncertainty (sigma) for candidate solutions
    S_torch = torch.tensor(S, dtype=torch.float32)
    with torch.no_grad():
        mu_S, cov_S = tgp_model.predict(S_torch)
        sigma_S = torch.sqrt(torch.diag(cov_S))

    mu_S_np = mu_S.cpu().numpy()
    sigma_S_np = sigma_S.cpu().numpy()

    # Step 11: Compute Similarity-based Lower Confidence Bound (SLCB)
    SLCB = mu_S_np - lambda_gen * sigma_S_np

    # Step 12: Compute transfer probability for each solution
    exp_neg_SLCB = np.exp(-SLCB)
    p_transfer_unnorm = lambda_gen * exp_neg_SLCB / np.sum(exp_neg_SLCB)
    exp_p_transfer = np.exp(p_transfer_unnorm)
    p_transfer = exp_p_transfer / np.sum(exp_p_transfer)

    if np.any(np.isnan(p_transfer)) or np.any(np.isinf(p_transfer)):
        p_transfer = np.ones(len(S)) / len(S)

    # Step 13: Sample solutions according to probability distribution
    n_transfer_actual = min(n_transfer, len(S))
    selected_indices = np.random.choice(len(S), size=n_transfer_actual, replace=False, p=p_transfer)

    # Step 14: Extract selected solutions for transfer
    transferred_decs = S[selected_indices]

    return transferred_decs


class LCBEMT:
    """
    Lower Confidence Bound-based Evolutionary Multi-Tasking.

    This algorithm uses Transfer Gaussian Process (TGP) to model task relationships
    and employs a Similarity-based Lower Confidence Bound (SLCB) strategy to select
    promising solutions for knowledge transfer between tasks.

    Attributes
    ----------
    algorithm_information : dict
        Dictionary containing algorithm capabilities and requirements
    """

    algorithm_information = {
        'n_tasks': '2-K',
        'dims': 'unequal',
        'objs': 'equal',
        'n_objs': '1',
        'cons': 'unequal',
        'n_cons': '0-C',
        'expensive': 'False',
        'knowledge_transfer': 'True',
        'n': 'unequal',
        'max_nfes': 'unequal'
    }

    @classmethod
    def get_algorithm_information(cls, print_info=True):
        return get_algorithm_information(cls, print_info)

    def __init__(self, problem, n=None, max_nfes=None, Nt=10, TGap=20, muc=2, mum=5, save_data=True,
                 save_path='./TestData', name='LCBEMT_test', disable_tqdm=True):
        """
        Initialize LCB-EMT algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n : int or List[int], optional
            Population size per task (default: 100)
        max_nfes : int or List[int], optional
            Maximum number of function evaluations per task (default: 10000)
        Nt : int, optional
            Number of solutions to transfer per knowledge transfer operation (default: 20)
        TGap : int, optional
            Transfer interval in generations (default: 20)
        muc : float, optional
            Distribution index for simulated binary crossover (SBX) (default: 2.0)
        mum : float, optional
            Distribution index for polynomial mutation (PM) (default: 5.0)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './TestData')
        name : str, optional
            Name for the experiment (default: 'LCBEMT_test')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n = n if n is not None else 100
        self.max_nfes = max_nfes if max_nfes is not None else 10000
        self.Nt = Nt
        self.TGap = TGap
        self.muc = muc
        self.mum = mum
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the LCB-EMT algorithm.

        Returns
        -------
        Results
            Optimization results containing decision variables, objectives, and runtime
        """
        start_time = time.time()
        problem = self.problem
        nt = problem.n_tasks
        dims = problem.dims
        n_per_task = par_list(self.n, nt)
        max_nfes_per_task = par_list(self.max_nfes, nt)

        # Initialize population and evaluate for each task
        decs = initialization(problem, n_per_task)
        objs, cons = evaluation(problem, decs)
        nfes_per_task = n_per_task.copy()

        # Store initial populations
        initial_decs = [d.copy() for d in decs]
        gen = 1

        all_decs, all_objs, all_cons = init_history(decs, objs, cons)

        pbar = tqdm(total=sum(max_nfes_per_task), initial=sum(n_per_task),
                    desc=f"{self.name}", disable=self.disable_tqdm)

        while sum(nfes_per_task) < sum(max_nfes_per_task):
            # Identify active tasks (tasks that haven't exhausted their evaluation budget)
            active_tasks = [i for i in range(nt) if nfes_per_task[i] < max_nfes_per_task[i]]
            if not active_tasks:
                break

            for i in active_tasks:
                # Generate offspring using genetic algorithm operators
                off_decs = ga_generation(decs[i], self.muc, self.mum)

                # Knowledge transfer using TGP and SLCB
                if self.Nt > 0 and gen % self.TGap == 0 and len(active_tasks) >= 2:
                    # Select source task using circular indexing: mod(k+1, K)
                    k = i  # current task index (target task)
                    K = len(active_tasks)  # number of active tasks
                    source_idx_in_active = (active_tasks.index(k) + 1) % K
                    source_task = active_tasks[source_idx_in_active]

                    # Perform knowledge transfer using independent function
                    transferred_decs = tgp_slcb_knowledge_transfer(
                        target_task_idx=i,
                        source_task_idx=source_task,
                        decs=decs,
                        objs=objs,
                        dims=dims,
                        n_transfer=self.Nt
                    )

                    # Add transferred solutions to offspring population
                    off_decs = np.vstack([off_decs, transferred_decs])
                else:
                    off_decs = off_decs.copy()

                # Evaluate offspring population
                offobjs, offcons = evaluation_single(problem, off_decs, i)

                # Merge parent and offspring populations
                objs[i], decs[i], cons[i] = vstack_groups(
                    (objs[i], offobjs), (decs[i], off_decs), (cons[i], offcons)
                )

                # Environmental selection: keep the best n individuals
                index = selection_elit(objs[i], n_per_task[i], cons[i])
                objs[i], decs[i], cons[i] = select_by_index(index, objs[i], decs[i], cons[i])

                # Update evaluation counter and progress bar
                nfes_per_task[i] += off_decs.shape[0]
                pbar.update(off_decs.shape[0])

                # Store optimization history
                append_history(all_decs[i], decs[i], all_objs[i], objs[i], all_cons[i], cons[i])

            gen += 1

        pbar.close()
        runtime = time.time() - start_time

        # Build and save results
        results = build_save_results(all_decs=all_decs, all_objs=all_objs, runtime=runtime, max_nfes=nfes_per_task,
                                     all_cons=all_cons, bounds=problem.bounds, save_path=self.save_path,
                                     filename=self.name, save_data=self.save_data)

        return results