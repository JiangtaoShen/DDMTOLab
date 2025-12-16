"""
Evolutionary Multitasking via Explicit Autoencoding (EMEA)

This module implements EMEA for multi-task optimization with knowledge transfer via autoencoding.

References
----------
    [1] Feng, Liang, et al. "Evolutionary multitasking via explicit autoencoding."
        IEEE transactions on cybernetics 49.9 (2018): 3457-3470.

Notes
-----
Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2025.10.25
Version: 1.0
"""
from tqdm import tqdm
import time
from Methods.Algo_Methods.algo_utils import *


class EMEA:
    """
    Evolutionary Multitasking via Explicit Autoencoding for multi-task optimization.

    Attributes
    ----------
    algorithm_information : dict
        Dictionary containing algorithm capabilities and requirements
    """

    algorithm_information = {
        'n_tasks': '2-K',
        'dims': 'unequal',
        'n_objs': '1',
        'n_cons': '1-C',
        'n': 'unequal',
        'max_nfes': 'unequal',
        'expensive': 'False',
        'knowledge_transfer': 'True'
    }

    @classmethod
    def get_algorithm_information(cls, print_info=True):
        """
        Get algorithm information.

        Parameters
        ----------
        print_info : bool, optional
            Whether to print information (default: True)

        Returns
        -------
        dict
            Algorithm information dictionary
        """
        return get_algorithm_information(cls, print_info)

    def __init__(self, problem, n=None, max_nfes=None, SNum=10, TGap=10, muc=2, mum=5, F=0.5, CR=0.6, save_data=True,
                 save_path='./TestData', name='EMEA_test', disable_tqdm=True):
        """
        Initialize EMEA algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n : int or List[int], optional
            Population size per task (default: 100)
        max_nfes : int or List[int], optional
            Maximum number of function evaluations per task (default: 10000)
        SNum : int, optional
            Number of transferred solutions (default: 10)
        TGap : int, optional
            Transfer interval in generations (default: 10)
        muc : float, optional
            Distribution index for simulated binary crossover (SBX) (default: 2.0)
        mum : float, optional
            Distribution index for polynomial mutation (PM) (default: 5.0)
        F : float, optional
            Scaling factor for DE mutation (default: 0.5)
        CR : float, optional
            Crossover rate for DE (default: 0.6)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './TestData')
        name : str, optional
            Name for the experiment (default: 'EMEA_test')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n = n if n is not None else 100
        self.max_nfes = max_nfes if max_nfes is not None else 10000
        self.SNum = SNum
        self.TGap = TGap
        self.muc = muc
        self.mum = mum
        self.F = F
        self.CR = CR
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the EMEA algorithm.

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

        # Store initial populations for domain adaptation
        initial_decs = [d.copy() for d in decs]
        gen = 1

        all_decs, all_objs, all_cons = init_history(decs, objs, cons)

        pbar = tqdm(total=sum(max_nfes_per_task), initial=sum(n_per_task),
                    desc=f"{self.name}", disable=self.disable_tqdm)

        while sum(nfes_per_task) < sum(max_nfes_per_task):
            # Skip tasks that have exhausted their evaluation budget
            active_tasks = [i for i in range(nt) if nfes_per_task[i] < max_nfes_per_task[i]]
            if not active_tasks:
                break

            for i in active_tasks:
                # Generate offspring: alternate between GA and DE operators
                if i % 2 == 0:
                    off_decs = ga_generation(decs[i], self.muc, self.mum)
                else:
                    off_decs = de_generation(decs[i], self.F, self.CR)

                # Knowledge transfer via mDA at specified intervals
                if self.SNum > 0 and gen % self.TGap == 0:
                    inject_num = int(round(self.SNum / (nt - 1)))
                    inject_decs = np.zeros((0, dims[i]))

                    # Collect best solutions from other tasks
                    for k in active_tasks:
                        if k == i:
                            continue
                        his_rank = np.lexsort((objs[k], cons[k]))
                        his_decs = decs[k][his_rank, :]
                        his_best_decs = his_decs[:inject_num, :dims[k]].squeeze()

                        # Transform via marginalized denoising autoencoder
                        inject_decs_k = mDA(initial_decs[i], initial_decs[k], his_best_decs)
                        inject_decs = np.vstack([inject_decs, inject_decs_k])

                    # Replace random offspring with transferred solutions
                    replace_idx = np.random.choice(off_decs.shape[0], size=inject_decs.shape[0], replace=False)
                    off_decs[replace_idx, :dims[i]] = inject_decs

                offobjs, offcons = evaluation_single(problem, off_decs, i)

                # Merge parent and offspring populations
                objs[i], decs[i], cons[i] = vstack_groups(
                    (objs[i], offobjs), (decs[i], off_decs), (cons[i], offcons)
                )

                # Elitist selection: keep best n individuals
                index = selection_elit(objs[i], n_per_task[i], cons[i])
                objs[i], decs[i], cons[i] = select_by_index(index, objs[i], decs[i], cons[i])

                nfes_per_task[i] += n_per_task[i]
                pbar.update(n_per_task[i])

                append_history(all_decs[i], decs[i], all_objs[i], objs[i], all_cons[i], cons[i])

            gen += 1

        pbar.close()
        runtime = time.time() - start_time

        # Save results
        results = build_save_results(all_decs=all_decs, all_objs=all_objs, runtime=runtime, max_nfes=nfes_per_task,
                                     all_cons=all_cons, bounds=problem.bounds, save_path=self.save_path,
                                     filename=self.name, save_data=self.save_data)

        return results


def mDA(curr_decs, his_decs, his_best_decs):
    """
    Marginalized Denoising Autoencoder for cross-domain knowledge transfer.

    Parameters
    ----------
    curr_decs : np.ndarray
        Current task population of shape (n, d_curr)
    his_decs : np.ndarray
        Historical task population of shape (n, d_his)
    his_best_decs : np.ndarray
        Best solution(s) from historical task of shape (m, d_his)

    Returns
    -------
    inject_decs : np.ndarray
        Transformed solution(s) mapped to current task domain of shape (m, d_curr)

    Notes
    -----
    The mDA learns a linear transformation W that maps solutions from the historical task
    domain to the current task domain using ridge regression:
    W = P @ (Q + λI)^(-1)
    where P = xx @ noise^T, Q = noise @ noise^T, and λ is the regularization parameter.
    """
    curr_len = curr_decs.shape[1]
    his_len = his_decs.shape[1]

    # Align dimensions by zero-padding the shorter one
    if curr_len < his_len:
        curr_decs = np.column_stack([curr_decs, np.zeros((curr_decs.shape[0], his_len - curr_len))])
    elif curr_len > his_len:
        his_decs = np.column_stack([his_decs, np.zeros((his_decs.shape[0], curr_len - his_len))])

    # Transpose for matrix operations
    xx = curr_decs.T
    noise = his_decs.T
    d, n = xx.shape

    # Add bias term
    xxb = np.vstack([xx, np.ones((1, n))])
    noise_xb = np.vstack([noise, np.ones((1, n))])

    # Compute transformation matrix using ridge regression
    Q = noise_xb @ noise_xb.T
    P = xxb @ noise_xb.T
    lambda_reg = 1e-5
    reg = lambda_reg * np.eye(d + 1)
    reg[-1, -1] = 0
    W = P @ np.linalg.inv(Q + reg)

    # Remove bias term from transformation matrix
    W = W[:-1, :-1]

    # Apply transformation to historical best solutions
    if curr_len <= his_len:
        inject_decs = (W @ his_best_decs.T).T[:, :curr_len]
    else:
        his_best_decs = np.column_stack([his_best_decs, np.zeros((his_best_decs.shape[0], curr_len - his_len))])
        inject_decs = (W @ his_best_decs.T).T

    # Clip to valid range [0, 1]
    inject_decs = np.clip(inject_decs, 0., 1.)

    return inject_decs