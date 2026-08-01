"""
Algorithm Template for DDMTOLab

This module is the reference for the algorithm paradigm every algorithm in
``ddmtolab.Algorithms`` follows. Copy it into the matching category folder,
rename the file and the class, and fill in ``optimize()``. The class is then
discovered automatically by the UI and usable with ``BatchExperiment``.

Platform rules
--------------
1. File name and class name are identical; the file lives in ``STSO/``,
   ``STMO/``, ``MTSO/`` or ``MTMO/``. Hyphenated published names are spelled
   with underscores in the file name (``NSGA_II.py`` -> class ``NSGA_II``).
2. The class exposes an ``algorithm_information`` dict with exactly the ten
   canonical keys, in the order shown below, and a
   ``get_algorithm_information`` classmethod delegating to the shared helper.
   The ninth key is ``n_initial`` for algorithms with a design-of-experiments
   phase and ``n`` for purely population-based ones.
3. ``__init__`` starts with ``problem``, then the sizing parameters
   (``n`` or ``n_initial``, then ``max_nfes``), then algorithm-specific
   parameters, and always ends with the four standard parameters
   ``save_data=True, save_path='./Data', name=<display name>,
   disable_tqdm=True``. ``name`` defaults to the file name with underscores
   replaced by hyphens.
4. ``optimize()`` measures runtime, shows a ``tqdm`` progress bar, and returns
   the ``Results`` object built by ``build_save_results``.
5. Populations live in the normalized [0, 1] space; ``initialization``,
   ``evaluation`` / ``evaluation_single`` and ``build_save_results`` handle the
   mapping to each task's real bounds. Never call the problem object directly.

References
----------
    [1] David E. Goldberg. Genetic Algorithms in Search, Optimization, and Machine Learning. Addison-Wesley, Reading, MA, 1989.

Notes
-----
Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2026.08.01
Version: 2.0
"""
import time
from tqdm import tqdm
from ddmtolab.Methods.Algo_Methods.algo_utils import *


class TEMPLATE_ALGORITHM:
    """
    Reference implementation of the DDMTOLab algorithm paradigm.

    The body is a plain (mu + lambda) genetic algorithm, kept minimal so that
    the surrounding structure — not the search logic — is what a new algorithm
    is copied from.

    Attributes
    ----------
    algorithm_information : dict
        Dictionary containing algorithm capabilities and requirements
    """

    algorithm_information = {
        'n_tasks': '[1, K]',        # '[1, K]' single-task, '[2, K]' multi-task
        'dims': 'unequal',          # 'equal' if all tasks must share a dimension
        'objs': 'equal',            # 'equal' if all tasks must share an objective count
        'n_objs': '1',              # '1' single-objective, '[2, M]' multi-objective
        'cons': 'unequal',          # 'equal' if all tasks must share a constraint count
        'n_cons': '[0, C]',         # '0' unconstrained only, '[0, C]' constraints supported
        'expensive': 'False',       # 'True' for surrogate-assisted algorithms
        'knowledge_transfer': 'False',  # 'True' for multi-task transfer algorithms
        'n': 'unequal',             # 'n_initial' instead when a DoE phase is used
        'max_nfes': 'unequal'       # 'equal' if the budget must be shared across tasks
    }

    @classmethod
    def get_algorithm_information(cls, print_info=True):
        return get_algorithm_information(cls, print_info)

    def __init__(self, problem, n=None, max_nfes=None, muc=2.0, mum=5.0, save_data=True, save_path='./Data',
                 name='TEMPLATE-ALGORITHM', disable_tqdm=True):
        """
        Initialize the template algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n : int or List[int], optional
            Population size per task (default: 100)
        max_nfes : int or List[int], optional
            Maximum number of function evaluations per task (default: 10000)
        muc : float, optional
            Distribution index for simulated binary crossover (SBX) (default: 2.0)
        mum : float, optional
            Distribution index for polynomial mutation (PM) (default: 5.0)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './Data')
        name : str, optional
            Name for the experiment (default: 'TEMPLATE-ALGORITHM')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n = n if n is not None else 100
        self.max_nfes = max_nfes if max_nfes is not None else 10000
        self.muc = muc
        self.mum = mum
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the template algorithm.

        Returns
        -------
        Results
            Optimization results containing decision variables, objectives, and runtime
        """
        start_time = time.time()
        problem = self.problem
        nt = problem.n_tasks
        n_per_task = par_list(self.n, nt)
        max_nfes_per_task = par_list(self.max_nfes, nt)

        # Initialize the population in [0, 1] space and evaluate it per task
        decs = initialization(problem, n_per_task)
        objs, cons = evaluation(problem, decs)
        nfes_per_task = n_per_task.copy()
        all_decs, all_objs, all_cons = init_history(decs, objs, cons)

        total_nfes = sum(max_nfes_per_task)
        pbar = tqdm(total=total_nfes, initial=sum(n_per_task), desc=f"{self.name}",
                    disable=self.disable_tqdm)

        while sum(nfes_per_task) < total_nfes:
            # Tasks that have exhausted their own budget stop being evolved
            active_tasks = [i for i in range(nt) if nfes_per_task[i] < max_nfes_per_task[i]]
            if not active_tasks:
                break

            for i in active_tasks:
                # ============================================================
                # Replace this block with the algorithm's own search operators
                # ============================================================
                off_decs = ga_generation(decs[i], self.muc, self.mum)
                off_objs, off_cons = evaluation_single(problem, off_decs, i)
                n_off = off_decs.shape[0]

                # Merge parents and offspring, then keep the best n individuals
                objs[i], decs[i], cons[i] = vstack_groups(
                    (objs[i], off_objs), (decs[i], off_decs), (cons[i], off_cons)
                )
                index = selection_elit(objs=objs[i], n=n_per_task[i], cons=cons[i])
                objs[i], decs[i], cons[i] = select_by_index(index, objs[i], decs[i], cons[i])
                # ============================================================

                nfes_per_task[i] += n_off
                pbar.update(n_off)
                append_history(all_decs[i], decs[i], all_objs[i], objs[i], all_cons[i], cons[i])

        pbar.close()
        runtime = time.time() - start_time

        results = build_save_results(all_decs=all_decs, all_objs=all_objs, runtime=runtime, max_nfes=nfes_per_task,
                                     all_cons=all_cons, bounds=problem.bounds, save_path=self.save_path,
                                     filename=self.name, save_data=self.save_data)

        return results
