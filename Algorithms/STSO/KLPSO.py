"""
Knowledge Learning Particle Swarm Optimization (KLPSO)

This module implements the Knowledge Learning PSO for single-objective optimization problems.

References
----------
    [1] Jiang, Y., Zhan, Z. H., Tan, K. C., & Zhang, J. (2023).
        Knowledge Learning for Evolutionary Computation.
        IEEE Transactions on Evolutionary Computation.
        DOI: 10.1109/TEVC.2023.3278132

Notes
-----
Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2025.12.12
Version: 1.0
"""
import time
from tqdm import tqdm
import numpy as np
from sklearn.neural_network import MLPRegressor
from Methods.Algo_Methods.algo_utils import *


class KLPSO:
    """
    Knowledge Learning Particle Swarm Optimization for single-objective optimization.

    Attributes
    ----------
    algorithm_information : dict
        Dictionary containing algorithm capabilities and requirements
    """

    algorithm_information = {
        'n_tasks': '1-K',
        'dims': 'unequal',
        'objs': 'unequal',
        'n_objs': '1',
        'cons': 'unequal',
        'n_cons': '0',
        'expensive': 'False',
        'knowledge_transfer': 'False',
        'n': 'unequal',
        'max_nfes': 'unequal'
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

    def __init__(self, problem, n=None, max_nfes=None, lr=0.2, epochs=10,
                 min_w=0.4, max_w=0.9, c1=0.2, c2=0.2,
                 save_data=True, save_path='./TestData', name='KLPSO_test', disable_tqdm=True):
        """
        Initialize Knowledge Learning PSO algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n : int or List[int], optional
            Population size per task (default: 100)
        max_nfes : int or List[int], optional
            Maximum number of function evaluations per task (default: 10000)
        lr : float, optional
            Learning rate (probability of using neural network) (default: 0.2)
        epochs : int, optional
            Training epochs for neural network (default: 10)
        min_w : float, optional
            Minimum inertia weight (default: 0.4)
        max_w : float, optional
            Maximum inertia weight (default: 0.9)
        c1 : float, optional
            Cognitive coefficient (default: 0.2)
        c2 : float, optional
            Social coefficient (default: 0.2)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './TestData')
        name : str, optional
            Name for the experiment (default: 'KLPSO_test')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n = n if n is not None else 100
        self.max_nfes = max_nfes if max_nfes is not None else 10000
        self.lr = lr
        self.epochs = epochs
        self.min_w = min_w
        self.max_w = max_w
        self.c1 = c1
        self.c2 = c2
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the Knowledge Learning PSO algorithm.

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

        # Initialize population in [0,1] space and evaluate for each task
        decs = initialization(problem, n_per_task)
        objs, cons = evaluation(problem, decs)
        nfes_per_task = n_per_task.copy()
        all_decs, all_objs, all_cons = init_history(decs, objs, cons)

        # Initialize particle velocities to zero
        vel = [np.zeros_like(d) for d in decs]

        # Initialize personal best positions and objectives
        pbest_decs = [d.copy() for d in decs]
        pbest_objs = [o.copy() for o in objs]
        pbest_cons = [c.copy() for c in cons]

        # Initialize global best for each task
        gbest_decs = []
        gbest_objs = []

        for i in range(nt):
            cvs = np.sum(np.maximum(0, cons[i]), axis=1)
            sort_indices = np.lexsort((objs[i].flatten(), cvs))
            gbest_decs.append(decs[i][sort_indices[0]:sort_indices[0] + 1, :])
            gbest_objs.append(objs[i][sort_indices[0]:sort_indices[0] + 1, :])

        # Initialize knowledge learning components for each task
        in_list = [[] for _ in range(nt)]
        out_list = [[] for _ in range(nt)]
        net = [None] * nt
        trained = [False] * nt

        total_nfes = sum(max_nfes_per_task)
        pbar = tqdm(total=total_nfes, initial=sum(n_per_task), desc=f"{self.name}",
                    disable=self.disable_tqdm)

        while sum(nfes_per_task) < total_nfes:
            # Linearly decrease inertia weight from max_w to min_w
            w = self.max_w - (self.max_w - self.min_w) * sum(nfes_per_task) / sum(max_nfes_per_task)

            # Skip tasks that have exhausted their evaluation budget
            active_tasks = [i for i in range(nt) if nfes_per_task[i] < max_nfes_per_task[i]]
            if not active_tasks:
                break

            for i in active_tasks:
                dim = problem.dims[i]
                old_decs = decs[i].copy()

                # Generate new positions
                new_decs = np.zeros_like(decs[i])

                for j in range(n_per_task[i]):
                    # Use neural network with probability lr if trained
                    if np.random.rand() < self.lr and trained[i]:
                        # Predict using neural network
                        test_input = decs[i][j:j + 1, :]
                        try:
                            prediction = net[i].predict(test_input)
                            new_decs[j] = decs[i][j] + 2 * np.random.rand() * prediction.flatten()
                        except:
                            # If prediction fails, use standard PSO
                            vel[i][j] = (w * vel[i][j] +
                                         self.c1 * np.random.rand() * (pbest_decs[i][j] - decs[i][j]) +
                                         self.c2 * np.random.rand() * (gbest_decs[i].flatten() - decs[i][j]))
                            new_decs[j] = decs[i][j] + vel[i][j]
                    else:
                        # Standard PSO velocity and position update
                        vel[i][j] = (w * vel[i][j] +
                                     self.c1 * np.random.rand() * (pbest_decs[i][j] - decs[i][j]) +
                                     self.c2 * np.random.rand() * (gbest_decs[i].flatten() - decs[i][j]))
                        new_decs[j] = decs[i][j] + vel[i][j]

                # Boundary constraint handling: clip to [0,1] space
                new_decs = np.clip(new_decs, 0, 1)
                decs[i] = new_decs

                # Evaluate new positions
                objs[i], cons[i] = evaluation_single(problem, decs[i], i)

                # Update personal best
                new_cvs = np.sum(np.maximum(0, cons[i]), axis=1)
                pbest_cvs = np.sum(np.maximum(0, pbest_cons[i]), axis=1)

                for j in range(n_per_task[i]):
                    if (new_cvs[j] < pbest_cvs[j]) or \
                            (new_cvs[j] == pbest_cvs[j] and objs[i][j] < pbest_objs[i][j]):
                        pbest_decs[i][j] = decs[i][j]
                        pbest_objs[i][j] = objs[i][j]
                        pbest_cons[i][j] = cons[i][j]

                # Update global best
                cvs = np.sum(np.maximum(0, pbest_cons[i]), axis=1)
                sort_indices = np.lexsort((pbest_objs[i].flatten(), cvs))
                best_idx = sort_indices[0]

                if (cvs[best_idx] < np.sum(np.maximum(0, cons[i][0]))) or \
                        (cvs[best_idx] == 0 and pbest_objs[i][best_idx] < gbest_objs[i][0]):
                    gbest_decs[i] = pbest_decs[i][best_idx:best_idx + 1, :]
                    gbest_objs[i] = pbest_objs[i][best_idx:best_idx + 1, :]

                # Collect training data (improved solutions)
                for j in range(n_per_task[i]):
                    if (new_cvs[j] < pbest_cvs[j]) or \
                            (new_cvs[j] == pbest_cvs[j] and objs[i][j] < pbest_objs[i][j]):
                        in_list[i].append(old_decs[j])
                        out_list[i].append(decs[i][j] - old_decs[j])

                # Train neural network
                if not trained[i] and len(in_list[i]) > 0:
                    # Initialize neural network: [16, 16, dim] hidden layers
                    net[i] = MLPRegressor(
                        hidden_layer_sizes=(16, 16, dim),
                        activation='logistic',  # sigmoid
                        solver='sgd',
                        learning_rate_init=0.1,
                        max_iter=self.epochs,
                        random_state=42,
                        verbose=False
                    )
                    trained[i] = True

                # Update neural network with collected data
                if trained[i] and len(in_list[i]) > 0:
                    X_train = np.array(in_list[i])
                    y_train = np.array(out_list[i])
                    try:
                        net[i].fit(X_train, y_train)
                    except:
                        pass  # Continue if training fails

                # Clear training data for next iteration
                in_list[i] = []
                out_list[i] = []

                nfes_per_task[i] += n_per_task[i]
                pbar.update(n_per_task[i])

                # Append current population to history
                append_history(all_decs[i], decs[i], all_objs[i], objs[i], all_cons[i], cons[i])

        pbar.close()
        runtime = time.time() - start_time

        # Save results
        results = build_save_results(all_decs, all_objs, runtime, nfes_per_task,
                                     all_cons=all_cons, save_path=self.save_path,
                                     filename=self.name, save_data=self.save_data)

        return results