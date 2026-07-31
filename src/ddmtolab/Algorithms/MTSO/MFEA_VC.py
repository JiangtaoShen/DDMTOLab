"""
Multifactorial Evolutionary Algorithm with Variational Crossover (MFEA-VC)

This module implements MFEA-VC for multi-task optimization using a contrastive
Variational Auto-Encoder (VAE) to guide knowledge transfer in early generations.

References
----------
    [1] Wang, Ruilin, et al. "Contrastive Variational Auto-Encoder Driven
        Convergence Guidance in Evolutionary Multitasking." Applied Soft
        Computing, 163: 111883, 2024.

Notes
-----
Author: Jiangtao Shen (DDMTOLab adaptation)
Date: 2026.02.22
Version: 1.0
"""
import time
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from ddmtolab.Methods.Algo_Methods.algo_utils import *


class MFEA_VC:
    """
    Multifactorial Evolutionary Algorithm with Variational Crossover.

    Uses a VAE (with random weights, no training) to generate cross-task
    individuals for the first `vae_gens` generations. The VAE encodes both
    tasks' population data into a shared latent space and decodes to produce
    mixed-task offspring used as SBX crossover partners.

    After `vae_gens` generations, reverts to standard MFEA behavior with
    SBX crossover and polynomial mutation.

    Attributes
    ----------
    algorithm_information : dict
        Dictionary containing algorithm capabilities and requirements
    """

    algorithm_information = {
        'n_tasks': '[2, K]',
        'dims': 'unequal',
        'objs': 'equal',
        'n_objs': '1',
        'cons': 'unequal',
        'n_cons': '[0, C]',
        'expensive': 'False',
        'knowledge_transfer': 'True',
        'n': 'equal',
        'max_nfes': 'equal'
    }

    @classmethod
    def get_algorithm_information(cls, print_info=True):
        return get_algorithm_information(cls, print_info)

    def __init__(self, problem, n=None, max_nfes=None, rmp=0.3, muc=2, mum=5,
                 vae_gens=25, lam=0.8, save_data=True, save_path='./Data',
                 name='MFEA-VC', disable_tqdm=True):
        """
        Initialize MFEA-VC algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n : int, optional
            Population size per task (default: 100)
        max_nfes : int, optional
            Maximum number of function evaluations per task (default: 10000)
        rmp : float, optional
            Random mating probability (default: 0.3)
        muc : float, optional
            Distribution index for SBX crossover (default: 2)
        mum : float, optional
            Distribution index for polynomial mutation (default: 5)
        vae_gens : int, optional
            Number of generations to use VAE-guided crossover (default: 25)
        lam : float, optional
            Lambda scaling factor for VAE latent space (default: 0.8)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './Data')
        name : str, optional
            Name for the experiment (default: 'MFEA-VC')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n = n if n is not None else 100
        self.max_nfes = max_nfes if max_nfes is not None else 10000
        self.rmp = rmp
        self.muc = muc
        self.mum = mum
        self.vae_gens = vae_gens
        self.lam = lam
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the MFEA-VC algorithm.

        Returns
        -------
        Results
            Optimization results containing decision variables, objectives, and runtime
        """
        start_time = time.time()
        problem = self.problem
        nt = problem.n_tasks
        dims = problem.dims
        n = self.n
        max_nfes_per_task = par_list(self.max_nfes, nt)
        max_nfes = self.max_nfes * nt
        pop_size = n * nt

        # Initialize population and evaluate
        decs = initialization(problem, n)
        objs, cons = evaluation(problem, decs)
        nfes = n * nt
        all_decs, all_objs, all_cons = init_history(decs, objs, cons)

        # Transform to unified space
        pop_decs, pop_cons = space_transfer(problem=problem, decs=decs, cons=cons,
                                            type='uni', padding='mid')
        pop_objs = objs
        pop_sfs = [np.full((n, 1), fill_value=i) for i in range(nt)]

        pbar = tqdm(total=max_nfes, initial=nfes, desc=f"{self.name}",
                    disable=self.disable_tqdm)

        # MToP increments Algo.Gen inside notTerminated before the first loop body,
        # so the first generation executed by the reference runs with Gen == 2 and
        # the VAE branch is therefore active for generations 2..25.
        gen = 2
        while nfes < max_nfes:
            # --- VAE generation (first vae_gens generations, 2-task only) ---
            vae_decs = None
            if gen <= self.vae_gens and nt == 2:
                vae_decs = _generate_vae_individuals(
                    pop_decs, pop_objs, nt, self.lam)

            # Merge populations
            m_decs, m_objs, m_cons, m_sfs = vstack_groups(
                pop_decs, pop_objs, pop_cons, pop_sfs)

            # --- Generation ---
            off_decs = np.zeros_like(m_decs)
            off_objs = np.full_like(m_objs, np.inf)
            off_cons = np.zeros_like(m_cons)
            off_sfs = np.zeros_like(m_sfs)

            shuffled = np.random.permutation(pop_size)

            for pair_idx in range(pop_size // 2):
                p1 = shuffled[pair_idx]
                p2 = shuffled[pair_idx + pop_size // 2]
                sf1 = m_sfs[p1].item()
                sf2 = m_sfs[p2].item()
                idx1 = pair_idx * 2
                idx2 = pair_idx * 2 + 1

                if sf1 == sf2 or np.random.rand() < self.rmp:
                    # --- Transfer: crossover ---
                    if vae_decs is not None and len(vae_decs) > 0:
                        # One shared VAE partner per pair; the reference calls
                        # GA_Crossover twice and keeps only the first child of each
                        # call, so the second call overwrites the second offspring.
                        vae_dec = vae_decs[np.random.randint(len(vae_decs))]

                        off_decs[idx1], _ = crossover(
                            m_decs[p1], vae_dec, mu=self.muc)
                        off_decs[idx2], _ = crossover(
                            m_decs[p2], vae_dec, mu=self.muc)
                    else:
                        # Standard SBX crossover
                        off_decs[idx1], off_decs[idx2] = crossover(
                            m_decs[p1], m_decs[p2], mu=self.muc)

                    # Task imitation: random parent's MFFactor
                    off_sfs[idx1] = np.random.choice([sf1, sf2])
                    off_sfs[idx2] = np.random.choice([sf1, sf2])
                else:
                    # --- No transfer: polynomial mutation ---
                    off_decs[idx1] = mutation(m_decs[p1], mu=self.mum)
                    off_decs[idx2] = mutation(m_decs[p2], mu=self.mum)
                    off_sfs[idx1] = sf1
                    off_sfs[idx2] = sf2

                # Clip to [0, 1]
                off_decs[idx1] = np.clip(off_decs[idx1], 0, 1)
                off_decs[idx2] = np.clip(off_decs[idx2], 0, 1)

            # --- Evaluation ---
            for idx in range(pop_size):
                t = off_sfs[idx].item()
                off_objs[idx], off_cons[idx] = evaluation_single(
                    problem, off_decs[idx, :dims[t]], t)

            nfes += pop_size
            pbar.update(pop_size)

            # --- Selection ---
            merged_decs = np.vstack([m_decs, off_decs])
            merged_objs = np.vstack([m_objs, off_objs])
            merged_cons = np.vstack([m_cons, off_cons])
            merged_sfs = np.vstack([m_sfs, off_sfs])

            pop_decs, pop_objs, pop_cons, pop_sfs = [], [], [], []
            for t in range(nt):
                indices = np.where(merged_sfs.flatten() == t)[0]
                t_decs, t_objs, t_cons = select_by_index(
                    indices, merged_decs, merged_objs, merged_cons)
                sel = selection_elit(objs=t_objs, n=n, cons=t_cons)
                pop_decs.append(t_decs[sel])
                pop_objs.append(t_objs[sel])
                pop_cons.append(t_cons[sel])
                pop_sfs.append(np.full((n, 1), t))

            # Record history
            real_decs, real_cons = space_transfer(
                problem, decs=pop_decs, cons=pop_cons, type='real')
            append_history(all_decs, real_decs, all_objs, pop_objs, all_cons, real_cons)

            gen += 1

        pbar.close()
        runtime = time.time() - start_time

        results = build_save_results(
            all_decs=all_decs, all_objs=all_objs, runtime=runtime,
            max_nfes=max_nfes_per_task, all_cons=all_cons,
            bounds=problem.bounds, save_path=self.save_path,
            filename=self.name, save_data=self.save_data)

        return results


# ============================================================
# VAE model and generation helpers
# ============================================================

class _SimpleVAE(nn.Module):
    """
    VAE matching the MyVAE network graph of the MToP reference.

    Encoder: input -> FC(H) -> ReLU -> FC(H) -> ReLU -> FC(H) -> ReLU ->
             FC(H) -> Sigmoid -> FC_mean(L) -> FC_logvar(L)
    Reparametrization: z = mu + exp(0.5 * logvar) * eps
    Decoder: FC(H) -> ReLU -> FC(H) -> ReLU -> FC(H) -> ReLU ->
             FC(H) -> Sigmoid -> FC(input_size)

    ``fc_logvar`` consumes ``fc_mean``'s output because MyVAE stacks the two
    fully connected layers sequentially. A full forward pass therefore already
    contains one reparametrization draw, exactly like ``predict(net, x)``.
    """

    def __init__(self, input_size, hidden_size=256, latent_size=200):
        super().__init__()
        self.encoder_body = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid(),
        )
        # Sequential: sigmoid -> fc_mean -> fc_logvar (logvar takes mean as input)
        self.fc_mean = nn.Linear(hidden_size, latent_size)
        self.fc_logvar = nn.Linear(latent_size, latent_size)

        self.decoder = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, input_size),
        )

        # The network is never trained (MyVAE runs with istraining = false), so
        # its weights ARE the transfer operator. Draw them from the NumPy global
        # stream instead of torch's, so that np.random.seed reproduces a whole
        # run -- torch keeps an independent RNG that np.random.seed cannot reach.
        self._init_weights_from_numpy()

    def _init_weights_from_numpy(self):
        """Re-draw every Linear layer from NumPy using PyTorch's own default law.

        ``nn.Linear`` initialises both weight and bias uniformly on
        ``[-1/sqrt(fan_in), 1/sqrt(fan_in)]``; reproducing that law here keeps
        the network statistically identical while making it seedable.
        """
        with torch.no_grad():
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    bound = 1.0 / np.sqrt(module.in_features)
                    w = np.random.uniform(-bound, bound, size=tuple(module.weight.shape))
                    module.weight.copy_(torch.as_tensor(w, dtype=module.weight.dtype))
                    if module.bias is not None:
                        b = np.random.uniform(-bound, bound, size=tuple(module.bias.shape))
                        module.bias.copy_(torch.as_tensor(b, dtype=module.bias.dtype))

    @staticmethod
    def _reparametrize(mu, logvar):
        """z = mu + exp(0.5 * logvar) * eps, guarded against exp() overflow."""
        # Drawn from NumPy for the same seeding reason as the weights above.
        eps = torch.as_tensor(np.random.randn(*tuple(mu.shape)), dtype=mu.dtype)
        return mu + torch.exp(0.5 * torch.clamp(logvar, max=30.0)) * eps

    def forward(self, x):
        """Full network pass: encoder -> reparametrization -> decoder."""
        h = self.encoder_body(x)
        z_mean = self.fc_mean(h)
        z_logvar = self.fc_logvar(z_mean)
        return self.decoder(self._reparametrize(z_mean, z_logvar))

    @torch.no_grad()
    def generate(self, x, lam=0.8):
        """
        Reproduce MyVAE.generate: encode(X1, X2) then decode(z * lambda).

        ``encode`` runs the whole network, splits the *batch* in half and treats
        the first half as the mean and the second half as the log-variance;
        ``decode`` then runs the whole network again on the scaled result.

        Parameters
        ----------
        x : torch.Tensor
            Combined data from both tasks, shape (n1 + n2, input_size).
            First n1 rows = task 1, last n2 rows = task 2.
        lam : float
            Lambda scaling factor applied before the second forward pass.

        Returns
        -------
        output : np.ndarray
            Decoded output, shape (n1, input_size) for equal halves.
        """
        out = self.forward(x)

        n_half = x.shape[0] // 2
        z_mean = out[:n_half]
        z_logvar = out[n_half:2 * n_half]
        z = self._reparametrize(z_mean, z_logvar)

        return self.forward(z * lam).numpy()


def _generate_vae_individuals(pop_decs, pop_objs, nt, lam):
    """
    Generate VAE-guided individuals for knowledge transfer.

    Prepares population data, passes through the untrained VAE, and extracts
    decision variables for use as crossover partners.

    Parameters
    ----------
    pop_decs : list of np.ndarray
        Population decision variables per task (unified space)
    pop_objs : list of np.ndarray
        Population objective values per task
    nt : int
        Number of tasks (the reference only handles nt == 2)
    lam : float
        Lambda scaling applied between the two forward passes

    Returns
    -------
    vae_decs : np.ndarray
        VAE-generated decision vectors for crossover, shape (n_new, maxD)

    Notes
    -----
    The reference builds each column as ``[Dec * 10000; MFObj; TaskLabel * 10000]``
    where ``MFObj`` holds one entry per task. From the second generation onwards
    the reference leaves the non-skill entries at ``inf``, which propagates NaN
    through the network and yields unusable offspring; here the individual's own
    objective is used as a finite stand-in for those entries.
    """
    if nt != 2:
        return np.zeros((0, pop_decs[0].shape[1]))

    maxD = pop_decs[0].shape[1]
    desired_rows = 100
    n_train = desired_rows // 2

    # Build data matrices: [Dec * 10000, MFObj (one column per task), Label * 10000]
    data_tasks = []
    for t in range(nt):
        n_t = pop_decs[t].shape[0]
        dec_scaled = pop_decs[t] * 10000.0
        mf_objs = np.repeat(pop_objs[t][:, :1], nt, axis=1)
        task_label = np.full((n_t, 1), (t + 1) * 10000.0)
        data_tasks.append(np.hstack([dec_scaled, mf_objs, task_label]))

    for t in range(nt):
        n_t = data_tasks[t].shape[0]
        if n_t > desired_rows:
            data_tasks[t] = data_tasks[t][:desired_rows]
        elif n_t < desired_rows:
            # datasample() draws with replacement
            extra_idx = np.random.randint(0, n_t, size=desired_rows - n_t)
            data_tasks[t] = np.vstack([data_tasks[t], data_tasks[t][extra_idx]])

        # randperm(numX1) only permutes the first floor(0.5 * 100) = 50 columns
        perm = np.random.permutation(n_train)
        data_tasks[t] = data_tasks[t][perm]

    # encode() drops the trailing TaskLabel row before running the network
    x_combined = np.vstack([data_tasks[0][:, :-1],
                            data_tasks[1][:, :-1]]).astype(np.float32)
    input_size = x_combined.shape[1]

    # Random weights, never trained (the reference keeps istraining = false)
    vae = _SimpleVAE(input_size, hidden_size=256, latent_size=200)
    vae.eval()

    output = vae.generate(torch.from_numpy(x_combined), lam=lam)

    # The reference splits the output into two task halves and re-attaches the
    # label row; only the leading maxD entries are ever read back as Dec.
    vae_decs = np.asarray(output[:, :maxD], dtype=np.float64)

    # An untrained network fed 1e4-scaled inputs can saturate to non-finite
    # values; the reference would propagate them into unusable offspring.
    bad = ~np.isfinite(vae_decs)
    if np.any(bad):
        vae_decs = vae_decs.copy()
        vae_decs[bad] = np.random.rand(int(np.count_nonzero(bad)))

    return vae_decs
