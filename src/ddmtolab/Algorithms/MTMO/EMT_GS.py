"""
Evolutionary Multitasking for Multi-objective Optimization Based on Generative Strategies (EMT-GS)

This module implements EMT-GS for multi-task multi-objective optimization problems.
EMT-GS uses Generative Adversarial Networks (GANs) to transfer knowledge between tasks.

References
----------
    [1] Liang, Zhengping, Yingmiao Zhu, Xiyu Wang, Zhi Li, and Zexuan Zhu. "Evolutionary Multitasking for Multi-objective Optimization Based on Generative Strategies." IEEE Transactions on Evolutionary Computation (2022): 1-1.

Notes
-----
Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2026.02.22
Version: 1.1
"""
import time
import torch
import torch.nn as nn
import torch.nn.functional as F_nn
from tqdm import tqdm
from ddmtolab.Methods.Algo_Methods.algo_utils import *

# MATLAB's ``eps`` used inside the GAN log-losses
_MATLAB_EPS = 2.220446049250313e-16


def _gaussian_linear(in_features, out_features, sigma):
    """
    Fully connected layer initialized as MTO-Platform ``initializeGaussian``.

    The weights are drawn from the NumPy global stream so that the whole
    algorithm stays reproducible through ``np.random.seed``, mirroring MATLAB
    where a single random stream drives both the evolution and the network.

    Parameters
    ----------
    in_features : int
        Number of input features
    out_features : int
        Number of output features
    sigma : float
        Standard deviation of the Gaussian weight initialization

    Returns
    -------
    layer : nn.Linear
        Initialized linear layer with zero bias
    """
    layer = nn.Linear(in_features, out_features)
    with torch.no_grad():
        layer.weight.copy_(torch.from_numpy(
            (np.random.randn(out_features, in_features) * sigma).astype(np.float32)))
        layer.bias.zero_()
    return layer


def _matlab_dropout(x, p=0.5):
    """
    Dropout of MTO-Platform ``InitialGAN/dropout``: a plain 0/1 mask, no rescaling.

    Parameters
    ----------
    x : torch.Tensor
        Input activations
    p : float, optional
        Drop probability (default: 0.5)

    Returns
    -------
    y : torch.Tensor
        Masked activations
    """
    mask = (np.random.randint(1, 11, size=tuple(x.shape)) > p * 10).astype(np.float32)
    return x * torch.from_numpy(mask)


class _Generator(nn.Module):
    """
    Generator of MTO-Platform ``InitialGAN``: FC -> LeakyReLU(0.5) -> BN, twice, then FC -> sigmoid.

    ``width`` is the size of the vectors the network maps. Following the MATLAB
    reference the network is applied along the *population* axis: a sample is one
    decision-variable column of the population matrix, hence ``width`` equals the
    population size and the mini-batches iterate over decision variables.
    """

    def __init__(self, width):
        super().__init__()
        self.fc1 = _gaussian_linear(width, width, 0.03)
        self.bn1 = nn.BatchNorm1d(width)
        self.fc2 = _gaussian_linear(width, width, 0.06)
        self.bn2 = nn.BatchNorm1d(width)
        self.fc_out = _gaussian_linear(width, width, 0.06)

    def forward(self, x):
        h = self.bn1(F_nn.leaky_relu(self.fc1(x), 0.5))
        h = self.bn2(F_nn.leaky_relu(self.fc2(h), 0.5))
        return torch.sigmoid(self.fc_out(h))


class _Discriminator(nn.Module):
    """
    Discriminator of MTO-Platform ``InitialGAN``: FC -> LeakyReLU(0.5) -> dropout -> BN -> FC -> sigmoid.
    """

    def __init__(self, width):
        super().__init__()
        self.fc1 = _gaussian_linear(width, width, 0.03)
        self.bn1 = nn.BatchNorm1d(width)
        self.fc_out = _gaussian_linear(width, 1, 0.06)

    def forward(self, x):
        h = _matlab_dropout(F_nn.leaky_relu(self.fc1(x), 0.5))
        h = self.bn1(h)
        return torch.sigmoid(self.fc_out(h))


class EMT_GS:
    """
    Evolutionary Multitasking for Multi-objective Optimization Based on Generative Strategies.

    This algorithm features:
    - GAN-based cross-task knowledge transfer, one GAN per ordered task pair
    - DE mutation with a rand-or-best base vector and the previous population
    - NSGA-II based environmental selection per task

    Attributes
    ----------
    algorithm_information : dict
        Dictionary containing algorithm capabilities and requirements
    """

    algorithm_information = {
        'n_tasks': '[2, K]',
        'dims': 'unequal',
        'objs': 'unequal',
        'n_objs': '[2, M]',
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

    def __init__(self, problem, n=None, max_nfes=None, G=10, lrD=0.0002, lrG=0.0003, BS=10,
                 save_data=True, save_path='./Data', name='EMT-GS', disable_tqdm=True):
        """
        Initialize EMT-GS algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n : int, optional
            Population size per task (default: 100)
        max_nfes : int, optional
            Maximum number of function evaluations per task (default: 10000)
        G : int, optional
            GAN training gap in generations (default: 10)
        lrD : float, optional
            Learning rate for the discriminator (default: 0.0002)
        lrG : float, optional
            Learning rate for the generator (default: 0.0003)
        BS : int, optional
            Batch size for GAN training (default: 10)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './Data')
        name : str, optional
            Name for the experiment (default: 'EMT-GS')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n = n if n is not None else 100
        self.max_nfes = max_nfes if max_nfes is not None else 10000
        self.G = G
        self.lrD = lrD
        self.lrG = lrG
        self.BS = BS
        # DE constants hard coded in the MATLAB reference
        self.pp = 0.5
        self.CR = 0.6
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the EMT-GS algorithm.

        Returns
        -------
        Results
            Optimization results containing decision variables, objectives, constraints, and runtime
        """
        start_time = time.time()
        problem = self.problem
        n = self.n
        nt = problem.n_tasks
        dims = problem.dims
        max_nfes_per_task = par_list(self.max_nfes, nt)
        max_nfes = self.max_nfes * nt

        # Population lives in the unified [0, 1] space of dimension max(dims); the
        # genes beyond a task's own dimension are initialized at random and keep
        # evolving, exactly as in the MATLAB reference (Dec = rand(1, max(D))).
        decs = space_transfer(problem, initialization(problem, n), type='uni', padding='random')

        objs, cons = [], []
        for t in range(nt):
            obj_t, con_t = evaluation_single(problem, decs[t][:, :dims[t]], t)
            objs.append(obj_t)
            cons.append(con_t)
        nfes = n * nt

        # Sort each task's population by non-dominated rank then crowding distance
        for t in range(nt):
            rank_t, _, _ = nsga2_sort(objs[t], cons[t])
            order = np.argsort(rank_t)
            decs[t], objs[t], cons[t] = decs[t][order], objs[t][order], cons[t][order]

        all_decs, all_objs, all_cons = init_history(
            [decs[t][:, :dims[t]] for t in range(nt)], objs, cons)

        # Previous generation population, used as the DE difference base
        prepop = [d.copy() for d in decs]

        # One GAN per ordered task pair: GAN[(t, k)] maps task k solutions to task t
        generators, discriminators, gan_off = {}, {}, {}
        for t in range(nt):
            for k in range(nt):
                if t == k:
                    continue
                generators[(t, k)] = _Generator(n)
                discriminators[(t, k)] = _Discriminator(n)
                gan_off[(t, k)] = self._train_gan(
                    generators[(t, k)], discriminators[(t, k)], decs[t], decs[k], epochs=20)

        pbar = tqdm(total=max_nfes, initial=nfes, desc=f"{self.name}", disable=self.disable_tqdm)

        # MTO-Platform increments Algo.Gen inside notTerminated before the loop
        # body runs, so the first executed generation already sees Gen == 2
        gen = 1
        while nfes < max_nfes:
            gen += 1

            for t in range(nt):
                for k in range(nt):
                    if t == k:
                        continue
                    if gen % self.G == 0:
                        # Train GAN
                        gan_off[(t, k)] = self._train_gan(
                            generators[(t, k)], discriminators[(t, k)], decs[t], decs[k], epochs=2)
                    else:
                        # Both MATLAB branches run the same generator parameters and
                        # normalize with the current batch statistics, so they only
                        # differ in the discarded running-statistics bookkeeping
                        np.random.rand()
                        gan_off[(t, k)] = self._generate_gan(generators[(t, k)], decs[k])

            # Generation
            off_decs, off_sfs = self._generation(decs, prepop, gan_off, n, nt)
            prepop = [d.copy() for d in decs]

            for t in range(nt):
                # Evaluation
                mask = off_sfs == t
                if not np.any(mask):
                    continue
                off_decs_t = off_decs[mask, :]
                off_objs_t, off_cons_t = evaluation_single(problem, off_decs_t[:, :dims[t]], t)
                nfes += off_decs_t.shape[0]
                pbar.update(off_decs_t.shape[0])

                # Selection: NSGA-II sorting on the merged parent + offspring pool
                merged_decs, merged_objs, merged_cons = vstack_groups(
                    (decs[t], off_decs_t), (objs[t], off_objs_t), (cons[t], off_cons_t))
                rank_t, _, _ = nsga2_sort(merged_objs, merged_cons)
                index = np.argsort(rank_t)[:n]
                decs[t], objs[t], cons[t] = select_by_index(index, merged_decs, merged_objs, merged_cons)

            append_history(all_decs, [decs[t][:, :dims[t]] for t in range(nt)],
                           all_objs, objs, all_cons, cons)

        pbar.close()
        runtime = time.time() - start_time

        results = build_save_results(all_decs=all_decs, all_objs=all_objs, runtime=runtime,
                                     max_nfes=max_nfes_per_task, all_cons=all_cons,
                                     bounds=problem.bounds, save_path=self.save_path,
                                     filename=self.name, save_data=self.save_data)

        return results

    def _train_gan(self, gen, disc, target_decs, source_decs, epochs):
        """
        Train one GAN (MTO-Platform ``InitialGAN`` / ``TrainGAN``).

        Following the MATLAB reference the population matrices are consumed
        column-wise: one training sample is one decision variable across the whole
        population, so a mini-batch holds ``BS`` decision variables. The Adam
        moment estimates are recreated at every call, as the reference resets
        ``avgG``/``avgGS`` on entry.

        Parameters
        ----------
        gen : _Generator
            Generator network
        disc : _Discriminator
            Discriminator network
        target_decs : np.ndarray
            Population of the target task, shape (n, d_uni)
        source_decs : np.ndarray
            Population of the source task used as latent input, shape (n, d_uni)
        epochs : int
            Number of training epochs (20 on the first call, 2 afterwards)

        Returns
        -------
        generated : np.ndarray
            Solutions generated from the source population, shape (n, d_uni)
        """
        gen.train()
        disc.train()
        opt_g = torch.optim.Adam(gen.parameters(), lr=self.lrG, betas=(0.7, 0.9))
        opt_d = torch.optim.Adam(disc.parameters(), lr=self.lrD, betas=(0.7, 0.9))

        d_uni = target_decs.shape[1]
        noise_data = source_decs[:, np.random.permutation(d_uni)]
        n_iter = d_uni // self.BS

        gen_params, disc_params = list(gen.parameters()), list(disc.parameters())
        for _ in range(epochs):
            train_shuffled = target_decs[:, np.random.permutation(d_uni)]
            for i in range(n_iter):
                sl = slice(i * self.BS, (i + 1) * self.BS)
                if sl.stop - sl.start < 2:
                    continue
                x_batch = torch.from_numpy(train_shuffled[:, sl].T.astype(np.float32))
                z_batch = torch.from_numpy(noise_data[:, sl].T.astype(np.float32))

                fake = gen(z_batch)
                d_real = disc(x_batch)
                d_fake = disc(fake)

                d_loss = -torch.mean(0.9 * torch.log(d_real + _MATLAB_EPS) +
                                     torch.log(1 - d_fake + _MATLAB_EPS))
                g_loss = -torch.mean(torch.log(d_fake + _MATLAB_EPS))

                # Both gradients come from the same forward pass, as in dlfeval
                grad_g = torch.autograd.grad(g_loss, gen_params, retain_graph=True)
                grad_d = torch.autograd.grad(d_loss, disc_params)

                opt_d.zero_grad()
                for prm, grd in zip(disc_params, grad_d):
                    prm.grad = grd
                opt_d.step()

                opt_g.zero_grad()
                for prm, grd in zip(gen_params, grad_g):
                    prm.grad = grd
                opt_g.step()

        with torch.no_grad():
            generated = gen(torch.from_numpy(noise_data.T.astype(np.float32))).numpy().T
        return generated.astype(np.float64)

    def _generate_gan(self, gen, source_decs):
        """
        Run a trained generator on a population (MTO-Platform ``GenerateGAN``).

        The reference always normalizes with the statistics of the batch it is
        given, so the network stays in training mode here (no running estimates).

        Parameters
        ----------
        gen : _Generator
            Trained generator network
        source_decs : np.ndarray
            Population used as latent input, shape (n, d_uni)

        Returns
        -------
        generated : np.ndarray
            Generated solutions, shape (n, d_uni)
        """
        gen.train()
        with torch.no_grad():
            generated = gen(torch.from_numpy(source_decs.T.astype(np.float32))).numpy().T
        return generated.astype(np.float64)

    def _generation(self, population, prepop, gan_off, n, nt):
        """
        Offspring generation (MTO-Platform ``EMT_GS.Generation``).

        The concatenated population is randomly permuted and paired as
        (i, i + floor(L/2)). Same-task pairs use DE/rand-or-best/1 with the
        difference taken against the previous generation; cross-task pairs
        replace the DE donor by the GAN image of the partner task.

        Parameters
        ----------
        population : list of np.ndarray
            Current population of each task in unified space, shape (n, d_uni)
        prepop : list of np.ndarray
            Population of each task at the previous generation
        gan_off : dict
            ``gan_off[(t, k)]`` holds the task-k population mapped into task t
        n : int
            Population size per task
        nt : int
            Number of tasks

        Returns
        -------
        off_decs : np.ndarray
            Offspring decision variables, shape (2 * ceil(L / 2), d_uni)
        off_sfs : np.ndarray
            Offspring skill factors, shape (2 * ceil(L / 2),)
        """
        parent = np.vstack(population)
        sfs = np.repeat(np.arange(nt), n)
        length, d = parent.shape

        rndper = np.random.permutation(length)
        parent = parent[rndper, :]
        rank = np.arange(length)[rndper]
        mff_pool = sfs[rndper]

        half = length // 2
        n_pairs = int(np.ceil(length / 2))
        off_decs = np.empty((2 * n_pairs, d))
        off_sfs = np.empty(2 * n_pairs, dtype=int)

        count = 0
        for i in range(n_pairs):
            p1, p2 = i, i + half
            sf1, sf2 = int(mff_pool[p1]), int(mff_pool[p2])

            f_scale = np.random.normal(0.5, 0.2)
            while f_scale > 1 or f_scale < 0:
                f_scale = np.random.normal(0.5, 0.2)

            if sf1 == sf2:
                donor1, donor2 = parent[p1, :], parent[p2, :]
                partner_is_base = False
            else:
                # GAN generation: the donor of parent p1 is the image of p1
                # inside the partner task's generator, and vice versa
                p1r = rank[p1] % n
                p2r = rank[p2] % n
                donor1 = gan_off[(sf2, sf1)][p1r, :]
                donor2 = gan_off[(sf1, sf2)][p2r, :]
                partner_is_base = True

            off_dec1 = self._de_offspring(population, prepop, sf1, parent[p1, :],
                                          donor1, f_scale, partner_is_base)
            off_dec2 = self._de_offspring(population, prepop, sf2, parent[p2, :],
                                          donor2, f_scale, partner_is_base)

            # Imitation
            off_sfs[count] = sf1
            off_sfs[count + 1] = sf2
            off_decs[count, :] = np.clip(off_dec1, 0, 1)
            off_decs[count + 1, :] = np.clip(off_dec2, 0, 1)
            count += 2

        return off_decs, off_sfs

    def _de_offspring(self, population, prepop, sf, par_dec, donor, f_scale, partner_is_base):
        """
        One DE/rand-or-best/1 offspring with the difference against the previous population.

        Parameters
        ----------
        population : list of np.ndarray
            Current population of each task
        prepop : list of np.ndarray
            Population of each task at the previous generation
        sf : int
            Skill factor (task index) of the parent
        par_dec : np.ndarray
            Parent decision vector
        donor : np.ndarray
            Vector added to the base: the parent itself for a same-task pair,
            its GAN image for a cross-task pair
        f_scale : float
            DE scaling factor, shared by the two children of a pair
        partner_is_base : bool
            If True the binomial crossover partner is the base vector (cross-task
            pair), otherwise it is the parent itself (same-task pair)

        Returns
        -------
        off_dec : np.ndarray
            Offspring decision vector, not yet clipped
        """
        n = population[sf].shape[0]
        r1 = np.random.randint(n)
        r2 = np.random.randint(n)
        if np.random.rand() < self.pp:
            base = population[sf][r1, :]
        else:
            base = population[sf][0, :]
        off_dec = base + f_scale * (donor - prepop[sf][r2, :])
        return self._de_crossover(off_dec, base if partner_is_base else par_dec, self.CR)

    @staticmethod
    def _de_crossover(off_dec, par_dec, cr):
        """
        Binomial crossover of MTO-Platform ``DE_Crossover``.

        Parameters
        ----------
        off_dec : np.ndarray
            Mutant vector, shape (d,)
        par_dec : np.ndarray
            Vector supplying the non-inherited genes, shape (d,)
        cr : float
            Crossover rate

        Returns
        -------
        off_dec : np.ndarray
            Trial vector, shape (d,)
        """
        d = off_dec.shape[0]
        replace = np.random.rand(d) > cr
        replace[np.random.randint(d)] = False
        off_dec = off_dec.copy()
        off_dec[replace] = par_dec[replace]
        return off_dec
