"""
Multifactorial Evolutionary Algorithm with Single-Step Generative Model (MFEA-SSG)

This module implements MFEA-SSG for expensive multi-task optimization using a diffusion-based
generative model with knowledge distillation for single-step inference.

References
----------
    [1] R. Wang, X. Feng, H. Yu, Y. Tan, and E. M. K. Lai, "Meta-Learning Inspired Single-Step Generative Model for Expensive Multitask Optimization Problems," IEEE Transactions on Evolutionary Computation, 2025.

Notes
-----
Reproduction of paper Table II (CEC17-MTSO, 20 runs): use n=50 (the paper's
"population size 100" counts the population shared across both tasks) and
max_nfes=400 with the remaining defaults. The paper text states E=5 training
epochs, but each refinement sees at most one mini-batch (elites < batch_size),
so 5 epochs = 5 gradient steps per refinement; 50 epochs per refinement with
cumulative (progressive) training is required to reach the reported accuracy.

Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2025.12.01
Version: 1.1
"""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from ddmtolab.Methods.Algo_Methods.algo_utils import *
from ddmtolab.Algorithms.MTSO.MFEA import mfea_selection


# ============================================================================
# Neural Network Components
# ============================================================================

class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional embedding for diffusion timestep."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half = self.dim // 2
        emb = np.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device, dtype=torch.float32) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb


class ResBlock(nn.Module):
    """Residual block with two conv layers and time embedding injection."""

    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.time_mlp = nn.Linear(time_dim, out_ch)
        self.residual = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x, t_emb):
        h = self.act(self.bn1(self.conv1(x)))
        t = self.act(self.time_mlp(t_emb))[:, :, None, None]
        h = h + t
        h = self.bn2(self.conv2(h))
        return self.act(h + self.residual(x))


class SelfAttention(nn.Module):
    """Self-attention block for feature maps."""

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.out = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        xn = self.norm(x)
        q = self.q(xn).view(b, c, -1)
        k = self.k(xn).view(b, c, -1)
        v = self.v(xn).view(b, c, -1)
        attn = torch.softmax(torch.bmm(q.transpose(1, 2), k) / (c ** 0.5), dim=-1)
        out = torch.bmm(v, attn.transpose(1, 2)).view(b, c, h, w)
        return x + self.out(out)


class TeacherUNet(nn.Module):
    """U-Net teacher model for diffusion-based denoising with attention (10 ResBlocks, 4 Attention)."""

    def __init__(self, in_ch=1, base_ch=64, time_dim=128):
        super().__init__()
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
        )
        self.conv_in = nn.Conv2d(in_ch, base_ch, 3, padding=1)
        # Encoder 1: 2 ResBlocks + 1 Attention
        self.enc1_res1 = ResBlock(base_ch, base_ch, time_dim)
        self.enc1_res2 = ResBlock(base_ch, base_ch, time_dim)
        self.enc1_attn = SelfAttention(base_ch)
        self.down1 = nn.Conv2d(base_ch, base_ch, 3, stride=2, padding=1)
        # Encoder 2: 2 ResBlocks + 1 Attention
        self.enc2_res1 = ResBlock(base_ch, base_ch * 2, time_dim)
        self.enc2_res2 = ResBlock(base_ch * 2, base_ch * 2, time_dim)
        self.enc2_attn = SelfAttention(base_ch * 2)
        self.down2 = nn.Conv2d(base_ch * 2, base_ch * 2, 3, stride=2, padding=1)
        # Bridge: 2 ResBlocks + 1 Attention
        self.bridge_res1 = ResBlock(base_ch * 2, base_ch * 4, time_dim)
        self.bridge_attn = SelfAttention(base_ch * 4)
        self.bridge_res2 = ResBlock(base_ch * 4, base_ch * 4, time_dim)
        # Decoder 1: 2 ResBlocks + 1 Attention
        self.up1 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 4, stride=2, padding=1)
        self.dec1_res1 = ResBlock(base_ch * 4, base_ch * 2, time_dim)
        self.dec1_res2 = ResBlock(base_ch * 2, base_ch * 2, time_dim)
        self.dec1_attn = SelfAttention(base_ch * 2)
        # Decoder 2: 2 ResBlocks
        self.up2 = nn.ConvTranspose2d(base_ch * 2, base_ch, 4, stride=2, padding=1)
        self.dec2_res1 = ResBlock(base_ch * 2, base_ch, time_dim)
        self.dec2_res2 = ResBlock(base_ch, base_ch, time_dim)
        self.conv_out = nn.Conv2d(base_ch, in_ch, 1)

    def forward(self, x, t):
        t_emb = self.time_embed(t)
        h = self.conv_in(x)
        # Encoder 1
        h = self.enc1_res1(h, t_emb)
        h1 = self.enc1_attn(self.enc1_res2(h, t_emb))
        h = self.down1(h1)
        # Encoder 2
        h = self.enc2_res1(h, t_emb)
        h2 = self.enc2_attn(self.enc2_res2(h, t_emb))
        h = self.down2(h2)
        # Bridge
        h = self.bridge_res1(h, t_emb)
        h = self.bridge_attn(h)
        h = self.bridge_res2(h, t_emb)
        # Decoder 1 (skip from enc2)
        h = self.up1(h)
        h = h[:, :, :h2.shape[2], :h2.shape[3]]
        h = torch.cat([h, h2], dim=1)
        h = self.dec1_res1(h, t_emb)
        h = self.dec1_attn(self.dec1_res2(h, t_emb))
        # Decoder 2 (skip from enc1)
        h = self.up2(h)
        h = h[:, :, :h1.shape[2], :h1.shape[3]]
        h = torch.cat([h, h1], dim=1)
        h = self.dec2_res1(h, t_emb)
        h = self.dec2_res2(h, t_emb)
        return self.conv_out(h)


class StudentUNet(nn.Module):
    """Lightweight student model for single-step generation (no attention)."""

    def __init__(self, in_ch=1, base_ch=64, time_dim=128):
        super().__init__()
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
        )
        self.conv_in = nn.Conv2d(in_ch, base_ch, 3, padding=1)
        self.enc_res = ResBlock(base_ch, base_ch, time_dim)
        self.down = nn.Conv2d(base_ch, base_ch, 3, stride=2, padding=1)
        self.bridge = ResBlock(base_ch, base_ch * 2, time_dim)
        self.up = nn.ConvTranspose2d(base_ch * 2, base_ch, 4, stride=2, padding=1)
        self.dec_res = ResBlock(base_ch * 2, base_ch, time_dim)
        self.conv_out = nn.Conv2d(base_ch, in_ch, 1)

    def forward(self, x, t):
        t_emb = self.time_embed(t)
        h = self.conv_in(x)
        h1 = self.enc_res(h, t_emb)
        h = self.down(h1)
        h = self.bridge(h, t_emb)
        h = self.up(h)
        h = h[:, :, :h1.shape[2], :h1.shape[3]]
        h = torch.cat([h, h1], dim=1)
        h = self.dec_res(h, t_emb)
        return self.conv_out(h)


# ============================================================================
# Diffusion Utilities
# ============================================================================

def get_diffusion_schedule(n_steps=100):
    """Create linear beta schedule and precompute alpha values."""
    betas = np.linspace(1e-4, 0.02, n_steps)
    alphas = 1.0 - betas
    alpha_bars = np.cumprod(alphas)
    return betas, alphas, alpha_bars


def diffusion_forward(x0, t, alpha_bars, device):
    """Add noise to clean data according to forward diffusion process."""
    alpha_bar_t = torch.tensor(alpha_bars[t], dtype=torch.float32, device=device).view(-1, 1, 1, 1)
    noise = torch.randn_like(x0)
    x_t = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise
    return x_t, noise


def generate_with_student(student, elite_data, grid_h, grid_w, grid_dim,
                          alpha_bars, device, n_samples=1, denoise_t=50):
    """
    Generate samples using single-step student model with elite-guided denoising.

    Implements Algorithm 1 line 6 ("Generate decisions Dec from noise Z using G"):
    noise Z enters through the forward diffusion of an elite solution
    (x_t = sqrt(a_t) x0 + sqrt(1 - a_t) Z), and the student maps the noisy input
    to a denoised sample in one forward pass (paper Sec. III-B). Sampling x_t as
    pure noise instead is out-of-distribution under the short schedule
    (alpha_bar_N ~ 0.37 for N=100) and empirically degrades results by 3-50x on
    CEC17-MTSO P1, so the elite-guided form is used. Dimension shuffling
    (meta-learning inspired) is applied before noising and inverse-shuffled
    after denoising.
    """
    # Keep the denoising timestep inside the diffusion schedule
    # (n_diffusion_steps is configurable and may be smaller than the default t)
    denoise_t = min(denoise_t, len(alpha_bars) - 1)

    student.eval()
    with torch.no_grad():
        indices = np.random.randint(0, len(elite_data), size=n_samples)
        x0_np = elite_data[indices].copy()

        # Random dimension shuffling — track permutations for inverse
        shuffle_perms = []
        for i in range(len(x0_np)):
            perm = np.random.permutation(x0_np.shape[1])
            shuffle_perms.append(perm)
            x0_np[i] = x0_np[i][perm]

        x0 = torch.tensor(x0_np, dtype=torch.float32, device=device).view(-1, 1, grid_h, grid_w)

        # Forward diffusion: add noise at timestep denoise_t
        t = torch.full((n_samples,), denoise_t, device=device, dtype=torch.long)
        alpha_bar_t = torch.tensor(alpha_bars[denoise_t], dtype=torch.float32, device=device)
        noise = torch.randn_like(x0)
        x_t = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise

        # Single-step denoising
        pred_noise = student(x_t, t)
        x_denoised = (x_t - torch.sqrt(1 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)
        x_denoised = torch.clamp(x_denoised, 0.0, 1.0)

    result = x_denoised.cpu().numpy().reshape(n_samples, -1)[:, :grid_dim]

    # Inverse shuffle to restore original dimension ordering
    for i in range(n_samples):
        inv_perm = np.argsort(shuffle_perms[i])
        result[i] = result[i][inv_perm[:grid_dim]]

    return result


# ============================================================================
# Training Functions
# ============================================================================

def train_teacher(teacher, train_data, alpha_bars, n_steps, device, grid_h, grid_w,
                  epochs=5, batch_size=512, lr=5e-4):
    """
    Train the teacher diffusion model (Algorithm 2 in paper).

    Applies random dimension shuffling and reshaping to grid_h x grid_w images.
    """
    teacher.train()
    optimizer = optim.Adam(teacher.parameters(), lr=lr, betas=(0.9, 0.9999))
    effective_bs = min(batch_size, len(train_data))

    for epoch in range(epochs):
        perm = np.random.permutation(len(train_data))
        for start in range(0, len(perm), effective_bs):
            batch_idx = perm[start:start + effective_bs]
            batch = train_data[batch_idx].copy()

            for i in range(len(batch)):
                shuffle_idx = np.random.permutation(batch.shape[1])
                batch[i] = batch[i][shuffle_idx]

            x0 = torch.tensor(batch, dtype=torch.float32, device=device)
            x0 = x0.view(-1, 1, grid_h, grid_w)

            t = torch.randint(0, n_steps, (x0.shape[0],), device=device)
            x_t, noise = diffusion_forward(x0, t.cpu().numpy(), alpha_bars, device)

            pred_noise = teacher(x_t, t)
            loss = nn.functional.mse_loss(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def distill_student(teacher, student, train_data, alpha_bars, n_steps, device, grid_h, grid_w,
                    epochs=5, batch_size=512, lr=5e-4):
    """
    Knowledge distillation from teacher to student (Algorithm 3 in paper).

    Student learns to mimic teacher's noise predictions for single-step generation.
    No dimension shuffling is applied during distillation (per Algorithm 3).
    """
    teacher.eval()
    student.train()
    optimizer = optim.Adam(student.parameters(), lr=lr, betas=(0.9, 0.9999))
    effective_bs = min(batch_size, len(train_data))

    for epoch in range(epochs):
        perm = np.random.permutation(len(train_data))
        for start in range(0, len(perm), effective_bs):
            batch_idx = perm[start:start + effective_bs]
            batch = train_data[batch_idx].copy()

            x0 = torch.tensor(batch, dtype=torch.float32, device=device)
            x0 = x0.view(-1, 1, grid_h, grid_w)

            t = torch.randint(0, n_steps, (x0.shape[0],), device=device)
            x_t, _ = diffusion_forward(x0, t.cpu().numpy(), alpha_bars, device)

            with torch.no_grad():
                teacher_pred = teacher(x_t, t)

            student_pred = student(x_t, t)
            loss = nn.functional.mse_loss(student_pred, teacher_pred)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


# ============================================================================
# MFEA-SSG Algorithm
# ============================================================================

class MFEA_SSG:
    """
    Multifactorial Evolutionary Algorithm with Single-Step Generative Model.

    Follows the MFEA architecture with a diffusion-based generative model replacing
    crossover in early generations. Knowledge distillation compresses the teacher
    model into a lightweight student for single-step inference.

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
        'expensive': 'True',
        'knowledge_transfer': 'True',
        'n': 'equal',
        'max_nfes': 'equal'
    }

    @classmethod
    def get_algorithm_information(cls, print_info=True):
        return get_algorithm_information(cls, print_info)

    def __init__(self, problem, n=None, max_nfes=None, rmp=0.3, muc=2, mum=5,
                 max_gen=None, refine_freq=1, n_pairs_per_gen=None,
                 n_diffusion_steps=100, train_epochs=50, distill_epochs=50,
                 batch_size=512, lr=5e-4, base_ch=64,
                 save_data=True, save_path='./Data', name='MFEA-SSG', disable_tqdm=True):
        """
        Initialize MFEA-SSG algorithm.

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
        max_gen : int, optional
            Maximum generation for generative phase. By default the generative
            model drives the search for the entire run (paper Fig. 4 shows the
            student model still active at Gen=100 under the 400-FE budget);
            pass a value to switch to pure GA crossover after that generation.
        refine_freq : int, optional
            Refinement frequency tau for generative model (default: 1)
        n_diffusion_steps : int, optional
            Number of diffusion timesteps N (default: 100)
        train_epochs : int, optional
            Training epochs for teacher model (default: 50)
        distill_epochs : int, optional
            Knowledge distillation epochs (default: 50)
        batch_size : int, optional
            Mini-batch size for training (default: 512)
        lr : float, optional
            Learning rate for Adam optimizer (default: 5e-4)
        base_ch : int, optional
            Base channel count for U-Net models (default: 64)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './Data')
        name : str, optional
            Name for the experiment (default: 'MFEA-SSG')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n = n if n is not None else 100
        self.max_nfes = max_nfes if max_nfes is not None else 10000
        self.rmp = rmp
        self.muc = muc
        self.mum = mum
        self.max_gen = max_gen
        self.refine_freq = refine_freq
        self.n_pairs_per_gen = n_pairs_per_gen  # None = auto (nt pairs per gen)
        self.n_diffusion_steps = n_diffusion_steps
        self.train_epochs = train_epochs
        self.distill_epochs = distill_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.base_ch = base_ch
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def _prepare_model_data(self, pop_decs, pop_objs, grid_dim, top_ratio=0.5):
        """
        Prepare training data for the generative model from population.

        Collects top-performing individuals from all tasks and pads to grid_dim.
        """
        all_data = []
        for i in range(len(pop_decs)):
            task_decs = pop_decs[i]
            task_objs = pop_objs[i]
            n_total = len(task_objs)
            n_elite = max(int(n_total * top_ratio), min(n_total, 2))
            indices = np.argsort(task_objs.flatten())[:n_elite]
            elite = task_decs[indices]

            # Pad to grid_dim if needed (constant 0.5 padding for unused grid cells)
            if elite.shape[1] < grid_dim:
                pad = np.full((elite.shape[0], grid_dim - elite.shape[1]), 0.5)
                elite = np.hstack([elite, pad])
            elif elite.shape[1] > grid_dim:
                elite = elite[:, :grid_dim]

            all_data.append(elite)
        return np.vstack(all_data)

    def optimize(self):
        """
        Execute the MFEA-SSG algorithm (Algorithm 1 in paper).

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
        max_dim = max(dims)

        # Grid dimensions for 2D reshape: fixed grid_h=5, grid_w=ceil(max_dim/5)
        # e.g., 50D→5×10, 15D→5×3, 13D→5×3 (truncate 2 extra dims)
        grid_h = 5
        grid_w = max((max_dim + grid_h - 1) // grid_h, 1)
        grid_dim = grid_h * grid_w

        # Device setup
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Diffusion schedule
        betas, alphas, alpha_bars = get_diffusion_schedule(self.n_diffusion_steps)

        # Initialize teacher and student models
        teacher = TeacherUNet(in_ch=1, base_ch=self.base_ch, time_dim=128).to(device)
        student = StudentUNet(in_ch=1, base_ch=self.base_ch, time_dim=128).to(device)

        # ============================================================
        # Line 1: Initialize population P; gen <- 0
        # ============================================================
        decs = initialization(problem, n)
        objs, cons = evaluation(problem, decs)
        nfes = n * nt
        all_decs, all_objs, all_cons = init_history(decs, objs, cons)

        # Transform to unified search space
        pop_decs, pop_cons = space_transfer(problem=problem, decs=decs, cons=cons, type='uni', padding='mid')
        pop_objs = objs
        pop_sfs = [np.full((n, 1), fill_value=i) for i in range(nt)]

        # Train initial generative model G on the initial population
        model_data = self._prepare_model_data(pop_decs, pop_objs, grid_dim, top_ratio=1.0)
        train_teacher(teacher, model_data, alpha_bars, self.n_diffusion_steps, device,
                      grid_h, grid_w, self.train_epochs, self.batch_size, self.lr)
        distill_student(teacher, student, model_data, alpha_bars, self.n_diffusion_steps, device,
                        grid_h, grid_w, self.distill_epochs, self.batch_size, self.lr)

        # Determine pairs per generation (controls offspring count per generation)
        # Default: nt pairs → ~nt offspring per gen → many generations for model refinement
        n_pairs_per_gen = self.n_pairs_per_gen if self.n_pairs_per_gen is not None else nt

        # MaxGen: by default the generative model is used for the whole run
        # (validated against paper Table II; an early switch to pure GA on the
        # collapsed population stalls refinement and worsens results)
        max_gen_generative = self.max_gen if self.max_gen is not None else float('inf')

        gen = 0

        pbar = tqdm(total=max_nfes, initial=nfes, desc=f"{self.name}", disable=self.disable_tqdm)

        # ============================================================
        # Line 2: WHILE termination condition not met
        # ============================================================
        while nfes < max_nfes:

            # Merge populations from all tasks into single arrays
            pop_decs, pop_objs, pop_cons, pop_sfs = vstack_groups(pop_decs, pop_objs, pop_cons, pop_sfs)
            uni_dim = pop_decs.shape[1]
            n_cons_uni = pop_cons.shape[1]

            # Precompute elite data ONCE per generation for generative model
            if gen <= max_gen_generative:
                elite_data = self._prepare_model_data(
                    [pop_decs[pop_sfs.flatten() == t] for t in range(nt)],
                    [pop_objs[pop_sfs.flatten() == t] for t in range(nt)],
                    grid_dim, top_ratio=0.5)

            off_decs_list = []
            off_objs_list = []
            off_sfs_list = []

            # Line 4: FOR each pair of parents (p1, p2) selected from P
            shuffled_index = np.random.permutation(pop_decs.shape[0])
            max_pairs = min(n_pairs_per_gen, len(shuffled_index) // 2)

            for pair_idx in range(max_pairs):
                i = pair_idx * 2
                if nfes >= max_nfes:
                    break

                p1 = shuffled_index[i]
                p2 = shuffled_index[i + 1]
                sf1 = int(pop_sfs[p1].item())
                sf2 = int(pop_sfs[p2].item())

                # Line 5: IF gen <= MaxGen AND (Same task OR rand < RMP)
                if gen <= max_gen_generative and (sf1 == sf2 or np.random.rand() < self.rmp):
                    # Lines 6-9: Generate from student model, mutate, create one offspring
                    dec_gen = generate_with_student(
                        student, elite_data, grid_h, grid_w, grid_dim,
                        alpha_bars, device, n_samples=1)
                    dec_gen = dec_gen.flatten()

                    # Truncate/pad to unified space dimension
                    if len(dec_gen) >= uni_dim:
                        dec_uni = dec_gen[:uni_dim]
                    else:
                        dec_uni = np.concatenate([dec_gen, np.random.rand(uni_dim - len(dec_gen))])
                    dec_uni = np.clip(dec_uni, 0.0, 1.0)

                    # Apply mutation (Line 7)
                    dec_mut = mutation(dec_uni, mu=self.mum)

                    # Assign one parent's task factor (Line 9)
                    assigned_sf = np.random.choice([sf1, sf2])

                    # Evaluate on assigned task
                    off_dec_trimmed = dec_mut[:dims[assigned_sf]]
                    off_obj, off_con = evaluation_single(problem, off_dec_trimmed, assigned_sf)
                    nfes += 1
                    pbar.update(1)

                    off_decs_list.append(dec_mut.reshape(1, -1))
                    off_objs_list.append(off_obj)
                    off_sfs_list.append(np.array([[assigned_sf]]))

                else:
                    # Lines 11-14: GA crossover, create two offspring
                    off_dec1, off_dec2 = crossover(pop_decs[p1, :], pop_decs[p2, :], mu=self.muc)

                    # Assign task factors
                    sf_o1 = np.random.choice([sf1, sf2])
                    sf_o2 = sf1 if sf_o1 == sf2 else sf2

                    for off_dec, sf in [(off_dec1, sf_o1), (off_dec2, sf_o2)]:
                        if nfes >= max_nfes:
                            break

                        off_dec_trimmed = off_dec[:dims[sf]]
                        off_obj, off_con = evaluation_single(problem, off_dec_trimmed, sf)
                        nfes += 1
                        pbar.update(1)

                        off_decs_list.append(off_dec.reshape(1, -1))
                        off_objs_list.append(off_obj)
                        off_sfs_list.append(np.array([[sf]]))

            if len(off_decs_list) == 0:
                break

            # Stack offspring (cons use unified space dimension)
            off_decs = np.vstack(off_decs_list)
            off_objs = np.vstack(off_objs_list)
            off_cons = np.zeros((len(off_decs_list), n_cons_uni))
            off_sfs = np.vstack(off_sfs_list)

            # Line 17: Evaluate Offspring, update population P by selecting from P ∪ Offspring
            pop_decs, pop_objs, pop_cons, pop_sfs = vstack_groups(
                (pop_decs, off_decs), (pop_objs, off_objs), (pop_cons, off_cons), (pop_sfs, off_sfs)
            )

            pop_decs, pop_objs, pop_cons, pop_sfs = mfea_selection(
                pop_decs, pop_objs, pop_cons, pop_sfs, n, nt)

            # Transform back to native search space for history
            decs, cons = space_transfer(problem, decs=pop_decs, cons=pop_cons, type='real')
            append_history(all_decs, decs, all_objs, pop_objs, all_cons, cons)

            # Lines 18-20: Progressively refine generative model G (IF mod(gen, tau) == 0)
            if gen % self.refine_freq == 0:
                model_data = self._prepare_model_data(pop_decs, pop_objs, grid_dim, top_ratio=0.5)
                train_teacher(teacher, model_data, alpha_bars, self.n_diffusion_steps, device,
                              grid_h, grid_w, self.train_epochs, self.batch_size, self.lr)
                distill_student(teacher, student, model_data, alpha_bars, self.n_diffusion_steps, device,
                                grid_h, grid_w, self.distill_epochs, self.batch_size, self.lr)

            # Line 21: gen <- gen + 1
            gen += 1

        pbar.close()
        runtime = time.time() - start_time

        results = build_save_results(all_decs=all_decs, all_objs=all_objs, runtime=runtime, max_nfes=max_nfes_per_task,
                                     all_cons=all_cons, bounds=problem.bounds, save_path=self.save_path,
                                     filename=self.name, save_data=self.save_data)

        return results
