import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


# ----------------------------------------------------
# 1. TGP Kernel Function - RBFKernel with Task Transfer
# ----------------------------------------------------
class TGPKernel(torch.nn.Module):
    """
    Transfer Gaussian Process Kernel with ARD.

    This kernel extends RBF kernel to handle multi-task learning by introducing
    a correlation factor λ between samples from different tasks.

    The kernel matrix element K_mn is defined as:
        K_mn = λ * k(x_n, x_m)  if x_n, x_m from different tasks
        K_mn = k(x_n, x_m)      otherwise

    where k(x_n, x_m) is the standard RBF kernel with ARD.

    Parameters
    ----------
    input_dim : int
        Number of input dimensions (excluding task indicator).
    lengthscale : float, array-like, or None, default=None
        Initial lengthscale parameter(s).
    output_variance : float, default=1.0
        Initial output variance parameter (θ_l in paper).
    b_init : float, default=0.0
        Initial value for parameter b, which controls task correlation λ.
        λ = 2 / (1 + exp(b)) - 1, range: (-1, 1)

    Attributes
    ----------
    input_dim : int
        Number of input dimensions.
    log_l : torch.nn.Parameter, shape (D,)
        Log-transformed lengthscale parameters for each dimension (θ_d).
    log_theta_l : torch.nn.Parameter, shape (1,)
        Log-transformed output variance parameter (θ_l).
    b : torch.nn.Parameter, shape (1,)
        Parameter controlling inter-task correlation λ.
    """

    def __init__(self, input_dim, lengthscale=None, output_variance=1.0, b_init=0.0):
        super().__init__()
        self.input_dim = input_dim

        # Initialize lengthscale for each dimension
        if lengthscale is None:
            lengthscale = np.ones(input_dim)
        elif np.isscalar(lengthscale):
            lengthscale = np.full(input_dim, lengthscale)
        else:
            lengthscale = np.array(lengthscale)
            if lengthscale.shape[0] != input_dim:
                raise ValueError(f"lengthscale must have shape ({input_dim},), got {lengthscale.shape}")

        # Define log-transformed hyperparameters as learnable parameters
        self.log_l = torch.nn.Parameter(torch.tensor(np.log(lengthscale), dtype=torch.float32))
        self.log_theta_l = torch.nn.Parameter(torch.tensor([np.log(output_variance)], dtype=torch.float32))

        # Task correlation parameter b (λ = 2/(1+exp(b)) - 1)
        self.b = torch.nn.Parameter(torch.tensor([b_init], dtype=torch.float32))

    def compute_lambda(self):
        """
        Compute task correlation factor λ from parameter b.

        λ = 2 / (1 + exp(b)) - 1

        When b → -∞: λ → 1 (high correlation)
        When b → 0:  λ → 0 (no correlation)
        When b → +∞: λ → -1 (negative correlation, clipped to 0)

        Returns
        -------
        lambda_val : torch.Tensor, scalar
            Task correlation factor, clipped to [0, 1] as in paper.
        """
        lambda_val = 2.0 / (1.0 + torch.exp(self.b)) - 1.0
        # Eliminate negative correlation as mentioned in paper
        lambda_val = torch.clamp(lambda_val, min=0.0, max=1.0)
        return lambda_val

    def forward(self, X1, X2, task1=None, task2=None):
        """
        Compute the TGP kernel matrix K(X1, X2).

        Parameters
        ----------
        X1 : torch.Tensor, shape (N1, D+1) or (N1, D)
            First set of input points. If shape is (N1, D+1), last column is task indicator.
        X2 : torch.Tensor, shape (N2, D+1) or (N2, D)
            Second set of input points. If shape is (N2, D+1), last column is task indicator.
        task1 : torch.Tensor, shape (N1,), optional
            Task indicators for X1 (0=target, 1=source). If None, extracted from X1[:, -1].
        task2 : torch.Tensor, shape (N2,), optional
            Task indicators for X2 (0=target, 1=source). If None, extracted from X2[:, -1].

        Returns
        -------
        K : torch.Tensor, shape (N1, N2)
            Kernel matrix with task transfer correlation.
        """
        # Extract task indicators and features
        if task1 is None:
            if X1.shape[1] == self.input_dim + 1:
                task1 = X1[:, -1]
                X1_features = X1[:, :-1]
            else:
                # Assume all from target task if no indicator
                task1 = torch.zeros(X1.shape[0], dtype=torch.float32)
                X1_features = X1
        else:
            # task1 provided, always extract features
            if X1.shape[1] == self.input_dim + 1:
                X1_features = X1[:, :-1]
            else:
                X1_features = X1

        if task2 is None:
            if X2.shape[1] == self.input_dim + 1:
                task2 = X2[:, -1]
                X2_features = X2[:, :-1]
            else:
                # Assume all from target task if no indicator
                task2 = torch.zeros(X2.shape[0], dtype=torch.float32)
                X2_features = X2
        else:
            # task2 provided, always extract features
            if X2.shape[1] == self.input_dim + 1:
                X2_features = X2[:, :-1]
            else:
                X2_features = X2

        # Recover positive parameters
        l = torch.exp(self.log_l)  # shape: (D,)
        theta_l = torch.exp(self.log_theta_l)

        # Compute base RBF kernel
        # X1_features: (N1, D) -> (N1, 1, D)
        X1_expanded = X1_features.unsqueeze(1)
        # X2_features: (N2, D) -> (1, N2, D)
        X2_expanded = X2_features.unsqueeze(0)

        # Calculate difference (N1, N2, D)
        diff = X1_expanded - X2_expanded

        # Scale each dimension by its lengthscale (ARD)
        scaled_diff = diff / l

        # Calculate squared distance (N1, N2)
        sq_dist = torch.sum(scaled_diff ** 2, dim=2)

        # Apply RBF formula: k = θ_l * exp(-0.5 * sum_d[(x1_d - x2_d)^2 / θ_d^2])
        K_base = theta_l * torch.exp(-0.5 * sq_dist)

        # Compute task correlation factor λ
        lambda_val = self.compute_lambda()

        # Create task difference matrix (N1, N2)
        # 1 if tasks are different, 0 if same
        task1_expanded = task1.unsqueeze(1)  # (N1, 1)
        task2_expanded = task2.unsqueeze(0)  # (1, N2)
        different_tasks = (task1_expanded != task2_expanded).float()

        # Apply task correlation: K_mn = λ * k(x_n, x_m) if different tasks, else k(x_n, x_m)
        K = K_base * (1.0 - different_tasks + lambda_val * different_tasks)

        return K


# ----------------------------------------------------
# 2. TGP Regression Model
# ----------------------------------------------------
class TGPRegression:
    """
    Transfer Gaussian Process Regression model with ARD kernel.

    This model extends standard GP to leverage information from related source tasks
    to improve prediction on the target task. The correlation between tasks is
    automatically learned through the parameter b (which controls λ).

    Usage:
        tgp = TGPRegression()
        # X_train: shape (N, D+1), last column is task indicator (0=target, 1=source)
        tgp.train(X_train, y_train)
        # X_test: shape (N*, D) or (N*, D+1), predictions for target task
        mean, cov = tgp.predict(X_test)

    Parameters
    ----------
    kernel : TGPKernel, optional
        Kernel function for computing covariances. If None, will be created in train().
    noise_variance_target : float, default=0.1
        Initial observation noise variance for target task (σ²_T1).
    noise_variance_source : float, default=0.1
        Initial observation noise variance for source task (σ²_T2).

    Attributes
    ----------
    kernel : TGPKernel or None
        The kernel function.
    log_sn2_target : torch.nn.Parameter or None
        Log-transformed noise variance for target task.
    log_sn2_source : torch.nn.Parameter or None
        Log-transformed noise variance for source task.
    params : list or None
        List of all learnable parameters.
    X : torch.Tensor or None
        Training input data (including task indicators).
    y : torch.Tensor or None
        Training target data.
    task_indicators : torch.Tensor or None
        Task indicators (0=target, 1=source).
    N : int or None
        Number of training points.
    """

    def __init__(self, kernel=None, noise_variance_target=0.1, noise_variance_source=0.1):
        self.kernel = kernel
        self.initial_noise_variance_target = noise_variance_target
        self.initial_noise_variance_source = noise_variance_source

        if kernel is not None:
            self.log_sn2_target = torch.nn.Parameter(
                torch.tensor([np.log(noise_variance_target)], dtype=torch.float32)
            )
            self.log_sn2_source = torch.nn.Parameter(
                torch.tensor([np.log(noise_variance_source)], dtype=torch.float32)
            )
            self.params = list(self.kernel.parameters()) + [self.log_sn2_target, self.log_sn2_source]
        else:
            self.log_sn2_target = None
            self.log_sn2_source = None
            self.params = None

        # Training data (initialized to None)
        self.X = None
        self.y = None
        self.task_indicators = None
        self.N = None

        self.verbose_default = True

    def set_data(self, X, y):
        """
        Load training data.

        Parameters
        ----------
        X : torch.Tensor, shape (N, D+1)
            Training input features with task indicator in last column.
            Task indicator: 0 = target task, 1 = source task.
        y : torch.Tensor, shape (N,)
            Training target values.
        """
        self.X = X.to(torch.float32)
        self.y = y.to(torch.float32)
        self.task_indicators = X[:, -1]
        self.N = X.shape[0]

    def negative_mll(self):
        """
        Calculate negative marginal log-likelihood (NLL) for TGP.

        The NLL follows Equation (14) in the paper:
        -log p(y_T1 | θ) = 0.5 * log|C_T1| + 0.5 * (y_T1 - μ_T1)^T C_T1^(-1) (y_T1 - μ_T1) + const

        Returns
        -------
        nll : torch.Tensor, scalar
            Negative marginal log-likelihood value.
        """
        # Recover positive observation noise variances
        sn2_target = torch.exp(self.log_sn2_target)
        sn2_source = torch.exp(self.log_sn2_source)

        # Get task indicators
        task_ind = self.task_indicators

        # Compute kernel matrix K with task correlation
        K = self.kernel.forward(self.X, self.X, task_ind, task_ind)

        # Add noise to diagonal based on task
        # Noise matrix: σ²_T1 for target samples, σ²_T2 for source samples
        noise_vector = torch.where(task_ind == 0, sn2_target, sn2_source)
        K_with_noise = K + torch.diag(noise_vector)

        # Compute Cholesky decomposition
        try:
            L = torch.linalg.cholesky(K_with_noise)
        except RuntimeError as e:
            if 'cholesky' in str(e).lower():
                return torch.tensor(1e10, dtype=torch.float32)
            raise e

        # Compute log determinant: log|K| = 2 * sum(log(diag(L)))
        log_det_K = 2.0 * torch.sum(torch.log(torch.diag(L)))

        # Compute y^T K^(-1) y (Data fit term)
        v = torch.linalg.solve(L, self.y.unsqueeze(-1))
        alpha = torch.linalg.solve(L.T, v)
        data_fit_term = torch.sum(self.y * alpha.squeeze())

        # Combine NLL terms
        nll = 0.5 * data_fit_term + 0.5 * log_det_K + 0.5 * self.N * torch.log(
            torch.tensor(2.0 * torch.pi)
        )

        return nll

    def train(self, X=None, y=None, lengthscale=None, output_variance=1.0,
              noise_variance_target=None, noise_variance_source=None, b_init=0.0,
              n_restarts=5, verbose=True, print_every=20):
        """
        Train TGP model (optimize hyperparameters).

        Parameters
        ----------
        X : torch.Tensor, shape (N, D+1), optional
            Training input features with task indicator in last column.
        y : torch.Tensor, shape (N,), optional
            Training target values.
        lengthscale : float, array-like, or None, default=None
            Initial lengthscale parameter(s).
        output_variance : float, default=1.0
            Initial output variance parameter (θ_l).
        noise_variance_target : float, optional
            Initial observation noise variance for target task.
        noise_variance_source : float, optional
            Initial observation noise variance for source task.
        b_init : float, default=0.0
            Initial value for parameter b (controls task correlation λ).
        n_restarts : int, default=5
            Number of random restarts for optimization.
        verbose : bool, default=True
            Whether to print training information.
        print_every : int, default=20
            Print progress every N epochs.
        """
        # Load training data if provided
        if X is not None and y is not None:
            self.set_data(X, y)

        if self.X is None or self.y is None:
            raise ValueError("Please provide training data!")

        # Get input dimension (excluding task indicator)
        input_dim = self.X.shape[1] - 1

        # Create kernel if not provided
        if self.kernel is None:
            self.kernel = TGPKernel(
                input_dim=input_dim,
                lengthscale=lengthscale,
                output_variance=output_variance,
                b_init=b_init
            )

        # Set noise variances
        if noise_variance_target is None:
            noise_variance_target = self.initial_noise_variance_target
        if noise_variance_source is None:
            noise_variance_source = self.initial_noise_variance_source

        # Initialize parameters if not already done
        if self.log_sn2_target is None:
            self.log_sn2_target = torch.nn.Parameter(
                torch.tensor([np.log(noise_variance_target)], dtype=torch.float32)
            )
            self.log_sn2_source = torch.nn.Parameter(
                torch.tensor([np.log(noise_variance_source)], dtype=torch.float32)
            )
            self.params = list(self.kernel.parameters()) + [self.log_sn2_target, self.log_sn2_source]

        if verbose:
            print(f"=" * 70)
            print(f"Starting TGP multi-start optimization (n_restarts={n_restarts})")
            n_target = (self.task_indicators == 0).sum().item()
            n_source = (self.task_indicators == 1).sum().item()
            print(f"Data: {n_target} target samples, {n_source} source samples")
            print(f"Parameters: {input_dim} lengthscales + 1 output_var + 2 noise_vars + 1 b = {input_dim + 4}")
            print(f"=" * 70)

        # Save best results
        best_loss = float('inf')
        best_params = None

        # Stage 1: L-BFGS Optimization
        lbfgs_lrs = [0.1, 0.01, 0.001, 0.0001]
        lbfgs_epochs = 100
        lbfgs_success = False

        for current_lr in lbfgs_lrs:
            if verbose:
                print(f"\n{'─' * 70}")
                print(f"Attempting L-BFGS optimization (lr: {current_lr}, epochs: {lbfgs_epochs})")
                print(f"{'─' * 70}")

            restart_result = self._multi_restart_lbfgs(
                n_restarts=n_restarts,
                lr=current_lr,
                epochs=lbfgs_epochs,
                verbose=verbose,
                print_every=print_every
            )

            if restart_result['success']:
                lbfgs_success = True
                if restart_result['best_loss'] < best_loss:
                    best_loss = restart_result['best_loss']
                    best_params = restart_result['best_params']

                if verbose:
                    print(f"\n✓ L-BFGS optimization successful (lr: {current_lr})")
                    print(f"  Best loss: {best_loss:.4f}")
                break
            else:
                if verbose:
                    print(f"\n✗ L-BFGS optimization failed (lr: {current_lr})")

        # Stage 2: Adam Optimization
        if not lbfgs_success:
            if verbose:
                print(f"\n{'=' * 70}")
                print(f"Switching to Adam optimizer")
                print(f"{'=' * 70}")

            adam_lrs = [0.01, 0.001, 0.0001]
            adam_epochs = 2000
            adam_success = False

            for current_lr in adam_lrs:
                if verbose:
                    print(f"\n{'─' * 70}")
                    print(f"Attempting Adam optimization (lr: {current_lr}, epochs: {adam_epochs})")
                    print(f"{'─' * 70}")

                adam_result = self._multi_restart_adam(
                    n_restarts=n_restarts,
                    lr=current_lr,
                    epochs=adam_epochs,
                    verbose=verbose,
                    print_every=print_every
                )

                if adam_result['success']:
                    adam_success = True
                    if adam_result['best_loss'] < best_loss:
                        best_loss = adam_result['best_loss']
                        best_params = adam_result['best_params']

                    if verbose:
                        print(f"\n✓ Adam optimization successful (lr: {current_lr})")
                        print(f"  Best loss: {best_loss:.4f}")
                    break
                else:
                    if verbose:
                        print(f"\n✗ Adam optimization failed (lr: {current_lr})")

            if not adam_success and verbose:
                print(f"\n✗ Warning: All optimization strategies failed!")

        # Load best parameters
        if best_params is not None:
            self._load_params(best_params)
        else:
            if verbose:
                print(f"\n{'=' * 70}")
                print(f"Using default hyperparameters")
                print(f"{'=' * 70}")
            default_params = self._get_default_params()
            self._load_params(default_params)
            with torch.no_grad():
                best_loss = self.negative_mll().item()

        if verbose:
            print(f"\n{'=' * 70}")
            print(f"Optimization complete!")
            if best_loss < float('inf'):
                print(f"Best loss: {best_loss:.4f}")
                self._print_hyperparameters()
            print(f"{'=' * 70}")

    def _get_default_params(self):
        """Get default hyperparameters."""
        X_features = self.X[:, :-1]
        X_range = (X_features.max(dim=0).values - X_features.min(dim=0).values)
        X_range = torch.where(X_range < 1e-6, torch.ones_like(X_range), X_range)

        default_l = 0.2 * X_range.numpy()

        y_var = torch.var(self.y).item()
        if y_var < 1e-6:
            y_var = 1.0

        default_theta_l = y_var
        default_sn2_target = 0.01 * y_var
        default_sn2_source = 0.01 * y_var
        default_b = 0.0

        return {
            'log_l': torch.tensor(np.log(default_l), dtype=torch.float32),
            'log_theta_l': torch.tensor([np.log(default_theta_l)], dtype=torch.float32),
            'b': torch.tensor([default_b], dtype=torch.float32),
            'log_sn2_target': torch.tensor([np.log(default_sn2_target)], dtype=torch.float32),
            'log_sn2_source': torch.tensor([np.log(default_sn2_source)], dtype=torch.float32)
        }

    def _multi_restart_lbfgs(self, n_restarts, lr, epochs, verbose, print_every):
        """Multi-start L-BFGS optimization."""
        best_loss = float('inf')
        best_params = None
        success_count = 0

        for restart in range(n_restarts):
            if verbose:
                print(f"\n[Restart {restart + 1}/{n_restarts}]")

            if restart > 0:
                self._random_init()

            try:
                optimizer = torch.optim.LBFGS(
                    self.params,
                    lr=lr,
                    max_iter=20,
                    history_size=100,
                    line_search_fn='strong_wolfe',
                    tolerance_grad=1e-7,
                    tolerance_change=1e-9
                )

                def closure():
                    optimizer.zero_grad()
                    loss = self.negative_mll()

                    if torch.isnan(loss) or torch.isinf(loss):
                        penalty = sum(torch.sum(p ** 2) for p in self.params)
                        return torch.tensor(1e8, dtype=torch.float32) + penalty

                    if loss.requires_grad:
                        loss.backward()

                    return loss

                for i in range(epochs):
                    try:
                        loss = optimizer.step(closure)

                        if verbose and (i + 1) % print_every == 0:
                            with torch.no_grad():
                                current_loss = self.negative_mll()
                                if not (torch.isnan(current_loss) or torch.isinf(current_loss)):
                                    self._print_progress(i + 1, epochs, current_loss)

                    except RuntimeError as e:
                        if verbose and i == 0:
                            print(f"  Warning: {str(e)[:50]}")
                        continue

                with torch.no_grad():
                    final_loss = self.negative_mll()

                    if not (torch.isnan(final_loss) or torch.isinf(final_loss)):
                        success_count += 1
                        if final_loss.item() < best_loss:
                            best_loss = final_loss.item()
                            best_params = self._save_params()
                            if verbose:
                                print(f"  ✓ New best loss: {best_loss:.4f}")
                    else:
                        if verbose:
                            print(f"  ✗ Optimization failed (invalid loss)")

            except Exception as e:
                if verbose:
                    print(f"  ✗ Error: {str(e)[:50]}")
                continue

        return {
            'success': success_count > 0,
            'best_loss': best_loss,
            'best_params': best_params,
            'success_count': success_count
        }

    def _multi_restart_adam(self, n_restarts, lr, epochs, verbose, print_every):
        """Multi-start Adam optimization."""
        best_loss = float('inf')
        best_params = None
        success_count = 0

        for restart in range(n_restarts):
            if verbose:
                print(f"\n[Restart {restart + 1}/{n_restarts}]")

            if restart > 0:
                self._random_init()

            try:
                optimizer = torch.optim.Adam(self.params, lr=lr)

                for i in range(epochs):
                    optimizer.zero_grad()
                    loss = self.negative_mll()

                    if torch.isnan(loss) or torch.isinf(loss):
                        if verbose and i == 0:
                            print(f"  Warning: Initial loss invalid")
                        break

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.params, max_norm=1.0)
                    optimizer.step()

                    if verbose and (i + 1) % print_every == 0:
                        self._print_progress(i + 1, epochs, loss)

                with torch.no_grad():
                    final_loss = self.negative_mll()

                    if not (torch.isnan(final_loss) or torch.isinf(final_loss)):
                        success_count += 1
                        if final_loss.item() < best_loss:
                            best_loss = final_loss.item()
                            best_params = self._save_params()
                            if verbose:
                                print(f"  ✓ New best loss: {best_loss:.4f}")
                    else:
                        if verbose:
                            print(f"  ✗ Optimization failed (invalid loss)")

            except Exception as e:
                if verbose:
                    print(f"  ✗ Error: {str(e)[:50]}")
                continue

        return {
            'success': success_count > 0,
            'best_loss': best_loss,
            'best_params': best_params,
            'success_count': success_count
        }

    def _random_init(self):
        """Random initialization of hyperparameters."""
        X_features = self.X[:, :-1]
        X_range = (X_features.max(dim=0).values - X_features.min(dim=0).values)

        if self.N > 1:
            X_sorted, _ = torch.sort(X_features, dim=0)
            avg_spacing = torch.mean(X_sorted[1:] - X_sorted[:-1], dim=0)
            l_min = torch.maximum(0.1 * avg_spacing, 0.01 * X_range).numpy()
            l_max = torch.minimum(2.0 * X_range, torch.ones_like(X_range)).numpy()
        else:
            l_min = (0.01 * X_range).numpy()
            l_max = X_range.numpy()

        self.kernel.log_l.data = torch.tensor(
            np.log(np.random.uniform(l_min, l_max)),
            dtype=torch.float32
        )

        y_var = torch.var(self.y).item()
        if y_var < 1e-6:
            y_var = 1.0

        theta_l_min = 0.1 * y_var
        theta_l_max = 3.0 * y_var
        self.kernel.log_theta_l.data = torch.tensor(
            [np.log(np.random.uniform(theta_l_min, theta_l_max))],
            dtype=torch.float32
        )

        sn2_min = 0.001 * y_var
        sn2_max = 0.5 * y_var
        self.log_sn2_target.data = torch.tensor(
            [np.log(np.random.uniform(sn2_min, sn2_max))],
            dtype=torch.float32
        )
        self.log_sn2_source.data = torch.tensor(
            [np.log(np.random.uniform(sn2_min, sn2_max))],
            dtype=torch.float32
        )

        # Initialize b to encourage positive correlation
        self.kernel.b.data = torch.tensor(
            [np.random.uniform(-5.0, 0)],
            dtype=torch.float32
        )

    def _save_params(self):
        """Save current parameters."""
        return {
            'log_l': self.kernel.log_l.data.clone(),
            'log_theta_l': self.kernel.log_theta_l.data.clone(),
            'b': self.kernel.b.data.clone(),
            'log_sn2_target': self.log_sn2_target.data.clone(),
            'log_sn2_source': self.log_sn2_source.data.clone()
        }

    def _load_params(self, params):
        """Load parameters."""
        self.kernel.log_l.data = params['log_l'].clone()
        self.kernel.log_theta_l.data = params['log_theta_l'].clone()
        self.kernel.b.data = params['b'].clone()
        self.log_sn2_target.data = params['log_sn2_target'].clone()
        self.log_sn2_source.data = params['log_sn2_source'].clone()

    def _print_progress(self, epoch, total_epochs, loss):
        """Print training progress."""
        l_val = torch.exp(self.kernel.log_l).detach().numpy()
        theta_l_val = torch.exp(self.kernel.log_theta_l).item()
        lambda_val = self.kernel.compute_lambda().item()
        sn2_target_val = torch.exp(self.log_sn2_target).item()
        sn2_source_val = torch.exp(self.log_sn2_source).item()

        l_str = np.array2string(l_val, precision=4, separator=', ', suppress_small=True)

        print(
            f"Epoch {epoch}/{total_epochs} - NLL: {loss.item():.4f} | "
            f"l: {l_str}, θ_l: {theta_l_val:.4f}, λ: {lambda_val:.4f}, "
            f"σ²_T: {sn2_target_val:.6f}, σ²_S: {sn2_source_val:.6f}"
        )

    def _print_hyperparameters(self):
        """Print final hyperparameters."""
        l_val = torch.exp(self.kernel.log_l).detach().numpy()
        theta_l_val = torch.exp(self.kernel.log_theta_l).item()
        lambda_val = self.kernel.compute_lambda().item()
        sn2_target_val = torch.exp(self.log_sn2_target).item()
        sn2_source_val = torch.exp(self.log_sn2_source).item()

        l_str = np.array2string(l_val, precision=4, separator=', ', suppress_small=True)

        print(f"Final hyperparameters:")
        print(f"  Lengthscales (per dimension): {l_str}")
        print(f"  Output variance (θ_l): {theta_l_val:.4f}")
        print(f"  Task correlation (λ): {lambda_val:.4f}")
        print(f"  Noise variance (target): {sn2_target_val:.6f}")
        print(f"  Noise variance (source): {sn2_source_val:.6f}")

    def predict(self, X_star, return_target_only=True):
        """
        Compute posterior mean and covariance for target task predictions.

        Follows Equations (12) and (13) in the paper:
        μ_T1 = K_21 (K_11 + σ²_T2 I)^(-1) y_T2
        C_T1 = (K_22 + σ²_T1 I) - K_21 (K_11 + σ²_T2 I)^(-1) K_12

        Parameters
        ----------
        X_star : torch.Tensor, shape (N*, D) or (N*, D+1)
            Test input points. If shape is (N*, D+1), last column is task indicator.
            Otherwise, assumes all test points are for target task (task=0).
        return_target_only : bool, default=True
            If True, only return predictions for target task samples.
            If False, return predictions for all samples.

        Returns
        -------
        mu_star : torch.Tensor, shape (N*,)
            Posterior mean at test points.
        Sigma_star : torch.Tensor, shape (N*, N*)
            Posterior covariance matrix at test points.
        """
        # Prepare test data
        if X_star.shape[1] == self.kernel.input_dim:
            # No task indicator, assume target task
            task_star = torch.zeros(X_star.shape[0], dtype=torch.float32)
            X_star_full = torch.cat([X_star, task_star.unsqueeze(-1)], dim=1)
        else:
            # Has task indicator
            X_star_full = X_star.to(torch.float32)
            task_star = X_star_full[:, -1]

        # Recover noise variances
        sn2_target = torch.exp(self.log_sn2_target)
        sn2_source = torch.exp(self.log_sn2_source)

        # Split training data by task
        task_train = self.task_indicators

        # Build noise matrix for training data
        noise_vector_train = torch.where(task_train == 0, sn2_target, sn2_source)

        # K_11: Training covariance with noise
        K_train = self.kernel.forward(self.X, self.X, task_train, task_train)
        K_train_noise = K_train + torch.diag(noise_vector_train)

        # K_21: Test-Train covariance
        K_star_train = self.kernel.forward(X_star_full, self.X, task_star, task_train)

        # K_22: Test-Test covariance
        K_star_star = self.kernel.forward(X_star_full, X_star_full, task_star, task_star)

        # Build noise matrix for test data (only for target task)
        noise_vector_star = torch.where(task_star == 0, sn2_target, torch.zeros_like(sn2_target))
        K_star_star_noise = K_star_star + torch.diag(noise_vector_star)

        # Compute posterior mean: μ* = K_21 K_11^(-1) y
        K_inv_y = torch.linalg.solve(K_train_noise, self.y.unsqueeze(-1))
        mu_star = torch.matmul(K_star_train, K_inv_y).squeeze()

        # Compute posterior covariance: Σ* = K_22 - K_21 K_11^(-1) K_12
        K_inv_K_star_T = torch.linalg.solve(K_train_noise, K_star_train.T)
        Sigma_star = K_star_star_noise - torch.matmul(K_star_train, K_inv_K_star_T)

        return mu_star, Sigma_star

    def get_task_correlation(self):
        """
        Get the learned task correlation factor λ.

        Returns
        -------
        lambda_val : float
            Task correlation factor in range [0, 1].
            Values close to 1 indicate high task similarity.
            Values close to 0 indicate low task similarity.
        """
        with torch.no_grad():
            lambda_val = self.kernel.compute_lambda().item()
        return lambda_val


# ============================================================
# Example Usage
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("TGP Example: 2D Function with Source Task Transfer")
    print("=" * 70)

    # Define target task function (complex)
    def target_func(X):
        x1 = X[:, 0]
        x2 = X[:, 1]
        return torch.sin(5 * x1) * torch.exp(-x1 ** 2) + torch.cos(3 * x2) * x2


    # Define source task function (similar but shifted)
    def source_func(X):
        x1 = X[:, 0]
        x2 = X[:, 1]
        return torch.sin(5 * x1) * torch.exp(-x1 ** 2) + torch.cos(3 * x2) * x2 + 0.2

    # Generate training data
    # Target task: 10 samples (limited)
    n_target = 10
    X_target = torch.rand(n_target, 2) * 2.0
    y_target = target_func(X_target) + torch.randn(n_target) * 0.1
    task_target = torch.zeros(n_target, 1)

    # Source task: 50 samples (abundant)
    n_source = 50
    X_source = torch.rand(n_source, 2) * 2.0
    y_source = source_func(X_source) + torch.randn(n_source) * 0.1
    task_source = torch.ones(n_source, 1)

    # Combine training data
    X_train = torch.cat([
        torch.cat([X_target, task_target], dim=1),
        torch.cat([X_source, task_source], dim=1)
    ], dim=0)
    y_train = torch.cat([y_target, y_source], dim=0)

    print(f"\nTraining data:")
    print(f"  Target task: {n_target} samples")
    print(f"  Source task: {n_source} samples")
    print(f"  Total: {len(y_train)} samples")

    # Train TGP model
    print(f"\n{'=' * 70}")
    print("Training TGP model...")
    print(f"{'=' * 70}")

    tgp_model = TGPRegression()
    tgp_model.train(X_train, y_train, n_restarts=3, verbose=True)

    # Get learned task correlation
    lambda_val = tgp_model.get_task_correlation()
    print(f"\n{'=' * 70}")
    print(f"Learned task correlation (λ): {lambda_val:.4f}")
    if lambda_val > 0.5:
        print("  → High correlation: Source task is helpful!")
    elif lambda_val > 0.2:
        print("  → Moderate correlation: Some transfer learning benefit")
    else:
        print("  → Low correlation: Limited transfer learning benefit")
    print(f"{'=' * 70}")

    # Generate test data (target task only)
    n_test = 100
    X_test = torch.rand(n_test, 2) * 2.0
    y_test_true = target_func(X_test)

    # Predict
    print(f"\nMaking predictions on {n_test} test points...")
    with torch.no_grad():
        y_test_pred, cov_test = tgp_model.predict(X_test)

    # Evaluate
    mse = torch.mean((y_test_pred - y_test_true) ** 2).item()
    rmse = np.sqrt(mse)

    ss_res = torch.sum((y_test_true - y_test_pred) ** 2).item()
    ss_tot = torch.sum((y_test_true - torch.mean(y_test_true)) ** 2).item()
    r2_score = 1 - (ss_res / ss_tot)

    print(f"\n{'=' * 70}")
    print("Test Results (Target Task Prediction)")
    print(f"{'=' * 70}")
    print(f"MSE:  {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"R²:   {r2_score:.6f}")
    print(f"{'=' * 70}")

    # Compare with standard GP (no transfer learning)
    print(f"\n{'=' * 70}")
    print("Comparison: Training standard GP without source task...")
    print(f"{'=' * 70}")

    from gp import GPRegression, RBFKernel

    gp_model = GPRegression()
    gp_model.train(X_target, y_target, n_restarts=3, verbose=False)

    with torch.no_grad():
        y_test_pred_gp, _ = gp_model.predict(X_test)

    mse_gp = torch.mean((y_test_pred_gp - y_test_true) ** 2).item()
    rmse_gp = np.sqrt(mse_gp)

    ss_res_gp = torch.sum((y_test_true - y_test_pred_gp) ** 2).item()
    ss_tot_gp = torch.sum((y_test_true - torch.mean(y_test_true)) ** 2).item()
    r2_score_gp = 1 - (ss_res_gp / ss_tot_gp)

    print(f"\nStandard GP Results (Target Task Only, No Transfer):")
    print(f"MSE:  {mse_gp:.6f}")
    print(f"RMSE: {rmse_gp:.6f}")
    print(f"R²:   {r2_score_gp:.6f}")

    print(f"\n{'=' * 70}")
    print("Performance Comparison:")
    print(f"{'=' * 70}")
    print(f"TGP RMSE:  {rmse:.6f}")
    print(f"GP RMSE:   {rmse_gp:.6f}")
    improvement = (rmse_gp - rmse) / rmse_gp * 100
    print(f"Improvement: {improvement:.2f}%")
    print(f"{'=' * 70}")