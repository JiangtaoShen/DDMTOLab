import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from ddmtolab.Algorithms.STSO.DE import DE
from ddmtolab.Methods.mtop import MTOP


# ----------------------------------------------------
# 1. Kernel Function - RBFKernel (ARD version with log constraints)
# ----------------------------------------------------
class RBFKernel(torch.nn.Module):
    """
    Radial Basis Function (RBF) kernel with ARD (Automatic Relevance Determination).

    This version uses separate lengthscale parameters for each input dimension,
    allowing the model to automatically determine the relevance of each feature.

    Parameters
    ----------
    input_dim : int
        Number of input dimensions.
    lengthscale : float, array-like, or None, default=None
        Initial lengthscale parameter(s).
        - If None: Initialize all dimensions to 1.0
        - If scalar: Initialize all dimensions to this value
        - If array-like: Must have shape (input_dim,)
    output_variance : float, default=1.0
        Initial output variance parameter.

    Attributes
    ----------
    input_dim : int
        Number of input dimensions.
    log_l : torch.nn.Parameter, shape (D,)
        Log-transformed lengthscale parameters for each dimension.
    log_sf2 : torch.nn.Parameter, shape (1,)
        Log-transformed output variance parameter.
    """

    def __init__(self, input_dim, lengthscale=None, output_variance=1.0):
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
        # Optimize log(l_d) for each dimension d
        self.log_l = torch.nn.Parameter(torch.tensor(np.log(lengthscale), dtype=torch.float32))
        # Optimize log(sigma_f^2)
        self.log_sf2 = torch.nn.Parameter(torch.tensor([np.log(output_variance)], dtype=torch.float32))

    def forward(self, X1, X2):
        """
        Compute the kernel matrix K(X1, X2) with ARD.

        Parameters
        ----------
        X1 : torch.Tensor, shape (N1, D)
            First set of input points.
        X2 : torch.Tensor, shape (N2, D)
            Second set of input points.

        Returns
        -------
        K : torch.Tensor, shape (N1, N2)
            Kernel matrix computed as:
            sigma_f^2 * exp(-0.5 * sum_d[(X1_d - X2_d)^2 / l_d^2])
        """

        # Recover positive parameters
        l = torch.exp(self.log_l)  # shape: (D,)
        sf2 = torch.exp(self.log_sf2)

        # 1. Reshape inputs to calculate squared Euclidean distance for all pairs
        # X1: (N1, D) -> (N1, 1, D)
        X1 = X1.unsqueeze(1)
        # X2: (N2, D) -> (1, N2, D)
        X2 = X2.unsqueeze(0)

        # Calculate difference (N1, N2, D)
        diff = X1 - X2

        # Scale each dimension by its lengthscale (ARD)
        # l broadcasts to (1, 1, D)
        scaled_diff = diff / l

        # Calculate squared distance (N1, N2)
        sq_dist = torch.sum(scaled_diff ** 2, dim=2)

        # 2. Apply RBF formula: K = sigma_f^2 * exp(-0.5 * sum_d[(x1_d - x2_d)^2 / l_d^2])
        K = sf2 * torch.exp(-0.5 * sq_dist)
        return K


# ----------------------------------------------------
# 2. GP Regression Model (ARD version with log constraints and training method)
# ----------------------------------------------------
class GPRegression:
    """
    Gaussian Process Regression model with ARD kernel and automatic hyperparameter optimization.

    Simple usage:
        gp = GPRegression()
        gp.train(X_train, y_train)
        mean, cov = gp.predict(X_test)

    Parameters
    ----------
    kernel : RBFKernel, optional
        Kernel function for computing covariances. If None, will be created in train().
    noise_variance : float, default=0.1
        Initial observation noise variance.

    Attributes
    ----------
    kernel : RBFKernel or None
        The kernel function.
    log_sn2 : torch.nn.Parameter or None
        Log-transformed noise variance parameter (optimizes log(sigma_n^2)).
    params : list or None
        List of all learnable parameters (kernel + noise).
    X : torch.Tensor or None
        Training input data.
    y : torch.Tensor or None
        Training target data.
    N : int or None
        Number of training points.
    initial_noise_variance : float
        Initial noise variance value.
    verbose_default : bool
        Default verbosity setting.
    """

    def __init__(self, kernel=None, noise_variance=0.1):
        self.kernel = kernel
        self.initial_noise_variance = noise_variance

        if kernel is not None:
            self.log_sn2 = torch.nn.Parameter(torch.tensor([np.log(noise_variance)], dtype=torch.float32))
            self.params = list(self.kernel.parameters()) + [self.log_sn2]
        else:
            self.log_sn2 = None
            self.params = None

        # Training data (initialized to None)
        self.X = None
        self.y = None
        self.N = None

        self.verbose_default = True  # Can be set to False to reduce output

    def set_data(self, X, y):
        """
        Load training data.

        Parameters
        ----------
        X : torch.Tensor, shape (N, D)
            Training input features.
        y : torch.Tensor, shape (N,)
            Training target values.
        """
        # Ensure input data is float32 type
        self.X = X.to(torch.float32)
        self.y = y.to(torch.float32)
        self.N = X.shape[0]

    def negative_mll(self):
        """
        Calculate negative marginal log-likelihood (NLL).

        The NLL is computed as:
        NLL = 0.5 * y^T K^-1 y + 0.5 * log|K| + Constant

        Returns
        -------
        nll : torch.Tensor, scalar
            Negative marginal log-likelihood value.
        """
        # Recover positive observation noise variance
        sn2 = torch.exp(self.log_sn2)

        # 1. Calculate K_with_noise = k(X, X) + sigma_n^2 * I
        K = self.kernel.forward(self.X, self.X)
        # Add noise to the diagonal for stability and to model observation noise
        # Ensure K_with_noise is a symmetric positive definite (SPD) matrix
        K_with_noise = K + sn2 * torch.eye(self.N, dtype=torch.float32)

        # 2. Compute Cholesky decomposition L (K = L L^T)
        try:
            L = torch.linalg.cholesky(K_with_noise)
        except RuntimeError as e:
            # Catch Cholesky error, usually occurs early in optimization
            # Return a very large loss value
            if 'cholesky' in str(e):
                return torch.tensor(1e10, dtype=torch.float32)
            raise e

        # 3. Compute log determinant: log|K| = 2 * sum(log(diag(L)))
        log_det_K = 2.0 * torch.sum(torch.log(torch.diag(L)))

        # 4. Compute y^T K^-1 y (Data fit term)
        # Solve L v = y, then L^T alpha = v => K alpha = y
        v = torch.linalg.solve(L, self.y.unsqueeze(-1))
        alpha = torch.linalg.solve(L.T, v)

        # y^T K^-1 y = y^T alpha
        data_fit_term = torch.sum(self.y * alpha.squeeze())

        # 5. Combine NLL terms
        nll = 0.5 * data_fit_term + 0.5 * log_det_K + 0.5 * self.N * torch.log(torch.tensor(2.0 * torch.pi))

        return nll

    def train(self, X=None, y=None, lengthscale=None, output_variance=1.0, noise_variance=None,
              n_restarts=5, verbose=True, print_every=20):
        """
        Train GP model (optimize hyperparameters) with automatic strategy selection.

        Optimization strategy (automatically executed):
        1. First attempt multi-start L-BFGS (lr=0.1, epochs=100)
        2. If failed, gradually reduce learning rate: 0.01 -> 0.001 -> 0.0001
        3. If all L-BFGS attempts fail, switch to multi-start Adam (lr=0.01, epochs=2000)
        4. When Adam fails, also gradually reduce learning rate: 0.001 -> 0.0001
        5. If all optimizations fail, use default hyperparameters

        Parameters
        ----------
        X : torch.Tensor, shape (N, D), optional
            Training input features. If None, must have called set_data() first.
        y : torch.Tensor, shape (N,), optional
            Training target values. If None, must have called set_data() first.
        lengthscale : float, array-like, or None, default=None
            Initial lengthscale parameter(s). If None, use 1.0 for all dimensions.
        output_variance : float, default=1.0
            Initial output variance parameter.
        noise_variance : float, optional
            Initial observation noise variance. If None, use the value from __init__.
        n_restarts : int, default=5
            Number of random restarts for multi-start optimization.
        verbose : bool, default=True
            Whether to print training information.
        print_every : int, default=20
            Print progress every N epochs.
        """
        # Load training data if provided
        if X is not None and y is not None:
            self.set_data(X, y)

        # Check if data has been loaded
        if self.X is None or self.y is None:
            raise ValueError("Please provide training data either via train(X, y, ...) or set_data() first!")

        # Get input dimension
        input_dim = self.X.shape[1]

        # Create kernel if not provided
        if self.kernel is None:
            self.kernel = RBFKernel(
                input_dim=input_dim,
                lengthscale=lengthscale,
                output_variance=output_variance
            )

        # Set noise variance
        if noise_variance is None:
            noise_variance = self.initial_noise_variance

        # Initialize log_sn2 and params if not already done
        if self.log_sn2 is None:
            self.log_sn2 = torch.nn.Parameter(torch.tensor([np.log(noise_variance)], dtype=torch.float32))
            self.params = list(self.kernel.parameters()) + [self.log_sn2]

        if verbose:
            print(f"=" * 70)
            print(f"Starting multi-start adaptive optimization (n_restarts={n_restarts})")
            print(
                f"Total parameters: {self.kernel.input_dim} lengthscales + 1 output_var + 1 noise_var = {self.kernel.input_dim + 2}")
            print(f"=" * 70)

        # Save best results
        best_loss = float('inf')
        best_params = None

        # ========== Stage 1: L-BFGS Optimization ==========
        lbfgs_lrs = [0.1, 0.01, 0.001, 0.0001]  # Learning rate candidates
        lbfgs_epochs = 100
        lbfgs_success = False

        for current_lr in lbfgs_lrs:
            if verbose:
                print(f"\n{'─' * 70}")
                print(f"Attempting L-BFGS optimization (lr: {current_lr}, epochs: {lbfgs_epochs})")
                print(f"{'─' * 70}")

            # Multi-start optimization
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
                    print(f"\n✗ L-BFGS optimization failed (lr: {current_lr}), trying lower learning rate...")

        # ========== Stage 2: Adam Optimization (if L-BFGS failed) ==========
        if not lbfgs_success:
            if verbose:
                print(f"\n{'=' * 70}")
                print(f"All L-BFGS attempts failed, switching to Adam optimizer")
                print(f"{'=' * 70}")

            adam_lrs = [0.01, 0.001, 0.0001]  # Adam learning rate candidates
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
                        print(f"\n✗ Adam optimization failed (lr: {current_lr}), trying lower learning rate...")

            if not adam_success:
                if verbose:
                    print(f"\n✗ Warning: All optimization strategies failed!")

        # ========== Stage 3: Use default parameters (if all optimizations failed) ==========
        if best_params is None:
            if verbose:
                print(f"\n{'=' * 70}")
                print(f"Using default hyperparameters")
                print(f"{'=' * 70}")

            # Set default parameters (adaptive based on data)
            default_params = self._get_default_params()
            self._load_params(default_params)

            # Evaluate loss with default parameters
            with torch.no_grad():
                best_loss = self.negative_mll().item()

            if verbose:
                print(f"Default parameter loss: {best_loss:.4f}")
        else:
            # Load best parameters
            self._load_params(best_params)

        if verbose:
            print(f"\n{'=' * 70}")
            print(f"Optimization complete!")
            if best_loss < float('inf'):
                print(f"Best loss: {best_loss:.4f}")
                self._print_hyperparameters()
            else:
                print(f"Warning: No valid optimization result found")
            print(f"{'=' * 70}")

    def _get_default_params(self):
        """
        Get default hyperparameters (adaptive based on data).

        Returns
        -------
        params : dict
            Dictionary containing default parameters:
            - 'log_l': Log-transformed lengthscales, shape (D,)
            - 'log_sf2': Log-transformed output variance
            - 'log_sn2': Log-transformed noise variance
        """
        # 1. Lengthscale: Based on input data range for each dimension
        X_range = (self.X.max(dim=0).values - self.X.min(dim=0).values)
        # Prevent very small ranges
        X_range = torch.where(X_range < 1e-6, torch.ones_like(X_range), X_range)

        # Use 20% of data range for each dimension as default lengthscale (moderate smoothness)
        default_l = 0.2 * X_range.numpy()

        # 2. Output Variance: Based on target data variance
        y_var = torch.var(self.y).item()
        if y_var < 1e-6:
            y_var = 1.0

        # Use data variance as default output variance
        default_sf2 = y_var

        # 3. Noise Variance: Set to small value, assuming good data quality
        # Use 1% of data variance as default noise
        default_sn2 = 0.01 * y_var

        if self.verbose_default:
            print(f"\nDefault parameter calculation:")
            print(f"  Input ranges per dimension: {X_range.numpy()}")
            print(f"  Target variance: {y_var:.4f}")
            print(f"  Default lengthscales: {default_l}")
            print(f"  Default output_variance: {default_sf2:.4f}")
            print(f"  Default noise_variance: {default_sn2:.4f}")

        return {
            'log_l': torch.tensor(np.log(default_l), dtype=torch.float32),
            'log_sf2': torch.tensor([np.log(default_sf2)], dtype=torch.float32),
            'log_sn2': torch.tensor([np.log(default_sn2)], dtype=torch.float32)
        }

    def _multi_restart_lbfgs(self, n_restarts, lr, epochs, verbose, print_every):
        """
        Multi-start L-BFGS optimization.

        Parameters
        ----------
        n_restarts : int
            Number of random restarts.
        lr : float
            Learning rate.
        epochs : int
            Number of epochs.
        verbose : bool
            Print progress.
        print_every : int
            Print frequency.

        Returns
        -------
        result : dict
            Dictionary containing optimization results:
            - 'success': Whether any restart succeeded
            - 'best_loss': Best loss achieved
            - 'best_params': Best parameters found
            - 'success_count': Number of successful restarts
        """
        best_loss = float('inf')
        best_params = None
        success_count = 0

        for restart in range(n_restarts):
            if verbose:
                print(f"\n[Restart {restart + 1}/{n_restarts}]")

            # Random initialization of hyperparameters (first time uses original initialization)
            if restart > 0:
                self._random_init()

            try:
                # Single L-BFGS optimization
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

                    # Check loss validity
                    if torch.isnan(loss) or torch.isinf(loss):
                        penalty = sum(torch.sum(p ** 2) for p in self.params)
                        return torch.tensor(1e8, dtype=torch.float32) + penalty

                    if loss.requires_grad:
                        loss.backward()

                    return loss

                # Training loop
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
                            print(f"  Warning: Optimization error: {str(e)[:50]}")
                        continue

                # Evaluate final loss
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
                    print(f"  ✗ Optimization error: {str(e)[:50]}")
                continue

        return {
            'success': success_count > 0,
            'best_loss': best_loss,
            'best_params': best_params,
            'success_count': success_count
        }

    def _multi_restart_adam(self, n_restarts, lr, epochs, verbose, print_every):
        """
        Multi-start Adam optimization.

        Parameters
        ----------
        n_restarts : int
            Number of random restarts.
        lr : float
            Learning rate.
        epochs : int
            Number of epochs.
        verbose : bool
            Print progress.
        print_every : int
            Print frequency.

        Returns
        -------
        result : dict
            Dictionary containing optimization results:
            - 'success': Whether any restart succeeded
            - 'best_loss': Best loss achieved
            - 'best_params': Best parameters found
            - 'success_count': Number of successful restarts
        """
        best_loss = float('inf')
        best_params = None
        success_count = 0

        for restart in range(n_restarts):
            if verbose:
                print(f"\n[Restart {restart + 1}/{n_restarts}]")

            # Random initialization of hyperparameters (first time uses original initialization)
            if restart > 0:
                self._random_init()

            try:
                optimizer = torch.optim.Adam(self.params, lr=lr)

                for i in range(epochs):
                    optimizer.zero_grad()
                    loss = self.negative_mll()

                    # Check loss validity
                    if torch.isnan(loss) or torch.isinf(loss):
                        if verbose and i == 0:
                            print(f"  Warning: Initial loss invalid, skipping this restart")
                        break

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.params, max_norm=1.0)
                    optimizer.step()

                    if verbose and (i + 1) % print_every == 0:
                        self._print_progress(i + 1, epochs, loss)

                # Evaluate final loss
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
                    print(f"  ✗ Optimization error: {str(e)[:50]}")
                continue

        return {
            'success': success_count > 0,
            'best_loss': best_loss,
            'best_params': best_params,
            'success_count': success_count
        }

    def _random_init(self):
        """
        Adaptive random initialization of hyperparameters.

        Initializes lengthscale for each dimension based on that dimension's
        data range and density, output variance based on target variance,
        and noise variance as a fraction of target variance.
        """

        # 1. Lengthscale: Based on input data range and density for each dimension
        X_range = (self.X.max(dim=0).values - self.X.min(dim=0).values)

        # Calculate average spacing between adjacent points for each dimension
        if self.N > 1:
            X_sorted, _ = torch.sort(self.X, dim=0)
            avg_spacing = torch.mean(X_sorted[1:] - X_sorted[:-1], dim=0)
            l_min = torch.maximum(0.1 * avg_spacing, 0.01 * X_range).numpy()
            l_max = torch.minimum(2.0 * X_range, torch.ones_like(X_range)).numpy()
        else:
            l_min = (0.01 * X_range).numpy()
            l_max = X_range.numpy()

        # Random initialization for each dimension
        self.kernel.log_l.data = torch.tensor(
            np.log(np.random.uniform(l_min, l_max)),
            dtype=torch.float32
        )

        # 2. Output Variance: Based on target data variance
        y_var = torch.var(self.y).item()
        if y_var < 1e-6:  # Prevent variance from being too small
            y_var = 1.0

        sf2_min = 0.1 * y_var
        sf2_max = 3.0 * y_var

        self.kernel.log_sf2.data = torch.tensor(
            [np.log(np.random.uniform(sf2_min, sf2_max))],
            dtype=torch.float32
        )

        # 3. Noise Variance: Based on target data variance
        sn2_min = 0.001 * y_var
        sn2_max = 0.5 * y_var

        self.log_sn2.data = torch.tensor(
            [np.log(np.random.uniform(sn2_min, sn2_max))],
            dtype=torch.float32
        )

    def _save_params(self):
        """
        Save current parameters.

        Returns
        -------
        params : dict
            Dictionary containing cloned parameter tensors.
        """
        return {
            'log_l': self.kernel.log_l.data.clone(),
            'log_sf2': self.kernel.log_sf2.data.clone(),
            'log_sn2': self.log_sn2.data.clone()
        }

    def _load_params(self, params):
        """
        Load parameters.

        Parameters
        ----------
        params : dict
            Dictionary containing parameter tensors to load.
        """
        self.kernel.log_l.data = params['log_l'].clone()
        self.kernel.log_sf2.data = params['log_sf2'].clone()
        self.log_sn2.data = params['log_sn2'].clone()

    def _print_progress(self, epoch, total_epochs, loss):
        """
        Print training progress.

        Parameters
        ----------
        epoch : int
            Current epoch.
        total_epochs : int
            Total number of epochs.
        loss : torch.Tensor
            Current loss value.
        """
        l_val = torch.exp(self.kernel.log_l).detach().numpy()
        sf2_val = torch.exp(self.kernel.log_sf2).item()
        sn2_val = torch.exp(self.log_sn2).item()

        # Format lengthscales nicely
        l_str = np.array2string(l_val, precision=4, separator=', ', suppress_small=True)

        print(
            f"Epoch {epoch}/{total_epochs} - NLL Loss: {loss.item():.4f} | "
            f"l: {l_str}, sf2: {sf2_val:.4f}, sn2: {sn2_val:.6f}"
        )

    def _print_hyperparameters(self):
        """Print final hyperparameters."""
        l_val = torch.exp(self.kernel.log_l).detach().numpy()
        sf2_val = torch.exp(self.kernel.log_sf2).item()
        sn2_val = torch.exp(self.log_sn2).item()

        # Format lengthscales nicely
        l_str = np.array2string(l_val, precision=4, separator=', ', suppress_small=True)

        print(f"Final hyperparameters:")
        print(f"  Lengthscales (per dimension): {l_str}")
        print(f"  Output variance: {sf2_val:.4f}")
        print(f"  Noise variance: {sn2_val:.6f}")

    def predict(self, X_star):
        """
        Compute posterior mean and covariance based on standard GP formulas.

        The posterior is computed as:
        mu* = K* K^-1 y
        Sigma* = K** - K* K^-1 K*^T

        Parameters
        ----------
        X_star : torch.Tensor, shape (N*, D)
            Test input points.

        Returns
        -------
        mu_star : torch.Tensor, shape (N*,)
            Posterior mean at test points.
        Sigma_star : torch.Tensor, shape (N*, N*)
            Posterior covariance matrix at test points.
        """
        X_star = X_star.to(torch.float32)
        sn2 = torch.exp(self.log_sn2)

        # 1. Training covariance K = k(X, X) + sn2 * I
        K = self.kernel.forward(self.X, self.X) + sn2 * torch.eye(self.N, dtype=torch.float32)

        # 2. Test-Training covariance K_star = k(X*, X)
        K_star = self.kernel.forward(X_star, self.X)

        # 3. Test-Test covariance K_star_star = k(X*, X*)
        K_star_star = self.kernel.forward(X_star, X_star)

        # 4. Solve K^-1 y (used in mean calculation)
        K_inv_y = torch.linalg.solve(K, self.y.unsqueeze(-1))

        # 5. Posterior Mean: mu* = K* K^-1 y
        mu_star = torch.matmul(K_star, K_inv_y).squeeze()

        # 6. Solve K^-1 K*^T (used in covariance calculation)
        K_inv_K_star_T = torch.linalg.solve(K, K_star.T)

        # 7. Posterior Covariance: Sigma* = K** - K* K^-1 K*^T
        Sigma_star = K_star_star - torch.matmul(K_star, K_inv_K_star_T)

        return mu_star, Sigma_star

    def plot_prediction(self, X_test=None, n_points=100, x_range=None, figsize=(5.5, 3.6), true_function=None):
        """
        Visualize GP prediction results (for 1D input).

        Parameters
        ----------
        X_test : torch.Tensor, optional
            Test points (if None, automatically generated).
        n_points : int, default=100
            Number of automatically generated test points.
        x_range : tuple, optional
            Range of test points (x_min, x_max).
        figsize : tuple, default=(5.5, 3.6)
            Figure size.
        true_function : callable, optional
            True function f(x) for plotting the true curve.
        """
        if self.X is None or self.y is None:
            raise ValueError("Please load training data using set_data() method first!")

        if self.X.shape[1] != 1:
            raise ValueError("plot_prediction only works for 1D input. For higher dimensions, use predict() directly.")

        # Generate test points
        if X_test is None:
            if x_range is None:
                x_min = self.X.min().item()
                x_max = self.X.max().item()
                w = x_max - x_min
                x_min -= 0.1 * w
                x_max += 0.1 * w
            else:
                x_min, x_max = x_range

            X_grid = torch.linspace(x_min, x_max, n_points).unsqueeze(-1)
            X_test = torch.cat([X_grid, self.X], dim=0)
            X_test, _ = torch.sort(X_test, dim=0)

        # Predict
        with torch.no_grad():
            mean, covariance = self.predict(X_test)
            std_dev = torch.sqrt(torch.diag(covariance))

        # Plot
        plt.figure(figsize=figsize, dpi=300)

        # True function (if provided)
        if true_function is not None:
            y_true = true_function(X_test.squeeze())
            plt.plot(X_test.squeeze().numpy(), y_true.numpy(), 'r--', linewidth=2, label='True Function', alpha=0.8)

        # Training data
        plt.plot(self.X.squeeze().numpy(), self.y.numpy(), 'kx', markersize=10, markeredgewidth=2,
                 label='Training Data')

        # Prediction mean
        plt.plot(X_test.squeeze().numpy(), mean.numpy(), 'b-', linewidth=2, label='Prediction Mean')

        # 95% confidence interval
        plt.fill_between(
            X_test.squeeze().numpy(),
            mean.numpy() - 2 * std_dev.numpy(),
            mean.numpy() + 2 * std_dev.numpy(),
            color='b', alpha=0.3, label='95% Confidence Interval'
        )

        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()


class ExpectedImprovement:
    """
    Expected Improvement (EI) acquisition function for Bayesian Optimization.

    The EI function balances exploration and exploitation by measuring the
    expected amount of improvement over the current best observation.

    Parameters
    ----------
    gp_model : GPRegression
        Trained Gaussian Process regression model.
    xi : float, optional (default=0.01)
        Exploration-exploitation trade-off parameter. Larger values encourage
        more exploration.

    Attributes
    ----------
    f_best : float or None
        Current best observed function value.

    References
    ----------
    Mockus, J., Tiesis, V., & Zilinskas, A. (1978). The application of
    Bayesian methods for seeking the extremum. Towards Global Optimization, 2, 117-129.
    """

    def __init__(self, gp_model, xi=0.01):
        self.gp_model = gp_model
        self.xi = xi
        self.f_best = None

    def compute(self, X, y_best=None):
        """
        Compute Expected Improvement values for candidate points.

        The EI is calculated as:
            EI(x) = (mu(x) - f_best - xi) * Phi(Z) + sigma(x) * phi(Z)
        where:
            Z = (mu(x) - f_best - xi) / sigma(x)
            Phi is the standard normal CDF
            phi is the standard normal PDF

        Parameters
        ----------
        X : torch.Tensor, shape (n_points, n_dims)
            Candidate points at which to evaluate EI.
        y_best : float or None, optional (default=None)
            Current best observed value. If None, uses the maximum value
            from the training data.

        Returns
        -------
        ei_values : torch.Tensor, shape (n_points,)
            Expected Improvement values for each candidate point.
        """
        # Get predictive mean and covariance matrix
        with torch.no_grad():
            mu, cov = self.gp_model.predict(X)
            # Extract standard deviation from diagonal of covariance matrix
            sigma = torch.sqrt(torch.diag(cov))

        # If best value not specified, use maximum from training data
        if y_best is None:
            if self.f_best is None:
                # Extract best value from training targets
                y_train = self.gp_model.y
                self.f_best = torch.min(y_train).item()  # min for minimization
            y_best = self.f_best
        else:
            self.f_best = y_best

        # Convert to numpy for computation
        mu_np = mu.cpu().numpy()
        sigma_np = sigma.cpu().numpy()

        # Compute improvement (for minimization, want mu < y_best)
        improvement = y_best - mu_np - self.xi

        # Avoid division by zero
        sigma_np = np.maximum(sigma_np, 1e-9)

        # Standardized improvement
        Z = improvement / sigma_np

        # Calculate EI using standard normal CDF and PDF
        ei = improvement * norm.cdf(Z) + sigma_np * norm.pdf(Z)

        # Set negative values to zero
        ei = np.maximum(ei, 0.0)

        return torch.tensor(ei, dtype=torch.float32)


def bo_next_point_ei(decs, objs, dim, gp_model, xi=0.01, data_type=torch.float):
    """
    Generate next sampling point using Bayesian Optimization with Expected Improvement.

    Parameters
    ----------
    decs : np.ndarray
        Decision variables (training data) of shape (n_samples, dim)
    objs : np.ndarray
        Objective values (training data) of shape (n_samples, 1) or (n_samples,)
    dim : int
        Dimension of the problem
    gp_model : GPRegression
        Your custom Gaussian Process regression model
    xi : float, optional
        Exploration-exploitation trade-off parameter (default: 0.01)
    data_type : torch.dtype, optional
        Data type for torch tensors (default: torch.float)

    Returns
    -------
    candidate_np : np.ndarray
        Next sampling point of shape (1, dim)
    """
    # Prepare training data for Gaussian Process
    train_X = torch.tensor(decs, dtype=data_type)
    train_Y = torch.tensor(objs.flatten(), dtype=data_type)

    # Train Gaussian Process model
    gp_model.train(train_X, train_Y, n_restarts=3, verbose=False)

    # Create Expected Improvement acquisition function
    ei = ExpectedImprovement(gp_model, xi=xi)

    # Get current best value (minimum for minimization)
    best_f = train_Y.min().item()

    # Wrap EI as numpy function for DE optimizer
    def ei_func(x):
        """
        Negative EI function for minimization.

        Parameters
        ----------
        x : np.ndarray
            Candidate point(s) of shape (n_points, dim)

        Returns
        -------
        neg_ei : np.ndarray
            Negative EI value(s), shape (n_points, 1)
        """
        # Ensure x is 2D
        if x.ndim == 1:
            x = x.reshape(1, -1)

        x_torch = torch.tensor(x, dtype=data_type)
        with torch.no_grad():
            ei_value = ei.compute(x_torch, y_best=best_f)

        ei_np = ei_value.detach().cpu().numpy()

        # Reshape to (n_points, 1) for MTOP compatibility
        if ei_np.ndim == 1:
            ei_np = ei_np.reshape(-1, 1)

        # Return negative EI for minimization (we want to maximize EI)
        return ei_np

    # Optimize EI using Differential Evolution
    problem = MTOP()
    problem.add_task(ei_func, dim=dim)
    de = DE(problem, n=50, max_nfes=1000, F=0.5, CR=0.9, save_data=False, disable_tqdm=True)
    result = de.optimize()

    return result.best_decs


# ============================================================
# Example Usage
# ============================================================
if __name__ == "__main__":
    print("Example 1: 1D Function (Simplified API)")

    # 1. Generate synthetic data
    X = torch.linspace(0, 1, 7).unsqueeze(-1)


    def true_func(x):
        return torch.sin(7 * x - 4) * x ** 2


    y = true_func(X.squeeze())

    # 2. Create and train model with ONE line!
    gp_model = GPRegression()
    gp_model.train(X, y, n_restarts=3, verbose=False)

    # 3. Visualize prediction results
    gp_model.plot_prediction(x_range=(0, 1), n_points=200, true_function=true_func)

    print("Example 2: 3D Function (Simplified API)")

    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)


    # Define 3D function
    def true_func_3d(X):
        x1 = X[:, 0]
        x2 = X[:, 1]
        x3 = X[:, 2]
        return (torch.sin(5 * x1) * torch.exp(-x1 ** 2) +
                torch.cos(2 * x2) * x2 +
                0.5 * x3 ** 2)


    # Generate training data
    n_train = 50
    X_train = torch.rand(n_train, 3) * 2.0
    y_train = true_func_3d(X_train) + torch.randn(n_train) * 0.1

    # Train model with ONE line!
    gp_model_3d = GPRegression()
    gp_model_3d.train(X_train, y_train, n_restarts=3, verbose=False)

    # Test
    n_test = 100
    X_test = torch.rand(n_test, 3) * 2.0
    y_test_true = true_func_3d(X_test)

    with torch.no_grad():
        y_test_pred, _ = gp_model_3d.predict(X_test)

    mse = torch.mean((y_test_pred - y_test_true) ** 2).item()
    rmse = np.sqrt(mse)

    ss_res = torch.sum((y_test_true - y_test_pred) ** 2).item()
    ss_tot = torch.sum((y_test_true - torch.mean(y_test_true)) ** 2).item()
    r2_score = 1 - (ss_res / ss_tot)

    print("\n" + "=" * 70)
    print("Test Results")
    print("=" * 70)
    print(f"MSE: {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"R² Score: {r2_score:.6f}")
    print("=" * 70)