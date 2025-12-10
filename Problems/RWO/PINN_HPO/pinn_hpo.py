import torch
import torch.nn as nn
from tqdm import tqdm
import time
from Problems.RWO.PINN_HPO.pinnhpo_utils import get_data_2d, plot_func_2d
from Problems.BasicFunctions.basic_functions import *
from Methods.mtop import MTOP
import warnings
warnings.filterwarnings('ignore', category=UserWarning)


class Sin(nn.Module):
    """
    Custom Sin activation function.

    Implements $f(x) = \sin(x)$ as a PyTorch module.
    """

    def forward(self, x):
        return torch.sin(x)


class Swish(nn.Module):
    """
    Custom Swish activation function.

    Implements $f(x) = x \cdot \sigma(x)$ where $\sigma$ is the sigmoid function.
    """

    def forward(self, x):
        return x * torch.sigmoid(x)


class PINNs(nn.Module):
    """
    Physics-Informed Neural Network (PINN).

    A fully-connected neural network designed for solving partial differential
    equations (PDEs) using physics-informed constraints.

    Parameters
    ----------
    in_dim : int
        Input dimension (number of spatial/temporal coordinates).
    hidden_dim : int
        Number of nodes in each hidden layer.
    out_dim : int
        Output dimension (number of solution components).
    num_layer : int
        Total number of layers including output layer.
        For example, num_layer=3 means 2 hidden layers + 1 output layer.
    activation : str or float, optional
        Activation function type (default is 'tanh').
        - String options: 'tanh', 'relu', 'sigmoid', 'sin', 'swish'
        - Numerical mapping: [0, 1) -> tanh, [1, 2) -> relu, [2, 3) -> sigmoid,
          [3, 4) -> sin, [4, 5] -> swish

    Attributes
    ----------
    network : nn.Sequential
        The sequential neural network model.
    """

    def __init__(self, in_dim, hidden_dim, out_dim, num_layer, activation='tanh'):
        super(PINNs, self).__init__()

        # Activation function mapping for string input
        activation_map = {
            'tanh': nn.Tanh,
            'relu': nn.ReLU,
            'sigmoid': nn.Sigmoid,
            'sin': Sin,
            'swish': Swish
        }

        # Activation function mapping for numerical input
        activation_num_map = {
            0: nn.Tanh,  # [0, 1) -> tanh
            1: nn.ReLU,  # [1, 2) -> relu
            2: nn.Sigmoid,  # [2, 3) -> sigmoid
            3: Sin,  # [3, 4) -> sin
            4: Swish  # [4, 5] -> swish
        }

        # Determine activation function class
        if isinstance(activation, (int, float)):
            # Numerical input: validate range first
            if activation < 0 or activation > 5:
                raise ValueError(f"Numerical activation {activation} out of range. "
                                 f"Expected [0, 5], maps to: 0=tanh, 1=relu, 2=sigmoid, 3=sin, 4=swish")

            # Convert to integer index, handle special case for 5
            activation_idx = int(activation) if activation < 5 else 4
            activation_class = activation_num_map[activation_idx]
        elif isinstance(activation, str):
            # String input
            if activation.lower() not in activation_map:
                raise ValueError(f"Activation '{activation}' not supported. "
                                 f"Available: {list(activation_map.keys())}")
            activation_class = activation_map[activation.lower()]
        else:
            raise TypeError(f"Activation must be string or number, got {type(activation)}")

        # Build network layers
        layers = []
        for i in range(num_layer - 1):
            # First layer: in_dim -> hidden_dim; Other layers: hidden_dim -> hidden_dim
            in_features = in_dim if i == 0 else hidden_dim
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(activation_class())

        # Output layer (no activation)
        layers.append(nn.Linear(hidden_dim, out_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def convection_2d(num_layers, num_nodes, activation_func, epochs, grid_size, learning_rate, test_name, beta=30,
                  plotshow=False, show_information=True, device='cuda:0'):
    """
    Solves the 2D **Convection equation** using PINN.

    The PDE is: $\frac{\partial u}{\partial t} + \beta \frac{\partial u}{\partial x} = 0$.
    Analytical solution: $u(x, t) = \sin(x - \beta t)$.

    Parameters
    ----------
    num_layers : int
        Number of layers in the neural network.
    num_nodes : int
        Number of nodes in each hidden layer.
    activation_func : str or float
        Activation function for the network.
    epochs : int
        Number of training epochs.
    grid_size : int
        Grid resolution for training points (grid_size x grid_size).
    learning_rate : float
        Learning rate for the Adam optimizer.
    test_name : str
        Name identifier for the test run.
    beta : float, optional
        Convection velocity parameter (default is 30).
    plotshow : bool, optional
        Whether to plot the results (default is False).
    show_information : bool, optional
        Whether to display training progress (default is True).
    device : str, optional
        Device for computation, e.g., 'cuda:0' (default is 'cuda:0').

    Returns
    -------
    dict
        Dictionary containing training results including final errors, losses, and runtime.
    """

    # PDEs
    def pde(x, y):
        x.requires_grad_(True)
        dy = torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]
        dy_t = dy[:, 1:2]
        dy_x = dy[:, 0:1]
        return dy_t + beta * dy_x

    # Analytical solution
    def func(x):
        return torch.sin(x[:, 0:1] - beta * x[:, 1:2])

    # Loss function
    def losses(model, points, b1, b2, b3, b4, points_test, lam_pde=1.0, lam_bc=1.0, lam_ic=1.0):
        points = points.clone().requires_grad_(True)
        pred_points = model(points)
        pde_residual = pde(points, pred_points)
        pde_loss = torch.mean(pde_residual ** 2)

        pred_b1 = model(b1)
        pred_b2 = model(b2)
        bc_loss = torch.mean((pred_b1 - pred_b2) ** 2)

        pred_b3 = model(b3)
        true_b3 = func(b3)
        ic_loss = torch.mean((pred_b3 - true_b3) ** 2)

        pred_test = model(points_test)
        true_test = func(points_test)
        l1_ab_metric = torch.mean(torch.abs(pred_test - true_test))
        l1_re_metric = torch.mean(torch.abs(pred_test - true_test)) / torch.mean(torch.abs(true_test))
        l2_ab_metric = torch.mean((pred_test - true_test) ** 2)
        l2_re_metric = torch.sqrt(torch.mean((pred_test - true_test) ** 2)) / torch.sqrt(torch.mean(true_test ** 2))

        total_loss = lam_pde * pde_loss + lam_bc * bc_loss + lam_ic * ic_loss

        return total_loss, {
            'pde_loss': pde_loss.item(),
            'bc_loss': bc_loss.item(),
            'ic_loss': ic_loss.item(),
            'total_loss': total_loss.item(),
            'l1_ab_metric': l1_ab_metric.item(),
            'l1_re_metric': l1_re_metric.item(),
            'l2_ab_metric': l2_ab_metric.item(),
            'l2_re_metric': l2_re_metric.item()
        }

    # Initialization
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    try:
        start_time = time.time()

        # Generate training and test points
        num_x = grid_size
        num_y = grid_size
        num_x_test = 100
        num_y_test = 100
        range_x = torch.tensor([[0., 2 * torch.pi], [0., 1.]]).to(device)

        points, b1, b2, b3, b4 = get_data_2d(num_x, num_y, range_x, device)
        points_test, _, _, _, _ = get_data_2d(num_x_test, num_y_test, range_x, device)

        # Create model with specified parameters
        model = PINNs(in_dim=2, hidden_dim=num_nodes, out_dim=1, num_layer=num_layers, activation=activation_func).to(
            device)
        model.apply(init_weights)

        # Setup optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Training loop
        train_losses_history = []

        # Use tqdm only if show_information is True
        if show_information:
            epoch_iterator = tqdm(range(epochs), desc=f"Training {test_name}")
        else:
            epoch_iterator = range(epochs)

        for epoch in epoch_iterator:
            def closure():
                total_loss, train_loss_dict = losses(model, points, b1, b2, b3, b4, points_test, lam_pde=1.0,
                                                     lam_bc=1.0, lam_ic=1.0)
                optimizer.zero_grad()
                total_loss.backward()
                if not hasattr(closure, 'latest_loss_dict'):
                    closure.latest_loss_dict = train_loss_dict
                else:
                    closure.latest_loss_dict.update(train_loss_dict)
                return total_loss

            optimizer.step(closure)

            if hasattr(closure, 'latest_loss_dict'):
                train_losses_history.append(closure.latest_loss_dict.copy())

        end_time = time.time()
        runtime = end_time - start_time

        if plotshow == True:
            plot_func_2d(points_test, model, func, range_x, save_path=None, test_name=test_name)

        # Get final metrics
        final_l1_ab_error = train_losses_history[-1]['l1_ab_metric']
        final_l1_re_error = train_losses_history[-1]['l1_re_metric']
        final_l2_ab_error = train_losses_history[-1]['l2_ab_metric']
        final_l2_re_error = train_losses_history[-1]['l2_re_metric']
        final_pde_loss = train_losses_history[-1]['pde_loss']
        final_bc_loss = train_losses_history[-1]['bc_loss']
        final_ic_loss = train_losses_history[-1]['ic_loss']
        final_total_loss = train_losses_history[-1]['total_loss']

        return {
            'test_name': test_name,
            'final_l1_ab_error': final_l1_ab_error,
            'final_l1_re_error': final_l1_re_error,
            'final_l2_ab_error': final_l2_ab_error,
            'final_l2_re_error': final_l2_re_error,
            'final_pde_loss': final_pde_loss,
            'final_bc_loss': final_bc_loss,
            'final_ic_loss': final_ic_loss,
            'final_total_loss': final_total_loss,
            'runtime': runtime,
            'num_layers': num_layers,
            'num_nodes': num_nodes,
            'activation_func': activation_func,
            'epochs': epochs,
            'grid_size': grid_size,
            'learning_rate': learning_rate
        }

    except Exception as e:
        if show_information:
            print("\n" + "=" * 80)
            print(f"ERROR in {test_name}: {str(e)}")
            print("=" * 80 + "\n")
            import traceback
            traceback.print_exc()

        return {
            'test_name': test_name,
            'final_l2_error': float('inf'),
            'runtime': 0,
            'error': str(e)
        }


def reaction_2d(num_layers, num_nodes, activation_func, epochs, grid_size, learning_rate, test_name, rho=4,
                plotshow=False, show_information=True, device='cuda:0'):
    """
    Solves the 2D **Reaction equation** using PINN.

    The PDE is: $\frac{\partial u}{\partial t} = \rho u (1 - u)$.
    Analytical solution involves a Gaussian initial condition evolving over time.

    Parameters
    ----------
    num_layers : int
        Number of layers in the neural network.
    num_nodes : int
        Number of nodes in each hidden layer.
    activation_func : str or float
        Activation function for the network.
    epochs : int
        Number of training epochs.
    grid_size : int
        Grid resolution for training points (grid_size x grid_size).
    learning_rate : float
        Learning rate for the Adam optimizer.
    test_name : str
        Name identifier for the test run.
    rho : float, optional
        Reaction rate parameter (default is 4).
    plotshow : bool, optional
        Whether to plot the results (default is False).
    show_information : bool, optional
        Whether to display training progress (default is True).
    device : str, optional
        Device for computation, e.g., 'cuda:0' (default is 'cuda:0').

    Returns
    -------
    dict
        Dictionary containing training results including final errors, losses, and runtime.
    """

    # PDEs
    def pde(x, y):
        x.requires_grad_(True)
        dy = torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]
        dy_t = dy[:, 1:2]
        return dy_t - rho * y * (1 - y)

    # Analytical solution
    def func(x):
        h = torch.exp(-(x[:, 0:1] - torch.pi) ** 2 / (2 * (torch.pi / 4) ** 2))
        return h * torch.exp(rho * x[:, 1:2]) / (h * torch.exp(rho * x[:, 1:2]) + 1 - h)

    # Loss function
    def losses(model, points, b1, b2, b3, b4, points_test, lam_pde=1.0, lam_bc=1.0, lam_ic=1.0):
        points = points.clone().requires_grad_(True)
        pred_points = model(points)
        pde_residual = pde(points, pred_points)
        pde_loss = torch.mean(pde_residual ** 2)

        pred_b1 = model(b1)
        pred_b2 = model(b2)
        bc_loss = torch.mean((pred_b1 - pred_b2) ** 2)

        pred_b3 = model(b3)
        true_b3 = torch.exp(-(b3[:, 0:1] - torch.pi) ** 2 / (2 * (torch.pi / 4) ** 2))
        ic_loss = torch.mean((pred_b3 - true_b3) ** 2)

        pred_test = model(points_test)
        true_test = func(points_test)
        l1_ab_metric = torch.mean(torch.abs(pred_test - true_test))
        l1_re_metric = torch.mean(torch.abs(pred_test - true_test)) / torch.mean(torch.abs(true_test))
        l2_ab_metric = torch.mean((pred_test - true_test) ** 2)
        l2_re_metric = torch.sqrt(torch.mean((pred_test - true_test) ** 2)) / torch.sqrt(torch.mean(true_test ** 2))

        total_loss = lam_pde * pde_loss + lam_bc * bc_loss + lam_ic * ic_loss

        return total_loss, {
            'pde_loss': pde_loss.item(),
            'bc_loss': bc_loss.item(),
            'ic_loss': ic_loss.item(),
            'total_loss': total_loss.item(),
            'l1_ab_metric': l1_ab_metric.item(),
            'l1_re_metric': l1_re_metric.item(),
            'l2_ab_metric': l2_ab_metric.item(),
            'l2_re_metric': l2_re_metric.item()
        }

    # Initialization
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    try:
        start_time = time.time()

        # Generate training and test points
        num_x = grid_size
        num_y = grid_size
        num_x_test = 100
        num_y_test = 100
        range_x = torch.tensor([[0., 2 * torch.pi], [0., 1.]]).to(device)

        points, b1, b2, b3, b4 = get_data_2d(num_x, num_y, range_x, device)
        points_test, _, _, _, _ = get_data_2d(num_x_test, num_y_test, range_x, device)

        # Create model with specified parameters
        model = PINNs(in_dim=2, hidden_dim=num_nodes, out_dim=1, num_layer=num_layers, activation=activation_func).to(
            device)
        model.apply(init_weights)

        # Setup optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Training loop
        train_losses_history = []

        # Use tqdm only if show_information is True
        if show_information:
            epoch_iterator = tqdm(range(epochs), desc=f"Training {test_name}")
        else:
            epoch_iterator = range(epochs)

        for epoch in epoch_iterator:
            def closure():
                total_loss, train_loss_dict = losses(model, points, b1, b2, b3, b4, points_test, lam_pde=1.0,
                                                     lam_bc=1.0, lam_ic=1.0)
                optimizer.zero_grad()
                total_loss.backward()
                if not hasattr(closure, 'latest_loss_dict'):
                    closure.latest_loss_dict = train_loss_dict
                else:
                    closure.latest_loss_dict.update(train_loss_dict)
                return total_loss

            optimizer.step(closure)

            if hasattr(closure, 'latest_loss_dict'):
                train_losses_history.append(closure.latest_loss_dict.copy())

        end_time = time.time()
        runtime = end_time - start_time

        if plotshow == True:
            plot_func_2d(points_test, model, func, range_x, save_path=None, test_name=test_name)

        # Get final metrics
        final_l1_ab_error = train_losses_history[-1]['l1_ab_metric']
        final_l1_re_error = train_losses_history[-1]['l1_re_metric']
        final_l2_ab_error = train_losses_history[-1]['l2_ab_metric']
        final_l2_re_error = train_losses_history[-1]['l2_re_metric']
        final_pde_loss = train_losses_history[-1]['pde_loss']
        final_bc_loss = train_losses_history[-1]['bc_loss']
        final_ic_loss = train_losses_history[-1]['ic_loss']
        final_total_loss = train_losses_history[-1]['total_loss']

        return {
            'test_name': test_name,
            'final_l1_ab_error': final_l1_ab_error,
            'final_l1_re_error': final_l1_re_error,
            'final_l2_ab_error': final_l2_ab_error,
            'final_l2_re_error': final_l2_re_error,
            'final_pde_loss': final_pde_loss,
            'final_bc_loss': final_bc_loss,
            'final_ic_loss': final_ic_loss,
            'final_total_loss': final_total_loss,
            'runtime': runtime,
            'num_layers': num_layers,
            'num_nodes': num_nodes,
            'activation_func': activation_func,
            'epochs': epochs,
            'grid_size': grid_size,
            'learning_rate': learning_rate
        }

    except Exception as e:
        if show_information:
            print("\n" + "=" * 80)
            print(f"ERROR in {test_name}: {str(e)}")
            print("=" * 80 + "\n")
            import traceback
            traceback.print_exc()

        return {
            'test_name': test_name,
            'final_l2_error': float('inf'),
            'runtime': 0,
            'error': str(e)
        }


def wave_2d(num_layers, num_nodes, activation_func, epochs, grid_size, learning_rate, test_name, alpha=4, beta=3,
            plotshow=False, show_information=True, device='cuda:0'):
    """
    Solves the 2D **Wave equation** using PINN.

    The PDE is: $\frac{\partial^2 u}{\partial t^2} = \alpha \frac{\partial^2 u}{\partial x^2}$.
    Analytical solution: $u(x, t) = \sin(\pi x) \cos(\sqrt{\alpha} \pi t) + 0.5 \sin(\beta \pi x) \cos(\sqrt{\alpha} \beta \pi t)$.

    Parameters
    ----------
    num_layers : int
        Number of layers in the neural network.
    num_nodes : int
        Number of nodes in each hidden layer.
    activation_func : str or float
        Activation function for the network.
    epochs : int
        Number of training epochs.
    grid_size : int
        Grid resolution for training points (grid_size x grid_size).
    learning_rate : float
        Learning rate for the Adam optimizer.
    test_name : str
        Name identifier for the test run.
    alpha : float, optional
        Wave speed squared parameter (default is 4).
    beta : float, optional
        Frequency multiplier for the second mode (default is 3).
    plotshow : bool, optional
        Whether to plot the results (default is False).
    show_information : bool, optional
        Whether to display training progress (default is True).
    device : str, optional
        Device for computation, e.g., 'cuda:0' (default is 'cuda:0').

    Returns
    -------
    dict
        Dictionary containing training results including final errors, losses, and runtime.
    """

    # PDEs
    def pde(x, u):
        x.requires_grad_(True)
        du = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
        du_t = du[:, 1:2]
        du_x = du[:, 0:1]
        du_tt = torch.autograd.grad(du_t, x, torch.ones_like(du_t), create_graph=True)[0][:, 1:2]
        du_xx = torch.autograd.grad(du_x, x, torch.ones_like(du_x), create_graph=True)[0][:, 0:1]
        return du_tt - alpha * du_xx

    def ic(x, u):
        x.requires_grad_(True)
        du = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
        ic1 = du[:, 1:2]
        ic2 = torch.sin(torch.pi * x[:, 0:1]) + 0.5 * torch.sin(beta * torch.pi * x[:, 0:1]) - u
        return ic1, ic2

    # Analytical solution
    def func(x):
        x_coord = x[:, 0:1]
        t_coord = x[:, 1:2]
        pi = torch.tensor(torch.pi)
        sqrt_alpha = torch.sqrt(torch.tensor(alpha))
        term1 = torch.sin(pi * x_coord) * torch.cos(sqrt_alpha * pi * t_coord)
        term2 = 0.5 * torch.sin(beta * pi * x_coord) * torch.cos(sqrt_alpha * beta * pi * t_coord)
        return term1 + term2

    # Loss function
    def losses(model, points, b1, b2, b3, b4, points_test, lam_pde=1.0, lam_bc=1.0, lam_ic=1.0):
        points = points.clone().requires_grad_(True)
        pred_points = model(points)
        pde_residual = pde(points, pred_points)
        pde_loss = torch.mean(pde_residual ** 2)

        pred_b1 = model(b1)
        pred_b2 = model(b2)
        bc_loss = torch.mean((pred_b1) ** 2) + torch.mean((pred_b2) ** 2)

        b3 = b3.clone().requires_grad_(True)
        pred_b3 = model(b3)
        ic_res_1, ic_res_2 = ic(b3, pred_b3)
        ic_loss = torch.mean((ic_res_1) ** 2) + torch.mean((ic_res_2) ** 2)

        pred_test = model(points_test)
        true_test = func(points_test)
        l1_ab_metric = torch.mean(torch.abs(pred_test - true_test))
        l1_re_metric = torch.mean(torch.abs(pred_test - true_test)) / torch.mean(torch.abs(true_test))
        l2_ab_metric = torch.mean((pred_test - true_test) ** 2)
        l2_re_metric = torch.sqrt(torch.mean((pred_test - true_test) ** 2)) / torch.sqrt(torch.mean(true_test ** 2))

        total_loss = lam_pde * pde_loss + lam_bc * bc_loss + lam_ic * ic_loss

        return total_loss, {
            'pde_loss': pde_loss.item(),
            'bc_loss': bc_loss.item(),
            'ic_loss': ic_loss.item(),
            'total_loss': total_loss.item(),
            'l1_ab_metric': l1_ab_metric.item(),
            'l1_re_metric': l1_re_metric.item(),
            'l2_ab_metric': l2_ab_metric.item(),
            'l2_re_metric': l2_re_metric.item()
        }

    # Initialization
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    try:
        start_time = time.time()

        # Generate training and test points
        num_x = grid_size
        num_y = grid_size
        num_x_test = 100
        num_y_test = 100
        range_x = torch.tensor([[0., 1.], [0., 1.]]).to(device)

        points, b1, b2, b3, b4 = get_data_2d(num_x, num_y, range_x, device)
        points_test, _, _, _, _ = get_data_2d(num_x_test, num_y_test, range_x, device)

        # Create model with specified parameters
        model = PINNs(in_dim=2, hidden_dim=num_nodes, out_dim=1, num_layer=num_layers, activation=activation_func).to(
            device)
        model.apply(init_weights)

        # Setup optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Training loop
        train_losses_history = []

        # Use tqdm only if show_information is True
        if show_information:
            epoch_iterator = tqdm(range(epochs), desc=f"Training {test_name}")
        else:
            epoch_iterator = range(epochs)

        for epoch in epoch_iterator:
            def closure():
                total_loss, train_loss_dict = losses(model, points, b1, b2, b3, b4, points_test, lam_pde=1.0,
                                                     lam_bc=1.0, lam_ic=1.0)
                optimizer.zero_grad()
                total_loss.backward()
                if not hasattr(closure, 'latest_loss_dict'):
                    closure.latest_loss_dict = train_loss_dict
                else:
                    closure.latest_loss_dict.update(train_loss_dict)
                return total_loss

            optimizer.step(closure)

            if hasattr(closure, 'latest_loss_dict'):
                train_losses_history.append(closure.latest_loss_dict.copy())

        end_time = time.time()
        runtime = end_time - start_time

        if plotshow == True:
            plot_func_2d(points_test, model, func, range_x, save_path=None, test_name=test_name)

        # Get final metrics
        final_l1_ab_error = train_losses_history[-1]['l1_ab_metric']
        final_l1_re_error = train_losses_history[-1]['l1_re_metric']
        final_l2_ab_error = train_losses_history[-1]['l2_ab_metric']
        final_l2_re_error = train_losses_history[-1]['l2_re_metric']
        final_pde_loss = train_losses_history[-1]['pde_loss']
        final_bc_loss = train_losses_history[-1]['bc_loss']
        final_ic_loss = train_losses_history[-1]['ic_loss']
        final_total_loss = train_losses_history[-1]['total_loss']

        return {
            'test_name': test_name,
            'final_l1_ab_error': final_l1_ab_error,
            'final_l1_re_error': final_l1_re_error,
            'final_l2_ab_error': final_l2_ab_error,
            'final_l2_re_error': final_l2_re_error,
            'final_pde_loss': final_pde_loss,
            'final_bc_loss': final_bc_loss,
            'final_ic_loss': final_ic_loss,
            'final_total_loss': final_total_loss,
            'runtime': runtime,
            'num_layers': num_layers,
            'num_nodes': num_nodes,
            'activation_func': activation_func,
            'epochs': epochs,
            'grid_size': grid_size,
            'learning_rate': learning_rate
        }

    except Exception as e:
        if show_information:
            print("\n" + "=" * 80)
            print(f"ERROR in {test_name}: {str(e)}")
            print("=" * 80 + "\n")
            import traceback
            traceback.print_exc()

        return {
            'test_name': test_name,
            'final_l2_error': float('inf'),
            'runtime': 0,
            'error': str(e)
        }


def helmholtz_2d(num_layers, num_nodes, activation_func, epochs, grid_size, learning_rate, test_name, n=2,
                 plotshow=False, show_information=True, device='cuda:0'):
    """
    Solves the 2D **Helmholtz equation** using PINN.

    The PDE is: $-\nabla^2 u - k^2 u = f(x, y)$ where $k = n\pi$ and $f(x, y) = k^2 \sin(kx) \sin(ky)$.
    Analytical solution: $u(x, y) = \sin(kx) \sin(ky)$.

    Parameters
    ----------
    num_layers : int
        Number of layers in the neural network.
    num_nodes : int
        Number of nodes in each hidden layer.
    activation_func : str or float
        Activation function for the network.
    epochs : int
        Number of training epochs.
    grid_size : int
        Grid resolution for training points (grid_size x grid_size).
    learning_rate : float
        Learning rate for the Adam optimizer.
    test_name : str
        Name identifier for the test run.
    n : int, optional
        Wave number multiplier (default is 2). Controls frequency via $k = n\pi$.
    plotshow : bool, optional
        Whether to plot the results (default is False).
    show_information : bool, optional
        Whether to display training progress (default is True).
    device : str, optional
        Device for computation, e.g., 'cuda:0' (default is 'cuda:0').

    Returns
    -------
    dict
        Dictionary containing training results including final errors, losses, and runtime.
    """

    k = n * torch.pi

    # PDEs
    def pde(x, y):
        x.requires_grad_(True)
        dy = torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]
        dy_x = dy[:, 0:1]
        dy_y = dy[:, 1:2]
        dy_xx = torch.autograd.grad(dy_x, x, torch.ones_like(dy_x), create_graph=True)[0][:, 0:1]
        dy_yy = torch.autograd.grad(dy_y, x, torch.ones_like(dy_y), create_graph=True)[0][:, 1:2]
        f = k ** 2 * torch.sin(k * x[:, 0:1]) * torch.sin(k * x[:, 1:2])
        return -dy_xx - dy_yy - k ** 2 * y - f

    # Analytical solution
    def func(x):
        return torch.sin(k * x[:, 0:1]) * torch.sin(k * x[:, 1:2])

    # Loss function
    def losses(model, points, b1, b2, b3, b4, points_test, lam_pde=1.0, lam_bc=1.0, lam_ic=1.0):
        points = points.clone().requires_grad_(True)
        pred_points = model(points)
        pde_residual = pde(points, pred_points)
        pde_loss = torch.mean(pde_residual ** 2)

        pred_b1 = model(b1)
        pred_b2 = model(b2)
        pred_b3 = model(b3)
        pred_b4 = model(b4)
        bc_loss = torch.mean((pred_b1) ** 2) + torch.mean((pred_b2) ** 2) + torch.mean((pred_b3) ** 2) + torch.mean(
            (pred_b4) ** 2)

        ic_loss = torch.tensor(0.0, device=device)

        pred_test = model(points_test)
        true_test = func(points_test)
        l1_ab_metric = torch.mean(torch.abs(pred_test - true_test))
        l1_re_metric = torch.mean(torch.abs(pred_test - true_test)) / torch.mean(torch.abs(true_test))
        l2_ab_metric = torch.mean((pred_test - true_test) ** 2)
        l2_re_metric = torch.sqrt(torch.mean((pred_test - true_test) ** 2)) / torch.sqrt(torch.mean(true_test ** 2))

        total_loss = lam_pde * pde_loss + lam_bc * bc_loss + lam_ic * ic_loss

        return total_loss, {
            'pde_loss': pde_loss.item(),
            'bc_loss': bc_loss.item(),
            'ic_loss': ic_loss.item(),
            'total_loss': total_loss.item(),
            'l1_ab_metric': l1_ab_metric.item(),
            'l1_re_metric': l1_re_metric.item(),
            'l2_ab_metric': l2_ab_metric.item(),
            'l2_re_metric': l2_re_metric.item()
        }

    # Initialization
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    try:
        start_time = time.time()

        # Generate training and test points
        num_x = grid_size
        num_y = grid_size
        num_x_test = 100
        num_y_test = 100
        range_x = torch.tensor([[0., 1.], [0., 1.]]).to(device)

        points, b1, b2, b3, b4 = get_data_2d(num_x, num_y, range_x, device)
        points_test, _, _, _, _ = get_data_2d(num_x_test, num_y_test, range_x, device)

        # Create model with specified parameters
        model = PINNs(in_dim=2, hidden_dim=num_nodes, out_dim=1, num_layer=num_layers, activation=activation_func).to(
            device)
        model.apply(init_weights)

        # Setup optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Training loop
        train_losses_history = []

        # Use tqdm only if show_information is True
        if show_information:
            epoch_iterator = tqdm(range(epochs), desc=f"Training {test_name}")
        else:
            epoch_iterator = range(epochs)

        for epoch in epoch_iterator:
            def closure():
                total_loss, train_loss_dict = losses(model, points, b1, b2, b3, b4, points_test, lam_pde=1.0,
                                                     lam_bc=1.0, lam_ic=1.0)
                optimizer.zero_grad()
                total_loss.backward()
                if not hasattr(closure, 'latest_loss_dict'):
                    closure.latest_loss_dict = train_loss_dict
                else:
                    closure.latest_loss_dict.update(train_loss_dict)
                return total_loss

            optimizer.step(closure)

            if hasattr(closure, 'latest_loss_dict'):
                train_losses_history.append(closure.latest_loss_dict.copy())

        end_time = time.time()
        runtime = end_time - start_time

        if plotshow == True:
            plot_func_2d(points_test, model, func, range_x, save_path=None, test_name=test_name)

        # Get final metrics
        final_l1_ab_error = train_losses_history[-1]['l1_ab_metric']
        final_l1_re_error = train_losses_history[-1]['l1_re_metric']
        final_l2_ab_error = train_losses_history[-1]['l2_ab_metric']
        final_l2_re_error = train_losses_history[-1]['l2_re_metric']
        final_pde_loss = train_losses_history[-1]['pde_loss']
        final_bc_loss = train_losses_history[-1]['bc_loss']
        final_ic_loss = train_losses_history[-1]['ic_loss']
        final_total_loss = train_losses_history[-1]['total_loss']

        return {
            'test_name': test_name,
            'final_l1_ab_error': final_l1_ab_error,
            'final_l1_re_error': final_l1_re_error,
            'final_l2_ab_error': final_l2_ab_error,
            'final_l2_re_error': final_l2_re_error,
            'final_pde_loss': final_pde_loss,
            'final_bc_loss': final_bc_loss,
            'final_ic_loss': final_ic_loss,
            'final_total_loss': final_total_loss,
            'runtime': runtime,
            'num_layers': num_layers,
            'num_nodes': num_nodes,
            'activation_func': activation_func,
            'epochs': epochs,
            'grid_size': grid_size,
            'learning_rate': learning_rate
        }

    except Exception as e:
        if show_information:
            print("\n" + "=" * 80)
            print(f"ERROR in {test_name}: {str(e)}")
            print("=" * 80 + "\n")
            import traceback
            traceback.print_exc()

        return {
            'test_name': test_name,
            'final_l2_error': float('inf'),
            'runtime': 0,
            'error': str(e)
        }


class PINN_HPO:
    """
    Physics-Informed Neural Network Hyperparameter Optimization (PINN-HPO) benchmark suite.

    This class provides a collection of multi-task optimization problems for tuning
    PINN hyperparameters across different PDEs (Convection, Reaction, Wave, Helmholtz).
    Each problem consists of multiple related tasks with varying PDE parameters.

    Notes
    -----
    Decision variables for all problems:
    - x[0]: Number of layers (integer, [2, 10])
    - x[1]: Number of nodes per layer (integer, [5, 100])
    - x[2]: Activation function (float, [0, 5] mapping to tanh/relu/sigmoid/sin/swish)
    - x[3]: Training epochs (integer, [5000, 100000])
    - x[4]: Grid size (integer, [10, 200])
    - x[5]: Learning rate (float, [1e-5, 0.1])
    """

    @staticmethod
    def _evaluate_convection(x, beta, test_name, device):
        """
        Evaluates PINN hyperparameters on the Convection equation task.

        Parameters
        ----------
        x : np.ndarray
            Hyperparameter configuration array.
        beta : float
            Convection velocity parameter.
        test_name : str
            Task identifier.
        device : str
            CUDA device for computation.

        Returns
        -------
        np.ndarray
            L2 relative error for each sample.
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)

        n_samples = x.shape[0]
        results = []

        for i in range(n_samples):
            sample = x[i]
            n_layers = int(round(sample[0]))
            n_nodes = int(round(sample[1]))
            activation = float(sample[2])
            epochs = int(round(sample[3]))
            grid_size = int(round(sample[4]))
            learning_rate = float(sample[5])

            result = convection_2d(
                num_layers=n_layers,
                num_nodes=n_nodes,
                activation_func=activation,
                epochs=epochs,
                grid_size=grid_size,
                learning_rate=learning_rate,
                test_name=test_name,
                beta=beta,
                show_information=False,
                device=device
            )

            error = result.get('final_l2_re_error', float('inf'))
            results.append(error)

        return np.array(results).reshape(-1, 1)

    @staticmethod
    def _evaluate_reaction(x, rho, test_name, device):
        """
        Evaluates PINN hyperparameters on the Reaction equation task.

        Parameters
        ----------
        x : np.ndarray
            Hyperparameter configuration array.
        rho : float
            Reaction rate parameter.
        test_name : str
            Task identifier.
        device : str
            CUDA device for computation.

        Returns
        -------
        np.ndarray
            L2 relative error for each sample.
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)

        n_samples = x.shape[0]
        results = []

        for i in range(n_samples):
            sample = x[i]
            n_layers = int(round(sample[0]))
            n_nodes = int(round(sample[1]))
            activation = float(sample[2])
            epochs = int(round(sample[3]))
            grid_size = int(round(sample[4]))
            learning_rate = float(sample[5])

            result = reaction_2d(
                num_layers=n_layers,
                num_nodes=n_nodes,
                activation_func=activation,
                epochs=epochs,
                grid_size=grid_size,
                learning_rate=learning_rate,
                test_name=test_name,
                rho=rho,
                show_information=False,
                device=device
            )

            error = result.get('final_l2_re_error', float('inf'))
            results.append(error)

        return np.array(results).reshape(-1, 1)

    @staticmethod
    def _evaluate_wave(x, alpha, beta, test_name, device):
        """
        Evaluates PINN hyperparameters on the Wave equation task.

        Parameters
        ----------
        x : np.ndarray
            Hyperparameter configuration array.
        alpha : float
            Wave speed squared parameter.
        beta : float
            Frequency multiplier for the second mode.
        test_name : str
            Task identifier.
        device : str
            CUDA device for computation.

        Returns
        -------
        np.ndarray
            L2 relative error for each sample.
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)

        n_samples = x.shape[0]
        results = []

        for i in range(n_samples):
            sample = x[i]
            n_layers = int(round(sample[0]))
            n_nodes = int(round(sample[1]))
            activation = float(sample[2])
            epochs = int(round(sample[3]))
            grid_size = int(round(sample[4]))
            learning_rate = float(sample[5])

            result = wave_2d(
                num_layers=n_layers,
                num_nodes=n_nodes,
                activation_func=activation,
                epochs=epochs,
                grid_size=grid_size,
                learning_rate=learning_rate,
                test_name=test_name,
                alpha=alpha,
                beta=beta,
                show_information=False,
                device=device
            )

            error = result.get('final_l2_re_error', float('inf'))
            results.append(error)

        return np.array(results).reshape(-1, 1)

    @staticmethod
    def _evaluate_helmholtz(x, n, test_name, device):
        """
        Evaluates PINN hyperparameters on the Helmholtz equation task.

        Parameters
        ----------
        x : np.ndarray
            Hyperparameter configuration array.
        n : int
            Wave number multiplier (controls frequency).
        test_name : str
            Task identifier.
        device : str
            CUDA device for computation.

        Returns
        -------
        np.ndarray
            L2 relative error for each sample.
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)

        n_samples = x.shape[0]
        results = []

        for i in range(n_samples):
            sample = x[i]
            n_layers = int(round(sample[0]))
            n_nodes = int(round(sample[1]))
            activation = float(sample[2])
            epochs = int(round(sample[3]))
            grid_size = int(round(sample[4]))
            learning_rate = float(sample[5])

            result = helmholtz_2d(
                num_layers=n_layers,
                num_nodes=n_nodes,
                activation_func=activation,
                epochs=epochs,
                grid_size=grid_size,
                learning_rate=learning_rate,
                test_name=test_name,
                n=n,
                show_information=False,
                device=device
            )

            error = result.get('final_l2_re_error', float('inf'))
            results.append(error)

        return np.array(results).reshape(-1, 1)

    # Problem 1: Convection (β=20, β=30)
    @staticmethod
    def T1_convection_beta20(x):
        return PINN_HPO._evaluate_convection(x, beta=20, test_name='T1_convection_beta20', device='cuda:0')

    @staticmethod
    def T1_convection_beta30(x):
        return PINN_HPO._evaluate_convection(x, beta=30, test_name='T1_convection_beta30', device='cuda:1')

    # Problem 2: Reaction (ρ=4, ρ=5)
    @staticmethod
    def T2_reaction_rho4(x):
        return PINN_HPO._evaluate_reaction(x, rho=4, test_name='T2_reaction_rho4', device='cuda:0')

    @staticmethod
    def T2_reaction_rho5(x):
        return PINN_HPO._evaluate_reaction(x, rho=5, test_name='T2_reaction_rho5', device='cuda:1')

    # Problem 3: Wave (α=3,β=3; α=4,β=3)
    @staticmethod
    def T3_wave_alpha3_beta3(x):
        return PINN_HPO._evaluate_wave(x, alpha=3, beta=3, test_name='T3_wave_alpha3_beta3', device='cuda:0')

    @staticmethod
    def T3_wave_alpha4_beta3(x):
        return PINN_HPO._evaluate_wave(x, alpha=4, beta=3, test_name='T3_wave_alpha4_beta3', device='cuda:1')

    # Problem 4: Helmholtz (n=3, n=4)
    @staticmethod
    def T4_helmholtz_n3(x):
        return PINN_HPO._evaluate_helmholtz(x, n=3, test_name='T4_helmholtz_n3', device='cuda:0')

    @staticmethod
    def T4_helmholtz_n4(x):
        return PINN_HPO._evaluate_helmholtz(x, n=4, test_name='T4_helmholtz_n4', device='cuda:1')

    # Problem 5: Convection (β=20, β=30, β=40)
    @staticmethod
    def T5_convection_beta20(x):
        return PINN_HPO._evaluate_convection(x, beta=20, test_name='T5_convection_beta20', device='cuda:0')

    @staticmethod
    def T5_convection_beta30(x):
        return PINN_HPO._evaluate_convection(x, beta=30, test_name='T5_convection_beta30', device='cuda:1')

    @staticmethod
    def T5_convection_beta40(x):
        return PINN_HPO._evaluate_convection(x, beta=40, test_name='T5_convection_beta40', device='cuda:2')

    # Problem 6: Reaction (ρ=4, ρ=5, ρ=6)
    @staticmethod
    def T6_reaction_rho4(x):
        return PINN_HPO._evaluate_reaction(x, rho=4, test_name='T6_reaction_rho4', device='cuda:0')

    @staticmethod
    def T6_reaction_rho5(x):
        return PINN_HPO._evaluate_reaction(x, rho=5, test_name='T6_reaction_rho5', device='cuda:1')

    @staticmethod
    def T6_reaction_rho6(x):
        return PINN_HPO._evaluate_reaction(x, rho=6, test_name='T6_reaction_rho6', device='cuda:2')

    # Problem 7: Wave (α=3,β=3; α=4,β=3; α=4,β=4)
    @staticmethod
    def T7_wave_alpha3_beta3(x):
        return PINN_HPO._evaluate_wave(x, alpha=3, beta=3, test_name='T7_wave_alpha3_beta3', device='cuda:0')

    @staticmethod
    def T7_wave_alpha4_beta3(x):
        return PINN_HPO._evaluate_wave(x, alpha=4, beta=3, test_name='T7_wave_alpha4_beta3', device='cuda:1')

    @staticmethod
    def T7_wave_alpha4_beta4(x):
        return PINN_HPO._evaluate_wave(x, alpha=4, beta=4, test_name='T7_wave_alpha4_beta4', device='cuda:2')

    # Problem 8: Helmholtz (n=3, n=4, n=5)
    @staticmethod
    def T8_helmholtz_n3(x):
        return PINN_HPO._evaluate_helmholtz(x, n=3, test_name='T8_helmholtz_n3', device='cuda:0')

    @staticmethod
    def T8_helmholtz_n4(x):
        return PINN_HPO._evaluate_helmholtz(x, n=4, test_name='T8_helmholtz_n4', device='cuda:1')

    @staticmethod
    def T8_helmholtz_n5(x):
        return PINN_HPO._evaluate_helmholtz(x, n=5, test_name='T8_helmholtz_n5', device='cuda:2')

    # Problem 9: Mixed (Convection β=30, Reaction ρ=5)
    @staticmethod
    def T9_convection_beta30(x):
        return PINN_HPO._evaluate_convection(x, beta=30, test_name='T9_convection_beta30', device='cuda:0')

    @staticmethod
    def T9_reaction_rho5(x):
        return PINN_HPO._evaluate_reaction(x, rho=5, test_name='T9_reaction_rho5', device='cuda:1')

    # Problem 10: Mixed (Wave α=4,β=3; Helmholtz n=4)
    @staticmethod
    def T10_wave_alpha4_beta3(x):
        return PINN_HPO._evaluate_wave(x, alpha=4, beta=3, test_name='T10_wave_alpha4_beta3', device='cuda:0')

    @staticmethod
    def T10_helmholtz_n4(x):
        return PINN_HPO._evaluate_helmholtz(x, n=4, test_name='T10_helmholtz_n4', device='cuda:1')

    # Problem 11: Mixed (Convection β=30, Reaction ρ=5, Wave α=4,β=3)
    @staticmethod
    def T11_convection_beta30(x):
        return PINN_HPO._evaluate_convection(x, beta=30, test_name='T11_convection_beta30', device='cuda:0')

    @staticmethod
    def T11_reaction_rho5(x):
        return PINN_HPO._evaluate_reaction(x, rho=5, test_name='T11_reaction_rho5', device='cuda:1')

    @staticmethod
    def T11_wave_alpha4_beta3(x):
        return PINN_HPO._evaluate_wave(x, alpha=4, beta=3, test_name='T11_wave_alpha4_beta3', device='cuda:2')

    # Problem 12: Mixed (Convection β=30, Reaction ρ=5, Wave α=4,β=3, Helmholtz n=4)
    @staticmethod
    def T12_convection_beta30(x):
        return PINN_HPO._evaluate_convection(x, beta=30, test_name='T12_convection_beta30', device='cuda:0')

    @staticmethod
    def T12_reaction_rho5(x):
        return PINN_HPO._evaluate_reaction(x, rho=5, test_name='T12_reaction_rho5', device='cuda:1')

    @staticmethod
    def T12_wave_alpha4_beta3(x):
        return PINN_HPO._evaluate_wave(x, alpha=4, beta=3, test_name='T12_wave_alpha4_beta3', device='cuda:2')

    @staticmethod
    def T12_helmholtz_n4(x):
        return PINN_HPO._evaluate_helmholtz(x, n=4, test_name='T12_helmholtz_n4', device='cuda:3')

    def P1(self):
        """
        Generates Problem 1: **Convection** (:math:`\\beta=20`, :math:`\\beta=30`).

        Two-task hyperparameter optimization for Convection PDE with different
        convection velocities.

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance.
        """
        problem = MTOP()
        lower_bound = np.array([2, 5, 0, 5000, 10, 1e-5])
        upper_bound = np.array([10, 100, 5, 100000, 200, 0.1])

        problem.add_task(PINN_HPO.T1_convection_beta20, dim=6, lower_bound=lower_bound, upper_bound=upper_bound)
        problem.add_task(PINN_HPO.T1_convection_beta30, dim=6, lower_bound=lower_bound, upper_bound=upper_bound)

        return problem

    def P2(self):
        """
        Generates Problem 2: **Reaction** (:math:`\\rho=4`, :math:`\\rho=5`).

        Two-task hyperparameter optimization for Reaction PDE with different
        reaction rates.

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance.
        """
        problem = MTOP()
        lower_bound = np.array([2, 5, 0, 5000, 10, 1e-5])
        upper_bound = np.array([10, 100, 5, 100000, 200, 0.1])

        problem.add_task(PINN_HPO.T2_reaction_rho4, dim=6, lower_bound=lower_bound, upper_bound=upper_bound)
        problem.add_task(PINN_HPO.T2_reaction_rho5, dim=6, lower_bound=lower_bound, upper_bound=upper_bound)

        return problem

    def P3(self):
        """
        Generates Problem 3: **Wave** (:math:`\\alpha=3, \\beta=3`; :math:`\\alpha=4, \\beta=3`).

        Two-task hyperparameter optimization for Wave PDE with different
        wave speed parameters.

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance.
        """
        problem = MTOP()
        lower_bound = np.array([2, 5, 0, 5000, 10, 1e-5])
        upper_bound = np.array([10, 100, 5, 100000, 200, 0.1])

        problem.add_task(PINN_HPO.T3_wave_alpha3_beta3, dim=6, lower_bound=lower_bound, upper_bound=upper_bound)
        problem.add_task(PINN_HPO.T3_wave_alpha4_beta3, dim=6, lower_bound=lower_bound, upper_bound=upper_bound)

        return problem

    def P4(self):
        """
        Generates Problem 4: **Helmholtz** (:math:`n=3`, :math:`n=4`).

        Two-task hyperparameter optimization for Helmholtz PDE with different
        wave number multipliers.

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance.
        """
        problem = MTOP()
        lower_bound = np.array([2, 5, 0, 5000, 10, 1e-5])
        upper_bound = np.array([10, 100, 5, 100000, 200, 0.1])

        problem.add_task(PINN_HPO.T4_helmholtz_n3, dim=6, lower_bound=lower_bound, upper_bound=upper_bound)
        problem.add_task(PINN_HPO.T4_helmholtz_n4, dim=6, lower_bound=lower_bound, upper_bound=upper_bound)

        return problem

    def P5(self):
        """
        Generates Problem 5: **Convection** (:math:`\\beta=20`, :math:`\\beta=30`, :math:`\\beta=40`).

        Three-task hyperparameter optimization for Convection PDE with different
        convection velocities.

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance.
        """
        problem = MTOP()
        lower_bound = np.array([2, 5, 0, 5000, 10, 1e-5])
        upper_bound = np.array([10, 100, 5, 100000, 200, 0.1])

        problem.add_task(PINN_HPO.T5_convection_beta20, dim=6, lower_bound=lower_bound, upper_bound=upper_bound)
        problem.add_task(PINN_HPO.T5_convection_beta30, dim=6, lower_bound=lower_bound, upper_bound=upper_bound)
        problem.add_task(PINN_HPO.T5_convection_beta40, dim=6, lower_bound=lower_bound, upper_bound=upper_bound)

        return problem

    def P6(self):
        """
        Generates Problem 6: **Reaction** (:math:`\\rho=4`, :math:`\\rho=5`, :math:`\\rho=6`).

        Three-task hyperparameter optimization for Reaction PDE with different
        reaction rates.

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance.
        """
        problem = MTOP()
        lower_bound = np.array([2, 5, 0, 5000, 10, 1e-5])
        upper_bound = np.array([10, 100, 5, 100000, 200, 0.1])

        problem.add_task(PINN_HPO.T6_reaction_rho4, dim=6, lower_bound=lower_bound, upper_bound=upper_bound)
        problem.add_task(PINN_HPO.T6_reaction_rho5, dim=6, lower_bound=lower_bound, upper_bound=upper_bound)
        problem.add_task(PINN_HPO.T6_reaction_rho6, dim=6, lower_bound=lower_bound, upper_bound=upper_bound)

        return problem

    def P7(self):
        """
        Generates Problem 7: **Wave** (:math:`\\alpha=3, \\beta=3`; :math:`\\alpha=4, \\beta=3`; :math:`\\alpha=4, \\beta=4`).

        Three-task hyperparameter optimization for Wave PDE with different
        wave speed and frequency parameters.

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance.
        """
        problem = MTOP()
        lower_bound = np.array([2, 5, 0, 5000, 10, 1e-5])
        upper_bound = np.array([10, 100, 5, 100000, 200, 0.1])

        problem.add_task(PINN_HPO.T7_wave_alpha3_beta3, dim=6, lower_bound=lower_bound, upper_bound=upper_bound)
        problem.add_task(PINN_HPO.T7_wave_alpha4_beta3, dim=6, lower_bound=lower_bound, upper_bound=upper_bound)
        problem.add_task(PINN_HPO.T7_wave_alpha4_beta4, dim=6, lower_bound=lower_bound, upper_bound=upper_bound)

        return problem

    def P8(self):
        """
        Generates Problem 8: **Helmholtz** (:math:`n=3`, :math:`n=4`, :math:`n=5`).

        Three-task hyperparameter optimization for Helmholtz PDE with different
        wave number multipliers.

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance.
        """
        problem = MTOP()
        lower_bound = np.array([2, 5, 0, 5000, 10, 1e-5])
        upper_bound = np.array([10, 100, 5, 100000, 200, 0.1])

        problem.add_task(PINN_HPO.T8_helmholtz_n3, dim=6, lower_bound=lower_bound, upper_bound=upper_bound)
        problem.add_task(PINN_HPO.T8_helmholtz_n4, dim=6, lower_bound=lower_bound, upper_bound=upper_bound)
        problem.add_task(PINN_HPO.T8_helmholtz_n5, dim=6, lower_bound=lower_bound, upper_bound=upper_bound)

        return problem

    def P9(self):
        """
        Generates Problem 9: **Mixed** (Convection :math:`\\beta=30`, Reaction :math:`\\rho=5`).

        Two-task mixed hyperparameter optimization combining Convection and
        Reaction PDEs.

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance.
        """
        problem = MTOP()
        lower_bound = np.array([2, 5, 0, 5000, 10, 1e-5])
        upper_bound = np.array([10, 100, 5, 100000, 200, 0.1])

        problem.add_task(PINN_HPO.T9_convection_beta30, dim=6, lower_bound=lower_bound, upper_bound=upper_bound)
        problem.add_task(PINN_HPO.T9_reaction_rho5, dim=6, lower_bound=lower_bound, upper_bound=upper_bound)

        return problem

    def P10(self):
        """
        Generates Problem 10: **Mixed** (Wave :math:`\\alpha=4, \\beta=3`; Helmholtz :math:`n=4`).

        Two-task mixed hyperparameter optimization combining Wave and
        Helmholtz PDEs.

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance.
        """
        problem = MTOP()
        lower_bound = np.array([2, 5, 0, 5000, 10, 1e-5])
        upper_bound = np.array([10, 100, 5, 100000, 200, 0.1])

        problem.add_task(PINN_HPO.T10_wave_alpha4_beta3, dim=6, lower_bound=lower_bound, upper_bound=upper_bound)
        problem.add_task(PINN_HPO.T10_helmholtz_n4, dim=6, lower_bound=lower_bound, upper_bound=upper_bound)

        return problem

    def P11(self):
        """
        Generates Problem 11: **Mixed** (Convection :math:`\\beta=30`, Reaction :math:`\\rho=5`, Wave :math:`\\alpha=4, \\beta=3`).

        Three-task mixed hyperparameter optimization combining Convection,
        Reaction, and Wave PDEs.

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance.
        """
        problem = MTOP()
        lower_bound = np.array([2, 5, 0, 5000, 10, 1e-5])
        upper_bound = np.array([10, 100, 5, 100000, 200, 0.1])

        problem.add_task(PINN_HPO.T11_convection_beta30, dim=6, lower_bound=lower_bound, upper_bound=upper_bound)
        problem.add_task(PINN_HPO.T11_reaction_rho5, dim=6, lower_bound=lower_bound, upper_bound=upper_bound)
        problem.add_task(PINN_HPO.T11_wave_alpha4_beta3, dim=6, lower_bound=lower_bound, upper_bound=upper_bound)

        return problem

    def P12(self):
        """
        Generates Problem 12: **Mixed** (Convection :math:`\\beta=30`, Reaction :math:`\\rho=5`, Wave :math:`\\alpha=4, \\beta=3`, Helmholtz :math:`n=4`).

        Four-task mixed hyperparameter optimization combining all PDE types:
        Convection, Reaction, Wave, and Helmholtz.

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance.
        """
        problem = MTOP()
        lower_bound = np.array([2, 5, 0, 5000, 10, 1e-5])
        upper_bound = np.array([10, 100, 5, 100000, 200, 0.1])

        problem.add_task(PINN_HPO.T12_convection_beta30, dim=6, lower_bound=lower_bound, upper_bound=upper_bound)
        problem.add_task(PINN_HPO.T12_reaction_rho5, dim=6, lower_bound=lower_bound, upper_bound=upper_bound)
        problem.add_task(PINN_HPO.T12_wave_alpha4_beta3, dim=6, lower_bound=lower_bound, upper_bound=upper_bound)
        problem.add_task(PINN_HPO.T12_helmholtz_n4, dim=6, lower_bound=lower_bound, upper_bound=upper_bound)

        return problem