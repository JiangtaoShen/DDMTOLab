import torch
import torch.nn as nn
from tqdm import tqdm
import time
from Problems.RWO.PINN_HPO.utility import get_data_2d, plot_func_2d
from Problems.RWO.PINN_HPO.NNs import PINNs


def wave_2d(num_layers, num_nodes, activation_func, epochs, grid_size, learning_rate, test_name, alpha=4, beta=3,
            plotshow=False, show_information=True, device='cuda:0'):

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

        # Print final results
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

wave_2d(num_layers=3, num_nodes=50, activation_func='tanh', epochs=20000, grid_size=50, learning_rate=0.001, test_name='aa', alpha=4, beta=4, plotshow=True, show_information=True, device='cuda:0')