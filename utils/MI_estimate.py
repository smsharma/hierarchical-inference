import torch, functools
import numpy as np
import models.utils as utils
from models.deep_set import DeepSet

def MINE_estimator(X_in, Y_in, max_number_epochs = 10000, lr = 1e-3, hidden_dim = 60, layers = 4,
                   number_burnin_steps = 20, losstype = "biased", batch_size = 1024,
                   patience = 10, rel_tol = 1e-2, eps = 1e-6):

    def _evaluate_T(model, x, y):
        idx_shuffled = torch.randperm(torch.Tensor.size(x, dim = 0))
        y_shuffled = y[idx_shuffled]
        
        samples_xy = torch.cat([x, y], axis = 1)
        samples_x_y = torch.cat([x, y_shuffled], axis = 1)
        
        T_xy = torch.squeeze(model(samples_xy))
        T_x_y = torch.squeeze(model(samples_x_y))

        return T_xy, T_x_y
    
    def _loss_biased(model, x, y):
        T_xy, T_x_y = _evaluate_T(model, x, y)
        loss = -(torch.mean(T_xy, dim = 0) - torch.log(eps + torch.mean(torch.exp(T_x_y), dim = 0)))        
        return loss

    class log_mean_exp(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, exp_t_mov_avg):
            ctx.save_for_backward(x, exp_t_mov_avg)
            return torch.log(eps + torch.mean(torch.exp(x), dim = 0))

        @staticmethod
        def backward(ctx, grad_output):
            x, exp_t_mov_avg = ctx.saved_tensors
            grad = grad_output / x.shape[0] * torch.exp(x).detach() / (eps + exp_t_mov_avg)
            return grad, None
    
    def _loss_unbiased(model, x, y, exp_T_x_y_mov_avg, alpha = 0.01):

        def _update_exp_mov_avg(val, alpha, old):
            return alpha * val + (1 - alpha) * old
        
        T_xy, T_x_y = _evaluate_T(model, x, y)
        exp_T_x_y = torch.exp(T_x_y).mean()

        if exp_T_x_y_mov_avg is None:
            exp_T_x_y_mov_avg = exp_T_x_y.detach()
        else:
            exp_T_x_y_mov_avg = _update_exp_mov_avg(exp_T_x_y, alpha, exp_T_x_y_mov_avg.item())

        loss = -(torch.mean(T_xy, dim = 0) - log_mean_exp.apply(T_x_y, exp_T_x_y_mov_avg))
        return loss, exp_T_x_y_mov_avg
    
    def _loss_fdiv(model, x, y):
        T_xy, T_x_y = _evaluate_T(model, x, y)
        loss = -(torch.mean(T_xy, dim = 0) - torch.mean(torch.exp(T_x_y - 1), dim = 0))
        return loss    
    
    x_tensor, y_tensor = torch.tensor(X_in, dtype = torch.float), torch.tensor(Y_in, dtype = torch.float)
    len_x, len_y = torch.Tensor.size(x_tensor, dim = 0), torch.Tensor.size(y_tensor, dim = 0)
    assert len_x == len_y
    
    dim_x, dim_y = torch.Tensor.size(x_tensor, dim = 1), torch.Tensor.size(y_tensor, dim = 1)

    model = utils.build_mlp(input_dim = dim_x + dim_y, hidden_dim = hidden_dim, output_dim = 1, layers = layers)
    opt = torch.optim.Adam(model.parameters(), lr = lr)

    # lr_expr = lambda epoch: 0.999
    # scheduler = torch.optim.lr_scheduler.MultiplicativeLR(opt, lr_lambda = lr_expr)

    exp_T_x_y_mov_avg = None

    global_MI_estimate = None
    steps_without_improvement = None
    for cur_epoch in range(max_number_epochs):

        perm = torch.randperm(len_x)

        for cur_batch_start in range(0, len_x, batch_size):
            model.zero_grad()
            
            inds = perm[cur_batch_start:cur_batch_start + batch_size]
            batch_x, batch_y = x_tensor[inds], y_tensor[inds]                          
        
            if losstype == "biased":
                loss = _loss_biased(model, x_tensor, y_tensor)
            elif losstype == "unbiased":
                loss, exp_T_x_y_mov_avg = _loss_unbiased(model, x_tensor, y_tensor, exp_T_x_y_mov_avg)
            elif losstype == "fdiv":
                loss = _loss_fdiv(model, x_tensor, y_tensor)
            else:
                raise RuntimeError(f"Error: loss type '{losstype}' not implemented")
        
            loss.backward()
            opt.step()
            # scheduler.step()

            print(loss)

        if losstype == "biased" or losstype == "unbiased":
            loss = _loss_biased(model, x_tensor, y_tensor)
        elif losstype == "fdiv":
            loss = _loss_fdiv(model, x_tensor, y_tensor)
            
        cur_MI_estimate = -loss.detach()
        
        if global_MI_estimate is None:
            global_MI_estimate = cur_MI_estimate
            steps_without_improvement = 0

        if cur_MI_estimate > global_MI_estimate:
            global_MI_estimate = cur_MI_estimate
            steps_without_improvement = 0
        else:
            steps_without_improvement += 1

        if steps_without_improvement > patience:
            break

        print("{} [{}] (lr = {})".format(global_MI_estimate, steps_without_improvement,
                                         opt.param_groups[0]["lr"]))
        
    return global_MI_estimate

def binned_estimator(X_in, Y_in, binning_method = "tukey"):

    assert len(X_in) == len(Y_in)

    def _get_cellucci_binning(data):
        data = data.flatten()
        data_min = np.min(data)
        data_max = np.max(data)

        number_bins = int(np.sqrt(float(len(data)) / 5.0))
        percentile_getter = functools.partial(np.quantile, a = data)
         
        # test the binning with the current number of bins
        percentiles = np.linspace(0, 1, number_bins + 1)
        uniform_occupancy_binning = [np.quantile(data, percentile) for percentile in percentiles]
        
        return uniform_occupancy_binning

    def _get_bias(self, bins_X, bins_Y, number_samples):
        if not isinstance(bins_X, int):
            bins_X = len(bins_X)

        if not isinstance(bins_Y, int):
            bins_Y = len(bins_Y)

        return (bins_X * bins_Y - 1.0) / (2.0 * number_samples)

    if binning_method == "tukey":
        bins_X = bins_Y = int(np.sqrt(len(X_in)))
    elif binning_method == "bendat_piersol":
        bins_X = bins_Y = int(1.87 * np.power(len(X_in) - 1, 0.4))
    elif binning_method == "cellucci_approximated":
        bins_X = bins_Y = int(np.sqrt(float(len(X_in)) / 5.0))
    elif binning_method == "cellucci":
        bins_X = _get_cellucci_binning(X_in)
        bins_Y = _get_cellucci_binning(Y_in)
    else:
        raise NotImplementedError("Error: selected heuristic not implemented!")         

    bins = [bins_X, bins_Y]
    eps = 1e-6

    X_in = X_in.flatten()
    Y_in = Y_in.flatten()

    # estimate the densities with histograms
    p_XY = np.histogram2d(X_in, Y_in, bins = bins)[0].astype(float) + eps
    p_XY /= float(np.sum(p_XY))

    # get the marginals
    p_X = np.sum(p_XY, axis = 0).flatten().astype(float)
    p_X /= float(np.sum(p_X)) # note: will already be normalised correctly anyways!

    p_Y = np.sum(p_XY, axis = 1).flatten().astype(float)
    p_Y /= float(np.sum(p_Y)) # note: will already be normalised correctly anyways!

    MI = np.sum(p_XY * np.log(p_XY)) - np.sum(p_X * np.log(p_X)) - np.sum(p_Y * np.log(p_Y))

    return MI