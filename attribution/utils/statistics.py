from captum.attr import IntegratedGradients
import torch

def get_attr_posx_E(input_model, input_baseline, t_idx, model):
    def forward_func_posx(x):
        weights, positions, saccades, cov2d, covsacc = model(x)
        return (weights[:, t_idx, :] * positions[:, t_idx, :, 0]).sum(dim=-1)
    ig = IntegratedGradients(forward_func_posx)
    attr, delta = ig.attribute(
        inputs=input_model,
        baselines=input_baseline,
        n_steps=64,
        return_convergence_delta=True
    )
    return attr

def get_attr_posy_E(input_model, input_baseline, t_idx, model):
    def forward_func_posx(x):
        weights, positions, saccades, cov2d, covsacc = model(x)
        return (weights[:, t_idx, :] * positions[:, t_idx, :, 1]).sum(dim=-1)
    ig = IntegratedGradients(forward_func_posx)
    attr, delta = ig.attribute(
        inputs=input_model,
        baselines=input_baseline,
        n_steps=64,
        return_convergence_delta=True
    )
    return attr

def get_attr_sacc_E(input_model, input_baseline, t_idx, model):
    def forward_func_posx(x):
        weights, positions, saccades, cov2d, covsacc = model(x)
        return (weights[:, t_idx, :] * saccades[:, t_idx, :]).sum(dim=-1)
    ig = IntegratedGradients(forward_func_posx)
    attr, delta = ig.attribute(
        inputs=input_model,
        baselines=input_baseline,
        n_steps=64,
        return_convergence_delta=True
    )
    return attr

def one_sample_posx(input_model, input_baseline, t_idx, model):
    attr = get_attr_posx_E(input_model, input_baseline, t_idx, model)
    tot_contributions = attr.squeeze(0).abs().sum(dim=0)
    div = tot_contributions.sum().item()
    tot_contributions /= div
    return [
        tot_contributions[0].item(),
        tot_contributions[1].item(),
        tot_contributions[2].item(),
        tot_contributions[3].item(),
        tot_contributions[-9].item(),
        tot_contributions[-8].item(),
        tot_contributions[-7].item(),
        tot_contributions[-6].item(),
    ]

def one_sample_posy(input_model, input_baseline, t_idx, model):
    attr = get_attr_posy_E(input_model, input_baseline, t_idx, model)
    tot_contributions = attr.squeeze(0).abs().sum(dim=0)
    div = tot_contributions.sum().item()
    tot_contributions /= div
    return [
        tot_contributions[0].item(),
        tot_contributions[1].item(),
        tot_contributions[2].item(),
        tot_contributions[3].item(),
        tot_contributions[-9].item(),
        tot_contributions[-8].item(),
        tot_contributions[-7].item(),
        tot_contributions[-6].item(),
    ]

def one_sample_sacc(input_model, input_baseline, t_idx, model):
    attr = get_attr_sacc_E(input_model, input_baseline, t_idx, model)
    tot_contributions = attr.squeeze(0).abs().sum(dim=0)
    div = tot_contributions.sum().item()
    tot_contributions /= div
    return [
        tot_contributions[0].item(),
        tot_contributions[1].item(),
        tot_contributions[2].item(),
        tot_contributions[3].item(),
        tot_contributions[-9].item(),
        tot_contributions[-8].item(),
        tot_contributions[-7].item(),
        tot_contributions[-6].item(),
    ]
    