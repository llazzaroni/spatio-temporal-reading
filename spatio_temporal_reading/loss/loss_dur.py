import torch
from torch.distributions import MultivariateNormal, LogNormal, Normal

def NegLogLikelihood(
        weights,
        durations_model,
        durations_target,
        std_dur
):
    dur_scale = std_dur.expand_as(durations_model)

    distribution_durations = LogNormal(durations_model, dur_scale)
    
    durations_expanded = durations_target.unsqueeze(2)
    log_prob_dur = distribution_durations.log_prob(durations_expanded)

    log_weights = torch.log(weights + 1e-12)

    log_prob = torch.logsumexp(log_prob_dur + log_weights, dim=-1)

    log_p_seq = log_prob.sum(dim=1)
    log_likelihood = log_p_seq.mean()
    

    # Return the negative log likelihood
    loss = -log_likelihood
    return loss

def NegLogLikelihood_np(
        weights,
        positions_model,
        saccades_model,
        positions,
        saccades,
        cov_pos,
        std_sacc
):
    # weights has dimensions (n_batches, len_sequence, n_admixtures_components)
    # positions_model has dimensions (n_batches, len_sequence, n_admixtures_components, 2)
    # saccades_model has dimensions (n_batches, len_sequence, n_admixtures_components)
    # positions has dimensions (n_batches, len_sequence, 2)
    # saccades_model has dimensions (n_batches, len_sequence)

    # var_pos has dimensions (2,2)
    # std_sacc has dimension (1)
    scale_sacc = std_sacc.expand_as(saccades_model)

    distribution_positions = MultivariateNormal(positions_model, cov_pos)

    distribution_saccades = LogNormal(saccades_model, scale_sacc)

    positions_expanded = positions.unsqueeze(2)
    saccades_expanded = saccades.unsqueeze(2)

    log_prob_pos = distribution_positions.log_prob(positions_expanded)
    log_prob_sacc = distribution_saccades.log_prob(saccades_expanded)

    log_weights = torch.log(weights + 1e-12)

    log_prob = torch.logsumexp(log_prob_pos + log_prob_sacc + log_weights, dim=-1) # (n_batches, len_sequence)

    neg_log_prob = -log_prob
    neg_log_prob_flattened = neg_log_prob.reshape(-1)
    neg_log_prob_np = neg_log_prob_flattened.detach().cpu().numpy()

    return neg_log_prob_np

def NegLogLikelihoodCov(
        weights,
        durations_model,
        durations_target,
        std,
        std_dur
):
    distribution_durations = Normal(durations_model, std)
    
    durations_expanded = durations_target.unsqueeze(2)
    log_prob_dur = distribution_durations.log_prob(durations_expanded)

    log_weights = torch.log(weights + 1e-12)

    log_prob = torch.logsumexp(log_prob_dur + log_weights, dim=-1)

    log_p_seq = log_prob.sum(dim=1)
    log_likelihood = log_p_seq.mean()
    

    # Return the negative log likelihood
    loss = -log_likelihood
    return loss

def NegLogLikelihoodCov_np(
        weights,
        durations_model,
        durations_target,
        std,
        std_dur
):
    distribution_durations = Normal(durations_model, std)
    
    durations_expanded = durations_target.unsqueeze(2)
    log_prob_dur = distribution_durations.log_prob(durations_expanded)

    log_weights = torch.log(weights + 1e-12)

    log_prob = torch.logsumexp(log_prob_dur + log_weights, dim=-1)

    neg_log_prob = -log_prob
    neg_log_prob_flattened = neg_log_prob.reshape(-1)
    neg_log_prob_np = neg_log_prob_flattened.detach().cpu().numpy()

    return neg_log_prob_np