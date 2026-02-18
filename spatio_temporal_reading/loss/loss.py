import torch
from torch.distributions import MultivariateNormal, LogNormal

def NegLogLikelihood(
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

    log_p_seq = log_prob.sum(dim=1) # (n_batches)

    log_likelihood = log_p_seq.mean()  # (1)

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
    neg_log_prob_np = neg_log_prob_flattened.numpy()

    return neg_log_prob_np

def NegLogLikelihoodCov(
        covariances2D,
        covariancesSacc,
        weights,
        positions_model,
        saccades_model,
        positions,
        saccades
):
    # weights has dimensions (n_batches, len_sequence, n_admixtures_components)
    # positions_model has dimensions (n_batches, len_sequence, n_admixtures_components, 2)
    # saccades_model has dimensions (n_batches, len_sequence, n_admixtures_components)
    # positions has dimensions (n_batches, len_sequence, 2)
    # saccades_model has dimensions (n_batches, len_sequence)

    # var_pos has dimensions (2,2)
    # std_sacc has dimension (1)

    distribution_positions = MultivariateNormal(positions_model, covariances2D)

    distribution_saccades = LogNormal(saccades_model, covariancesSacc)

    positions_expanded = positions.unsqueeze(2)
    saccades_expanded = saccades.unsqueeze(2)

    log_prob_pos = distribution_positions.log_prob(positions_expanded)
    log_prob_sacc = distribution_saccades.log_prob(saccades_expanded)

    log_weights = torch.log(weights + 1e-12)

    log_prob = torch.logsumexp(log_prob_pos + log_prob_sacc + log_weights, dim=-1) # (n_batches, len_sequence)

    log_p_seq = log_prob.sum(dim=1) # (n_batches)

    log_likelihood = log_p_seq.mean()  # (1)

    # Return the negative log likelihood
    loss = -log_likelihood
    return loss

def NegLogLikelihoodCov_np(
        covariances2D,
        covariancesSacc,
        weights,
        positions_model,
        saccades_model,
        positions,
        saccades
        ):
    # weights has dimensions (n_batches, len_sequence, n_admixtures_components)
    # positions_model has dimensions (n_batches, len_sequence, n_admixtures_components, 2)
    # saccades_model has dimensions (n_batches, len_sequence, n_admixtures_components)
    # positions has dimensions (n_batches, len_sequence, 2)
    # saccades_model has dimensions (n_batches, len_sequence)

    # var_pos has dimensions (2,2)
    # std_sacc has dimension (1)

    distribution_positions = MultivariateNormal(positions_model, covariances2D)

    distribution_saccades = LogNormal(saccades_model, covariancesSacc)

    positions_expanded = positions.unsqueeze(2)
    saccades_expanded = saccades.unsqueeze(2)

    log_prob_pos = distribution_positions.log_prob(positions_expanded)
    log_prob_sacc = distribution_saccades.log_prob(saccades_expanded)

    log_weights = torch.log(weights + 1e-12)

    log_prob = torch.logsumexp(log_prob_pos + log_prob_sacc + log_weights, dim=-1) # (n_batches, len_sequence)

    neg_log_prob = -log_prob
    neg_log_prob_flattened = neg_log_prob.reshape(-1)
    neg_log_prob_np = neg_log_prob_flattened.numpy()

    return neg_log_prob_np