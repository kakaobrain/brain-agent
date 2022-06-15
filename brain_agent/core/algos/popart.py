import torch

def update_mu_sigma(nu, mu, vs, task_ids, popart_clip_min, clamp_max, beta):
    oldnu = nu.clone()
    oldsigma = torch.sqrt(oldnu - mu ** 2)
    oldsigma[torch.isnan(oldsigma)] = popart_clip_min
    oldsigma = torch.clamp(oldsigma, min=popart_clip_min, max=clamp_max)
    oldmu = mu.clone()

    for i in range(len(task_ids)):
        task_id = task_ids[i]
        v = torch.mean(vs[i])

        mu[task_id] = (1 - beta) * mu[task_id] + beta * v
        nu[task_id] = (1 - beta) * nu[task_id] + beta * (v ** 2)

    sigma = torch.sqrt(nu - mu ** 2)
    sigma[torch.isnan(sigma)] = popart_clip_min
    sigma = torch.clamp(sigma, min=popart_clip_min, max=clamp_max)

    return mu, nu, sigma, oldmu, oldsigma

def update_parameters(weight, bias, mu, sigma, oldmu, oldsigma):
    new_weight = (weight.t() * oldsigma / sigma).t()
    new_bias = (oldsigma * bias + oldmu - mu) / sigma
    return new_weight, new_bias