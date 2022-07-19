import torch


def calculate_vtrace(values, rewards, dones, vtrace_rho, vtrace_c,
                     num_trajectories, recurrence, gamma, exclude_last=False):
    values_cpu = values.cpu()
    rewards_cpu = rewards.cpu()
    dones_cpu = dones.cpu()
    vtrace_rho_cpu = vtrace_rho.cpu()
    vtrace_c_cpu = vtrace_c.cpu()

    vs = torch.zeros((num_trajectories * recurrence))
    adv = torch.zeros((num_trajectories * recurrence))

    bootstrap_values = values_cpu[recurrence - 1::recurrence]
    values_BT = values_cpu.view(-1, recurrence)
    next_values = torch.cat([values_BT[:, 1:], bootstrap_values.view(-1, 1)], dim=1).view(-1)
    next_vs = next_values[recurrence - 1::recurrence]

    masked_gammas = (1.0 - dones_cpu) * gamma

    if exclude_last:
        rollout_recurrence = recurrence - 1
        adv[recurrence - 1::recurrence] = rewards_cpu[recurrence - 1::recurrence] + (masked_gammas[recurrence - 1::recurrence] - 1) * next_vs
        vs[recurrence - 1::recurrence] = next_vs * vtrace_rho_cpu[recurrence - 1::recurrence] * adv[recurrence - 1::recurrence]
    else:
        rollout_recurrence = recurrence

    for i in reversed(range(rollout_recurrence)):
        rewards = rewards_cpu[i::recurrence]
        not_done_times_gamma = masked_gammas[i::recurrence]

        curr_values = values_cpu[i::recurrence]
        curr_next_values = next_values[i::recurrence]
        curr_vtrace_rho = vtrace_rho_cpu[i::recurrence]
        curr_vtrace_c = vtrace_c_cpu[i::recurrence]

        delta_s = curr_vtrace_rho * (rewards + not_done_times_gamma * curr_next_values - curr_values)
        adv[i::recurrence] = rewards + not_done_times_gamma * next_vs - curr_values
        next_vs = curr_values + delta_s + not_done_times_gamma * curr_vtrace_c * (next_vs - curr_next_values)
        vs[i::recurrence] = next_vs

    return vs, adv
