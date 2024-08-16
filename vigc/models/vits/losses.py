import torch


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    real_losses = 0
    generated_losses = 0
    for disc_real, disc_generated in zip(disc_real_outputs, disc_generated_outputs):
        real_loss = torch.mean((1 - disc_real) ** 2)
        generated_loss = torch.mean(disc_generated ** 2)
        loss += real_loss + generated_loss
        real_losses += real_loss
        generated_losses += generated_loss

    return loss, real_losses, generated_losses


def feature_loss(feature_maps_real, feature_maps_generated):
    loss = 0
    for feature_map_real, feature_map_generated in zip(feature_maps_real, feature_maps_generated):
        for real, generated in zip(feature_map_real, feature_map_generated):
            real = real.detach()
            loss += torch.mean(torch.abs(real - generated))

    return loss * 2


def generator_loss(disc_outputs):
    total_loss = 0
    gen_losses = []
    for disc_output in disc_outputs:
        disc_output = disc_output
        loss = torch.mean((1 - disc_output) ** 2)
        gen_losses.append(loss)
        total_loss += loss

    return total_loss, gen_losses


def kl_loss(prior_latents, posterior_log_variance, prior_means, prior_log_variance, labels_mask):
    """
    z_p, logs_q: [b, h, t_t]
    prior_means, prior_log_variance: [b, h, t_t]
    """

    kl = prior_log_variance - posterior_log_variance - 0.5
    kl += 0.5 * ((prior_latents - prior_means) ** 2) * torch.exp(-2.0 * prior_log_variance)
    kl = torch.sum(kl * labels_mask)
    loss = kl / torch.sum(labels_mask)
    return loss
