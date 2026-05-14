
import torch
from torchvision import transforms

@torch.no_grad()
def ddpm_sample(model, scheduler, shape, device, num_steps=1000):
    x = torch.randn(shape, device=device)

    for t_val in reversed(range(num_steps)):
        t = torch.full((shape[0],), t_val, device=device, dtype=torch.long)
        pred_noise = model(x, t)

        alpha = scheduler.alphas[t_val]
        alpha_bar = scheduler.alphas_cumprod[t_val]
        beta = scheduler.betas[t_val]

        x0_pred = (x - (1 - alpha_bar).sqrt() * pred_noise) / alpha_bar.sqrt()
        x0_pred = x0_pred.clamp(-1, 1)

        mean = (alpha.sqrt() * (1 - scheduler.alphas_cumprod_prev[t_val]) * x
                + scheduler.alphas_cumprod_prev[t_val].sqrt() * beta * x0_pred) \
               / (1 - alpha_bar)

        if t_val > 0:
            noise = torch.randn_like(x)
            var = scheduler.posterior_variance[t_val].clamp(min=1e-20)
            x = mean + var.sqrt() * noise
        else:
            x = mean
    x = transforms.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225])(x)
    
    x = x * 255.0
    return x