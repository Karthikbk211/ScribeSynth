import torch
from tqdm import tqdm


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())


class Diffusion:
    def __init__(self, noise_steps=1000, noise_offset=0, beta_start=1e-4, beta_end=0.02, device=None):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.noise_offset = noise_offset
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(
            1 - self.alpha_hat[t])[:, None, None, None]
        eps = torch.randn_like(x) + self.noise_offset * torch.randn(
            x.shape[0], x.shape[1], 1, 1).to(self.device)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * eps, eps

    def sample_timesteps(self, n, finetune=False):
        low = 6 if finetune else 0
        return torch.randint(low=low, high=self.noise_steps, size=(n,))

    def train_ddim(self, model, x, styles, freq_input, content, total_t, sampling_timesteps=50, eta=0):
        total_timesteps = total_t
        times = [-1] + [i /
                        sampling_timesteps for i in range(1, sampling_timesteps + 1)]
        times = list(reversed(times))
        time_pairs = list(zip(times[:-1], times[1:]))

        noise_list = []

        for time, time_next in tqdm(time_pairs, position=1, leave=False, desc='sampling'):
            time = (total_timesteps * time).long().to(self.device)
            time_next = (total_timesteps * time_next).long().to(self.device)

            predicted_noise, high_nce_emb, low_nce_emb, confidence = model(
                x, time, styles, freq_input, content, tag='train')
            noise_list.append(predicted_noise)

            beta = self.beta[time][:, None, None, None]
            alpha_hat = self.alpha_hat[time][:, None, None, None]
            alpha_hat_next = self.alpha_hat[time_next][:, None, None, None]

            x_start = (x - (1 - alpha_hat).sqrt() *
                       predicted_noise) / alpha_hat.sqrt()

            if time_next[0] < 0:
                x = x_start
                continue

            sigma = eta * (beta * (1 - alpha_hat_next) /
                           (1 - alpha_hat)).sqrt()
            c = (1 - alpha_hat_next - sigma ** 2).sqrt()
            noise = torch.randn_like(x)

            x = x_start * alpha_hat_next.sqrt() + c * predicted_noise + sigma * noise

        return x, noise_list[0], high_nce_emb, low_nce_emb, confidence

    @torch.no_grad()
    def ddim_sample(self, model, vae, n, x, styles, freq_input, content, sampling_timesteps=50, eta=0):
        model.eval()

        total_timesteps = self.noise_steps
        times = torch.linspace(-1, total_timesteps - 1,
                               steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        for time, time_next in tqdm(time_pairs, position=1, leave=False, desc='sampling'):
            time = (torch.ones(n) * time).long().to(self.device)
            time_next = (torch.ones(n) * time_next).long().to(self.device)

            predicted_noise, confidence = model(
                x, time, styles, freq_input, content)

            beta = self.beta[time][:, None, None, None]
            alpha_hat = self.alpha_hat[time][:, None, None, None]
            alpha_hat_next = self.alpha_hat[time_next][:, None, None, None]

            x_start = (x - (1 - alpha_hat).sqrt() *
                       predicted_noise) / alpha_hat.sqrt()

            if time_next[0] < 0:
                x = x_start
                continue

            sigma = eta * (beta * (1 - alpha_hat_next) /
                           (1 - alpha_hat)).sqrt()
            c = (1 - alpha_hat_next - sigma ** 2).sqrt()
            noise = torch.randn_like(x)

            x = x_start * alpha_hat_next.sqrt() + c * predicted_noise + sigma * noise

        model.train()

        latents = 1 / 0.18215 * x
        image = vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).contiguous().numpy()
        image = torch.from_numpy(image)
        x = image.permute(0, 3, 1, 2).contiguous()

        return x, confidence

    @torch.no_grad()
    def ddpm_sample(self, model, vae, n, x, styles, freq_input, content):
        model.eval()

        for i in tqdm(reversed(range(0, self.noise_steps)), position=1, leave=False, desc='sampling'):
            time = (torch.ones(n) * i).long().to(self.device)
            predicted_noise, confidence = model(
                x, time, styles, freq_input, content)

            alpha = self.alpha[time][:, None, None, None]
            alpha_hat = self.alpha_hat[time][:, None, None, None]
            beta = self.beta[time][:, None, None, None]

            noise = torch.randn_like(x) if i > 0 else torch.zeros_like(x)
            x = 1 / torch.sqrt(alpha) * (
                x - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * predicted_noise
            ) + torch.sqrt(beta) * noise

        model.train()

        latents = 1 / 0.18215 * x
        image = vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).contiguous().numpy()
        image = torch.from_numpy(image)
        x = image.permute(0, 3, 1, 2).contiguous()

        return x, confidence
