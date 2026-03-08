import os
import torch
import torch.nn as nn
from tqdm import tqdm
import torchvision


class Trainer:
    def __init__(self, diffusion, model, vae, criterion, optimizer,
                 train_loader, logs, test_loader, device,
                 ocr_model=None, ctc_loss=None, max_steps=200000):
        self.diffusion = diffusion
        self.model = model
        self.vae = vae
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.logs = logs
        self.device = device
        self.ocr_model = ocr_model
        self.ctc_loss = ctc_loss
        self.global_step = 0
        self.save_every = 5000
        self.sample_every = 2000
        self.max_steps = max_steps
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_steps, eta_min=1e-6)

    def encode_with_vae(self, imgs):
        with torch.no_grad():
            latents = self.vae.encode(imgs).latent_dist.sample()
            latents = latents * 0.18215
        return latents

    def decode_with_vae(self, latents):
        with torch.no_grad():
            latents = 1 / 0.18215 * latents
            images = self.vae.decode(latents).sample
        return images

    def save_checkpoint(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Unwrap DDP so checkpoint keys don't have 'module.' prefix
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        torch.save(model_to_save.state_dict(), path)
        print(f'checkpoint saved to {path}')

    def train_step(self, batch):
        imgs = batch['img'].to(self.device)
        style = batch['style'].to(self.device)
        freq = batch['freq'].to(self.device)
        content = batch['content'].to(self.device)
        target = batch['target'].to(self.device)
        target_lengths = batch['target_lengths'].to(self.device)

        latents = self.encode_with_vae(imgs)
        t = self.diffusion.sample_timesteps(latents.shape[0],
                                            finetune=self.ocr_model is not None).to(self.device)
        noisy_latents, noise = self.diffusion.noise_images(latents, t)

        predicted_noise, high_nce_emb, low_nce_emb, confidence = self.model(
            noisy_latents, t, style, freq, content, tag='train')

        recon_loss = self.criterion['recon'](predicted_noise, noise)
        high_nce_loss = self.criterion['nce'](high_nce_emb)
        low_nce_loss = self.criterion['nce'](low_nce_emb)
        nce_loss = high_nce_loss + low_nce_loss

        total_loss = recon_loss + 0.1 * nce_loss

        if self.ocr_model is not None:
            # Inline x_start recovery (predict_start_from_noise equivalent)
            alpha_hat = self.diffusion.alpha_hat[t][:, None, None, None]
            x_start = (noisy_latents - (1 - alpha_hat).sqrt() * predicted_noise) / alpha_hat.sqrt()

            # Detach from the diffusion grad graph before decoding;
            # OCR loss must not backprop through the frozen VAE or OCR model.
            with torch.no_grad():
                generated_imgs = self.decode_with_vae(x_start.detach())
                generated_imgs = (generated_imgs / 2 + 0.5).clamp(0, 1)
                ocr_input = generated_imgs.mean(dim=1, keepdim=True)
                ocr_input = ocr_input.repeat(1, 4, 1, 1)

            ocr_output = self.ocr_model(ocr_input)
            ocr_output = ocr_output.log_softmax(2)
            input_lengths = torch.full((ocr_output.shape[1],),
                                       ocr_output.shape[0], dtype=torch.long).to(self.device)
            ctc = self.ctc_loss(ocr_output, target, input_lengths, target_lengths)
            total_loss = total_loss + 0.1 * ctc

        return total_loss, recon_loss, nce_loss

    def train(self):
        self.model.train()
        epoch = 0

        while self.global_step < self.max_steps:
            epoch += 1
            self.train_loader.sampler.set_epoch(epoch)
            pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')

            for batch in pbar:
                if self.global_step >= self.max_steps:
                    break
                self.optimizer.zero_grad()
                total_loss, recon_loss, nce_loss = self.train_step(batch)
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()
                self.global_step += 1

                current_lr = self.scheduler.get_last_lr()[0]
                pbar.set_postfix({
                    'total': f'{total_loss.item():.4f}',
                    'recon': f'{recon_loss.item():.4f}',
                    'nce': f'{nce_loss.item():.4f}',
                    'lr': f'{current_lr:.2e}'
                })

                if self.logs is not None:
                    self.logs.info(
                        f'step {self.global_step} | total: {total_loss.item():.4f} '
                        f'| recon: {recon_loss.item():.4f} | nce: {nce_loss.item():.4f} '
                        f'| lr: {current_lr:.2e}')

                if self.global_step % self.save_every == 0:
                    self.save_checkpoint(
                        f'checkpoints/scribesynth_step{self.global_step}.pth')

                if self.global_step % self.sample_every == 0:
                    self.sample_and_save(batch)

    @torch.no_grad()
    def sample_and_save(self, batch):
        self.model.eval()
        style = batch['style'][:1].to(self.device)
        freq = batch['freq'][:1].to(self.device)
        content = batch['content'][:1].to(self.device)

        x = torch.randn((1, 4, style.shape[2] // 8,
                         (content.shape[1] * 32) // 8)).to(self.device)

        sampled, confidence = self.diffusion.ddim_sample(
            self.model, self.vae, 1, x, style, freq, content)

        os.makedirs('samples', exist_ok=True)
        torchvision.utils.save_image(
            sampled, f'samples/step_{self.global_step}.png')

        self.model.train()
