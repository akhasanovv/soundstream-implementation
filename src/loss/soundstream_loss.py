import torch
import torchaudio
from torch import nn

import torch.nn.functional as F

class SoundStreamLoss(nn.Module):
    """
    losses for soundstream 

    generator loss:
        1*reconstruction + 1*commitment + 1*adversarial + 100*feature_matching

    discriminator loss::
        1*mean(relu(1 - d(audio_real))) + 1*mean(relu(1 + d(audio_fake)))
    """

    def __init__(
        self,
        sample_rate=16000,
        n_mels=64,
        scales=(64, 128, 256, 512, 1024, 2048),
        recon_weight=1.0,
        commit_weight=1.0,
        adv_weight=1.0,
        feature_matching_weight=100.0,
        log_eps=1e-5
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.scales = scales
        self.recon_weight = recon_weight
        self.commit_weight = commit_weight
        self.adv_weight = adv_weight
        self.feature_matching_weight = feature_matching_weight
        self.log_eps = log_eps

        self.mel_spectrograms = nn.ModuleDict({
            str(scale): torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=scale,
                win_length=scale,
                hop_length=scale//4,
                n_mels=n_mels,
                power=1.0,
                center=True,
                mel_scale="htk"
            )
            for scale in self.scales
        })
        
    def get_mel(self, audio, scale):
        if audio.dim() == 3:
            audio = audio.squeeze(1)

        mel = self.mel_spectrograms[str(scale)](audio)
        return mel.transpose(1, 2)

    def reconstruction_loss(self, audio, reconstructed_audio):
        total = audio.new_zeros(())

        for scale in self.scales:
            real_mel = self.get_mel(audio, scale)
            fake_mel = self.get_mel(reconstructed_audio, scale)

            mel_loss = F.l1_loss(fake_mel, real_mel)
            log_mel_loss = F.mse_loss(
                fake_mel.clamp_min(self.log_eps).log(),
                real_mel.clamp_min(self.log_eps).log()
            )

            total += mel_loss + ((scale / 2) ** 0.5) * log_mel_loss

        return total

    def get_iter_discriminator(self, outputs):
        yield outputs["stft"]
        yield from outputs["wave"]

    def discriminator_loss(self, real_outputs, fake_outputs):
        real_losses = []
        fake_losses = []

        for (real_logits, _), (fake_logits, _) in zip(
            self.get_iter_discriminator(real_outputs),
            self.get_iter_discriminator(fake_outputs)
        ):
            real_losses.append(F.relu(1.0 - real_logits).mean())
            fake_losses.append(F.relu(1.0 + fake_logits).mean())

        loss_real = sum(real_losses) / len(real_losses)
        loss_fake = sum(fake_losses) / len(fake_losses)

        return {
            "loss_d": loss_real + loss_fake,
            "loss_d_real": loss_real,
            "loss_d_fake": loss_fake
        }

    def adversarial_loss(self, fake_outputs):
        fake_losses = []

        for fake_logits, _ in self.get_iter_discriminator(fake_outputs):
            fake_losses.append(F.relu(1.0 - fake_logits).mean())

        return sum(fake_losses) / len(fake_losses)

    def feature_matching_loss(self, real_outputs, fake_outputs):
        losses = []

        for (_, real_features), (_, fake_features) in zip(
            self.get_iter_discriminator(real_outputs),
            self.get_iter_discriminator(fake_outputs)
        ):
            for real_feature, fake_feature in zip(real_features[:-1], fake_features[:-1]):
                losses.append(F.l1_loss(fake_feature, real_feature.detach()))

        if len(losses) == 0:
            fake_logits, _ = fake_outputs["stft"]
            return fake_logits.new_zeros(())

        return sum(losses) / len(losses)

    def forward(self, audio, reconstructed_audio, commit_loss=None, **batch):
        """
        Loss function calculation logic.
        """
        recon_loss = self.reconstruction_loss(audio, reconstructed_audio)

        if commit_loss is None:
            commit_loss = reconstructed_audio.new_zeros(())

        loss = self.recon_weight * recon_loss + self.commit_weight * commit_loss

        return {
            "loss": loss,
            "recon_loss": recon_loss,
            "commit_loss": commit_loss
        }

    def generator_loss(self, audio, reconstructed_audio, commit_loss, real_outputs, fake_outputs):
        base_losses = self.forward(audio, reconstructed_audio, commit_loss)
        adv_loss = self.adversarial_loss(fake_outputs)
        fm_loss = self.feature_matching_loss(real_outputs, fake_outputs)

        loss = base_losses["loss"] + self.adv_weight * adv_loss + self.feature_matching_weight * fm_loss

        return {
            "loss": loss,
            "recon_loss": base_losses["recon_loss"],
            "commit_loss": base_losses["commit_loss"],
            "adv_loss": adv_loss,
            "fm_loss": fm_loss
        }
        