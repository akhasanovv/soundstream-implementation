import torch

from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    trainer for soundstream
    """
    def set_grad(self, module, requires_grad):
        if module is None:
            return
        
        for param in module.parameters():
            param.requires_grad_(requires_grad)

    def step_scheduler(self, scheduler):
        if scheduler is None:
            return
        
        scheduler.step()
        
    def compute_perplexity(self, codebook_indices, num_embeddings):
        perplexities = {}
        all_probs = []

        for i, indices in enumerate(codebook_indices):
            flat = indices.reshape(-1)
            counts = torch.bincount(flat, minlength=num_embeddings).float()
            probs = counts / counts.sum().clamp_min(1.0)
            entropy = -(probs * (probs + 1e-12).log()).sum()
            perplexities[f"codebook_perplexity_{i}"] = entropy.exp().item()
            all_probs.append(probs)

        if all_probs:
            avg_probs = torch.stack(all_probs, dim=0).mean(dim=0)
            agg_entropy = -(avg_probs * (avg_probs + 1e-12).log()).sum()
            perplexities["codebook_perplexity"] = agg_entropy.exp().item()

        return perplexities
    
    def _process_batch_adversarial(self, batch, metrics: MetricTracker):
        # discriminator sstep
        self.optimizer["discriminator"].zero_grad()

        with torch.no_grad():
            outputs = self.model(**batch)
        batch.update(outputs)

        real_audio = batch["audio"]
        fake_audio = batch["reconstructed_audio"].detach()

        d_real_outputs = self.discriminator(real_audio)
        d_fake_outputs = self.discriminator(fake_audio)
        
        d_losses = self.criterion.discriminator_loss(
            real_outputs=d_real_outputs,
            fake_outputs=d_fake_outputs,
        )
        
        # upd discriminator
        d_losses["loss_d"].backward()
        self.optimizer["discriminator"].step()
        self.step_scheduler(self.lr_scheduler.get("discriminator"))
        
        # generator step
        self.optimizer["generator"].zero_grad()
        outputs = self.model(**batch)
        batch.update(outputs)

        # eval with discriminator, no upd
        self.set_grad(self.discriminator, False)
        
        d_real_outputs = self.discriminator(real_audio)
        d_fake_outputs = self.discriminator(batch["reconstructed_audio"])
        
        g_losses = self.criterion.generator_loss(
            audio=real_audio,
            reconstructed_audio=batch["reconstructed_audio"],
            commit_loss=batch.get("commit_loss"),
            real_outputs=d_real_outputs,
            fake_outputs=d_fake_outputs,
        )
        
        g_losses["loss"].backward()
        self.optimizer["generator"].step()
        
        self.set_grad(self.discriminator, True)
        self.step_scheduler(self.lr_scheduler.get("generator"))

        batch.update(g_losses)
        batch.update(d_losses)
        
        if "codebook_indices" in batch and self.config.model.get("num_embeddings") is not None:
            batch.update(
                self.compute_perplexity(
                    batch["codebook_indices"], self.config.model.num_embeddings
                )
            )
        
        return batch

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]

            if self.discriminator is not None:
                batch = self._process_batch_adversarial(batch=batch, metrics=metrics)
            else:
                self.optimizer.zero_grad()
                
                outputs = self.model(**batch)
                batch.update(outputs)
                all_losses = self.criterion(**batch)
                batch.update(all_losses)
                batch["loss"].backward()
                
                self._clip_grad_norm()
                self.optimizer.step()
                
                self.step_scheduler(self.lr_scheduler)
                if "codebook_indices" in batch and self.config.model.get("num_embeddings") is not None:
                    batch.update(
                        self.compute_perplexity(
                            batch["codebook_indices"], self.config.model.num_embeddings
                        )
                    )
        else:
            outputs = self.model(**batch)
            batch.update(outputs)
            all_losses = self.criterion(**batch)
            batch.update(all_losses)
            
            if "codebook_indices" in batch and self.config.model.get("num_embeddings") is not None:
                batch.update(
                    self.compute_perplexity(
                        batch["codebook_indices"], self.config.model.num_embeddings
                    )
                )

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))
        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        if self.writer is None:
            return

        sample_rate = self.config.metrics.get("sample_rate", 16000)

        if "audio" in batch:
            self.writer.add_audio(
                f"{mode}_audio_original",
                batch["audio"][0].detach().cpu(),
                sample_rate=sample_rate
            )

        if "reconstructed_audio" in batch:
            self.writer.add_audio(
                f"{mode}_audio_reconstructed",
                batch["reconstructed_audio"][0].detach().cpu(),
                sample_rate=sample_rate
            )

        if "codebook_indices" in batch:
            for i, indices in enumerate(batch["codebook_indices"]):
                self.writer.add_histogram(
                    f"{mode}_codebook_indices_{i}",
                    indices.detach().float().cpu(),
                    bins=self.config.model.get("num_embeddings", None)
                )
