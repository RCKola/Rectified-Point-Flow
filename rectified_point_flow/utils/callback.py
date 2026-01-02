import torch
from lightning.pytorch.callbacks import Callback

class NaNTracker(Callback):
    def on_after_backward(self, trainer, pl_module):
        for name, param in pl_module.named_parameters():
            if param.grad is not None:
                if not torch.isfinite(param.grad).all():
                    print(f"\n[!] NaN Gradients detected in: {name}", flush=True)
                    # This will give you the stack trace you've been missing
                    raise ValueError(f"Gradient collapse in {name} at step {trainer.global_step}")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Checks the loss returned to the trainer
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs
        if not torch.isfinite(loss):
            print(f"\n[!] NaN Loss detected at step {trainer.global_step}", flush=True)
            raise ValueError("Loss exploded to NaN")