import torch

from specforge.lr_scheduler import CosineAnnealingWarmupLR, LinearWarmupLR
from specforge.utils import print_on_rank0


class BF16Optimizer:
    def __init__(
        self,
        model,
        lr,
        weight_decay=0.0,
        max_grad_norm=0.5,
        total_steps=800_000,
        warmup_ratio=0.015,
        warmup_steps=None,
        scheduler_type="cosine",
        betas=(0.9, 0.95),
    ):
        # These magic numbers are copied from official EAGLE3:
        # https://github.com/SafeAILab/EAGLE/blob/main/eagle/traineagle3/ds_config.json
        # - weight_decay=0.0, max_grad_norm=0.5, total_steps=800k, warmup_steps=12k
        # - betas=(0.9, 0.95)
        self.model = model
        self.model_params = [p for p in model.parameters() if p.requires_grad]
        self.max_grad_norm = max_grad_norm
        self.fp32_params = [
            p.detach().clone().to(torch.float32) for p in self.model_params
        ]
        for mp in self.fp32_params:
            mp.requires_grad = True
        self.optimizer = torch.optim.AdamW(
            self.fp32_params, lr=lr, weight_decay=weight_decay, betas=betas
        )
        # Use warmup_steps if provided, otherwise calculate from warmup_ratio
        if warmup_steps is not None:
            actual_warmup_steps = warmup_steps
        else:
            actual_warmup_steps = int(warmup_ratio * total_steps)

        # Select scheduler type (default: cosine, official EAGLE3 uses linear)
        if scheduler_type == "linear":
            self.scheduler = LinearWarmupLR(
                self.optimizer,
                total_steps=total_steps,
                warmup_steps=actual_warmup_steps,
            )
        elif scheduler_type == "cosine":
            self.scheduler = CosineAnnealingWarmupLR(
                self.optimizer,
                total_steps=total_steps,
                warmup_steps=actual_warmup_steps,
            )
        else:
            raise ValueError(f"Unknown scheduler_type: {scheduler_type}. Use 'linear' or 'cosine'.")

    def step(self):
        with torch.no_grad():
            for p, mp in zip(self.model_params, self.fp32_params):
                mp.grad = (
                    p.grad.detach().to(torch.float32) if p.grad is not None else None
                )
        torch.nn.utils.clip_grad_norm_(self.fp32_params, self.max_grad_norm)
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()
        with torch.no_grad():
            for p, mp in zip(self.model_params, self.fp32_params):
                p.data.copy_(mp.data.to(p.dtype))
                p.grad = None

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict["optimizer_state_dict"])
        print_on_rank0("Successfully loaded optimizer state_dict.")
        self.scheduler.load_state_dict(state_dict["scheduler_state_dict"])
        print_on_rank0("Successfully loaded scheduler state_dict.")

    def state_dict(self):
        return {
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }

    def get_learning_rate(self):
        return self.optimizer.param_groups[0]["lr"]
