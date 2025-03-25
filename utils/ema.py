"""
Exponential Moving Average (EMA) implementation for model weights.
Used in PAL method for model stabilization.
"""
import torch
from copy import deepcopy


class ModelEMA(object):
    """
    Model Exponential Moving Average
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    """
    def __init__(self, args, model, decay):
        """
        Initialize EMA with a model and a decay factor.
        
        Args:
            args: arguments object containing device information
            model: model to create EMA from
            decay: decay factor for EMA (typically close to 1, e.g., 0.999)
        """
        self.ema = deepcopy(model)
        self.ema.to(args.device)
        self.ema.eval()
        self.decay = decay
        self.ema_has_module = hasattr(self.ema, 'module')
        self.param_keys = [k for k, _ in self.ema.named_parameters()]
        self.buffer_keys = [k for k, _ in self.ema.named_buffers()]
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        """
        Update EMA parameters using the current model state.
        
        Args:
            model: current model with updated parameters
        """
        needs_module = hasattr(model, 'module') and not self.ema_has_module
        with torch.no_grad():
            msd = model.state_dict()
            esd = self.ema.state_dict()
            for k in self.param_keys:
                if needs_module:
                    j = 'module.' + k
                else:
                    j = k
                model_v = msd[j].detach()
                ema_v = esd[k]
                esd[k].copy_(ema_v * self.decay + (1. - self.decay) * model_v)

            for k in self.buffer_keys:
                if needs_module:
                    j = 'module.' + k
                else:
                    j = k
                esd[k].copy_(msd[j])
