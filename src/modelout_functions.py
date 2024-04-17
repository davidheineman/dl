from typing import Iterable
from torch import Tensor
import torch

from trak.modelout_functions import AbstractModelOutput

class CausalLMModelOutput(AbstractModelOutput):
    """Margin for text classification models. This assumes that the model takes
    in input_ids, token_type_ids, and attention_mask.

    .. math::

        \\text{logit}[\\text{correct}] - \\log\\left(\\sum_{i \\neq
        \\text{correct}} \\exp(\\text{logit}[i])\\right)

    """

    def __init__(self, temperature=1.0) -> None:
        super().__init__()
        self.softmax = torch.nn.Softmax(-1)
        self.loss_temperature = temperature

    @staticmethod
    def get_output(
        model,
        weights: Iterable[Tensor],
        buffers: Iterable[Tensor],
        input_id: Tensor,
        attention_mask: Tensor,
        label: Tensor,
    ) -> Tensor:
        kw_inputs = {
            "input_ids": input_id.unsqueeze(0),
            "attention_mask": attention_mask.unsqueeze(0),
        }

        label = label.to(torch.long)

        logits = torch.func.functional_call(
            model, (weights, buffers), args=(), kwargs=kw_inputs
        )
        bindex = torch.arange(logits.shape[0]).to(logits.device, non_blocking=False)
        logits_correct = logits[bindex, label.unsqueeze(0)]

        cloned_logits = logits.clone()
        cloned_logits[bindex, label.unsqueeze(0)] = torch.tensor(
            -torch.inf, device=logits.device, dtype=logits.dtype
        )

        margins = logits_correct - cloned_logits.logsumexp(dim=-1)
        return margins.sum()

    @staticmethod
    def get_out_to_loss_grad(
        model, weights, buffers, batch: Iterable[Tensor]
    ) -> Tensor:
        input_ids, attention_mask, labels = batch
        kw_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        logits = torch.func.functional_call(
            model, (weights, buffers), args=(), kwargs=kw_inputs
        )
        
        labels = labels.to(torch.long)
        softmax, loss_temperature = torch.nn.Softmax(-1), 1
        ps = softmax(logits / loss_temperature)[torch.arange(logits.size(0)), labels]

        return (1 - ps).clone().detach().unsqueeze(-1)

