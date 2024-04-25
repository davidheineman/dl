from typing import Iterable
from torch import Tensor
import torch

from trak.modelout_functions import AbstractModelOutput

class CausalLMModelOutput(AbstractModelOutput):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def get_output(
        model,
        weights: Iterable[Tensor],
        buffers: Iterable[Tensor],
        input_id: Tensor,
        attention_mask: Tensor,
        label: Tensor,
    ) -> Tensor:
        # Call OLMo for prediction
        kw_inputs = {
            "input_ids": input_id.unsqueeze(0),
            "attention_mask": attention_mask.unsqueeze(0),
        }
        logits = torch.func.functional_call(
            model, (weights, buffers), args=(), kwargs=kw_inputs
        )

        # Compare logits to gold labels
        label = label.to(torch.long)

        bindex = torch.arange(logits.shape[0]).to(logits.device, non_blocking=False)
        logits_correct = logits[bindex, label.unsqueeze(0)]

        # Calculate model output function
        cloned_logits = logits.clone()
        cloned_logits[bindex, label.unsqueeze(0)] = torch.tensor(
            -torch.inf, device=logits.device, dtype=logits.dtype
        )
        margins = logits_correct - cloned_logits.logsumexp(dim=-1)
        out = margins.sum()

        return out

    @staticmethod
    def get_out_to_loss_grad(
        model, weights, buffers, batch: Iterable[Tensor]
    ) -> Tensor:
        input_ids, attention_mask, labels = batch

        # Call OLMo for prediction
        kw_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        logits = torch.func.functional_call(
            model, (weights, buffers), args=(), kwargs=kw_inputs
        )

        softmax = torch.nn.Softmax(-1)
        loss_temperature = 1
        
        # Return loss term for prediction
        labels = labels.to(torch.long)

        ps = softmax(logits / loss_temperature)[torch.arange(logits.size(0)), labels]
        out = (1 - ps)
        out = out.clone().detach().unsqueeze(-1)

        return out
