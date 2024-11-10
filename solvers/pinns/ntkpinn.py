

from .pinn import PINN
import torch

try:
    from torch.optim.lr_scheduler import LRScheduler  # torch >= 2.0
except ImportError:
    from torch.optim.lr_scheduler import (
        _LRScheduler as LRScheduler,
    )  # torch < 2.0

from torch.optim.lr_scheduler import ConstantLR

class NTKPINN(PINN):
    def __init__(self,
        problem,
        model,
        extra_features=None,
        loss=torch.nn.MSELoss(),
        optimizer=torch.optim.Adam,
        optimizer_kwargs={"lr": 0.001},
        scheduler=ConstantLR,
        scheduler_kwargs={"factor": 1, "total_iters": 0},):
        super().__init__(problem, model, extra_features, loss, optimizer, optimizer_kwargs, scheduler, scheduler_kwargs)

        # storing
        _lambdas = None
        self._reset_lambdas()

    def training_step(self, batch, _):
        loss_value = super().training_step(batch, _)
        if len(self._lambdas) != len(self.problem.conditions):
            raise TypeError("The number of collected lamdas does not correspond to the number of conditions of the problem")

        rescaled_loss_value = sum(self._lambdas) * loss_value
        self._reset_lambdas()
        return rescaled_loss_value.as_subclass(torch.Tensor) #as_subclass serve a trattarlo come un tensore normale, altrimenti dà errore
    
    def loss_phys(self, samples, equation):
        residual = self.compute_residual(samples=samples, equation=equation)
        loss_value = self.loss(
            torch.zeros_like(residual, requires_grad=True), residual
        )
        self.store_log(loss_value=float(loss_value))
        lam = torch.trace(torch.matmul(residual.view(1,-1), residual.view(-1,1)))          # compute lam via trace of r@rT
        self._lambdas.append(lam)
        return (1/lam)*loss_value
    
    def _reset_lambdas(self):
        self._lambdas = []

    #Here _lambdas is not defined as a dictionary for two reasons:
    #1) It may be not useful since I only need _lambdas when I have to sum them all, so I don't ever need to pick them
    #2) When I compute the loss_phys I only have the equation and the points, but not the corresponding condition

    #def _reset_lambdas(self):
    #    self._lambdas = {condition : None for condition in self.problem.conditions}
