from dataclasses import dataclass

from fast_web.core.schema import ActionType

from .action import Action


@dataclass
class StartPlanningAction(Action):
    eta_seconds: float
    action: str = ActionType.START_PLANNING

    @property
    def message(self) -> str:
        return f'Planning... ETA: {self.eta_seconds:.1f} seconds'

    def __str__(self) -> str:
        return f'**StartPlanning** (eta_seconds={self.eta_seconds:.1f})'


@dataclass
class FinishPlanningAction(Action):
    next_step: str
    action: str = ActionType.FINISH_PLANNING

    @property
    def message(self) -> str:
        return self.next_step

    def __str__(self) -> str:
        return f'**FinishPlanning**\nNEXT STEP: {self.next_step}'
