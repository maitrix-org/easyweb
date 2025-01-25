from dataclasses import dataclass

from fast_web.core.schema import ActionType

from .action import Action


@dataclass
class NullAction(Action):
    """An action that does nothing."""

    action: str = ActionType.NULL

    @property
    def message(self) -> str:
        return 'No action'
