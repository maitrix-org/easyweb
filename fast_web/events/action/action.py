from dataclasses import dataclass
from typing import ClassVar

from fast_web.events.event import Event


@dataclass
class Action(Event):
    runnable: ClassVar[bool] = False
