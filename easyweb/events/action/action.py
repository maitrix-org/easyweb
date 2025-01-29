from dataclasses import dataclass
from typing import ClassVar

from easyweb.events.event import Event


@dataclass
class Action(Event):
    runnable: ClassVar[bool] = False
