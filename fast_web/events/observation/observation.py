from dataclasses import dataclass

from fast_web.events.event import Event


@dataclass
class Observation(Event):
    content: str
