from dataclasses import dataclass

from easyweb.events.event import Event


@dataclass
class Observation(Event):
    content: str
