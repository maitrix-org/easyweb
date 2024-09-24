from opendevin.controller.agent import Agent

from .few_shot_world_model_agent import FewShotWorldModelAgent

Agent.register('FewShotWorldModelAgent', FewShotWorldModelAgent)
