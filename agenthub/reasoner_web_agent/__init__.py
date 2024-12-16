from opendevin.controller.agent import Agent

from .reasoner_web_agent import ReasonerWebAgent

Agent.register('ReasonerWebAgent', ReasonerWebAgent)
