from opendevin.controller.agent import Agent

from .web_planning_agent import WebPlanningAgent

Agent.register('WebPlanningAgent', WebPlanningAgent)
