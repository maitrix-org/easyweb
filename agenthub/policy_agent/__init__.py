from opendevin.controller.agent import Agent

from .policy_agent import PolicyAgent

Agent.register('PolicyAgent', PolicyAgent)
