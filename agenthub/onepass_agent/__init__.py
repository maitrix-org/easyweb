from opendevin.controller.agent import Agent

from .onepass_agent import OnepassAgent

Agent.register('OnepassAgent', OnepassAgent)
