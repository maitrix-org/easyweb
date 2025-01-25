from fast_web.controller.agent import Agent

from .browsing_agent import BrowsingAgent

Agent.register('BrowsingAgent', BrowsingAgent)
