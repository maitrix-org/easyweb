from dotenv import load_dotenv

from opendevin.controller.agent import Agent

from .micro.agent import MicroAgent
from .micro.registry import all_microagents

load_dotenv()


from . import (  # noqa: E402
    SWE_agent,
    agent_model_agent,
    browsing_agent,
    codeact_agent,
    codeact_swe_agent,
    delegator_agent,
    dummy_agent,
    dummy_web_agent,
    few_shot_world_model_agent,
    monologue_agent,
    new_world_model_agent,
    onepass_agent,
    planner_agent,
    policy_agent,
    web_planning_agent,
    world_model_agent,
)

__all__ = [
    'monologue_agent',
    'codeact_agent',
    'codeact_swe_agent',
    'planner_agent',
    'SWE_agent',
    'delegator_agent',
    'dummy_agent',
    'browsing_agent',
    'world_model_agent',
    'dummy_web_agent',
    'new_world_model_agent',
    'onepass_agent',
    'policy_agent',
    'few_shot_world_model_agent',
    'agent_model_agent',
    'web_planning_agent',
]

for agent in all_microagents.values():
    name = agent['name']
    prompt = agent['prompt']

    anon_class = type(
        name,
        (MicroAgent,),
        {
            'prompt': prompt,
            'agent_definition': agent,
        },
    )

    Agent.register(name, anon_class)
