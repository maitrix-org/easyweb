import json
from functools import partial

from opendevin.controller.agent import Agent
from opendevin.core.logger import opendevin_logger as logger
from opendevin.llm.llm import LLM
from opendevin.runtime.plugins import (
    PluginRequirement,
)
from opendevin.runtime.tools import RuntimeTool

from .agent_model.llms import OpenDevinParserLLM
from .agent_model.modules import (
    PolicyPlanner,
    PromptedActor,
    PromptedEncoder,
    PromptedPolicy,
)
from .agent_model.variables import (
    AgentInstructionEnvironmentIdentity,
    OpenDevinBrowserActionSpace,
    OpenDevinBrowserObservationSpace,
    StepKeyValueMemory,
)
from .agent_model_prompts import (
    actor_prompt_template,
    encoder_prompt_template,
    policy_prompt_template,
)
from .utils import ParseError, parse_html_tags_raise


def parser(text, keys, optional_keys=()):
    try:
        ans_dict = parse_html_tags_raise(text, keys, optional_keys)
    except ParseError as e:
        return None, False, str(e)
    return ans_dict, True, ''


class WebPlanningAgent(Agent):
    VERSION = '0.1'
    """
    An agent that uses agent model abstractions to interact with the browser.
    """

    sandbox_plugins: list[PluginRequirement] = []
    runtime_tools: list[RuntimeTool] = [RuntimeTool.BROWSER]

    def __init__(
        self,
        llm: LLM,
    ) -> None:
        """
        Initializes a new instance of the AbstractBrowsingAgent class.

        Parameters:
        - llm (LLM): The llm to be used by this agent
        """
        super().__init__(llm)

        self.action_space = OpenDevinBrowserActionSpace(
            action_subsets=['chat', 'bid'],
            use_nav=True,
            strict=False,
            multiaction=False,
        )
        self.observation_space = OpenDevinBrowserObservationSpace(eval_mode=False)

        agent_name = 'Web Browsing Agent'
        agent_description = 'An automation assistant who responds to \
user instructions by browsing the internet. The assistant strives to \
complete each step accurately, thoroughly, efficiently, and only based on contents \
derived from web observations. It is important to be forthright when it is \
impossible to answer the question or carry out the instruction.'
        self.identity = AgentInstructionEnvironmentIdentity(
            agent_name=agent_name,
            agent_description=agent_description,
            observation_space=self.observation_space,
            action_space=self.action_space,
        )
        encoder_parser = partial(parser, keys=['state'])
        self.encoder_llm = OpenDevinParserLLM(llm, default_parser=encoder_parser)
        # TODO: Produce the prompt templates
        self.encoder = PromptedEncoder(
            self.identity, self.encoder_llm, prompt_template=encoder_prompt_template
        )
        self.memory = StepKeyValueMemory(['state', 'intent'])
        # self.encoder = StateMemoryEncoder(state=self.state, memory=self.memory)

        policy_parser = partial(parser, keys=['intent'])
        self.policy_llm = OpenDevinParserLLM(llm, default_parser=policy_parser)
        self.policy = PromptedPolicy(
            self.identity, self.policy_llm, prompt_template=policy_prompt_template
        )
        self.planner = PolicyPlanner(self.policy)

        action_parser = partial(parser, keys=['action'])
        self.actor_llm = OpenDevinParserLLM(llm, default_parser=action_parser)
        self.actor = PromptedActor(
            self.identity, self.actor_llm, prompt_template=actor_prompt_template
        )

        self.reset()

    def reset(self):
        self.identity.reset()
        self.memory.reset()

    def step(self, env_state):
        observation, info = self.observation_space.parse_observation(env_state)
        if info.get('return_action') is not None:
            return info['return_action']
        self.identity.update(user_instruction=observation['goal'])

        obs_txt = observation['clean_axtree_txt']
        logger.info(f'*Observation*: {obs_txt}')

        state = self.encoder(obs_txt, self.memory, temperature=0)['state']
        logger.info(f'*State*: {state}')

        intent = self.planner(state, self.memory, temperature=0)['intent']
        logger.info(f'*Intent*: {intent}')

        action = self.actor(obs_txt, state, self.memory, intent, temperature=0)[
            'action'
        ]
        logger.info(f'*Action*: {action}')

        step = {
            'observation': observation,
            'state': state,
            'intent': intent,
            'action': action,
        }
        self.memory.update(**step)
        self.memory.step()

        return self.action_space.parse_action(action, thought=json.dumps(step))

    def search_memory(self, query: str) -> list[str]:
        raise NotImplementedError('Implement this abstract method')
