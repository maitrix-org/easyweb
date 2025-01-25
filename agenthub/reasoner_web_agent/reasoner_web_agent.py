from reasoners import ReasonerAgent

from opendevin.controller.agent import Agent
from opendevin.controller.state.state import State
from opendevin.core.logger import opendevin_logger as logger
from opendevin.events.action import Action
from opendevin.llm.llm import LLM
from opendevin.runtime.plugins import (
    PluginRequirement,
)
from opendevin.runtime.tools import RuntimeTool


class ReasonerWebAgent(Agent):
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
        if 'Meta-Llama-3.1-70B-Instruct' in llm.model_name:
            self.config_name = 'opendevin_llama'
        elif 'gpt-4o-mini' in llm.model_name:
            self.config_name = 'opendevin_mini_world_model'
        else:
            # self.config_name = 'opendevin'
            self.config_name = 'opendevin_world_model'

        logger.info(f'Using {self.config_name}')
        self.agent = ReasonerAgent(llm, config_name=self.config_name, logger=logger)
        self.reset()

    def reset(self) -> None:
        """
        Resets the agent.
        """
        self.agent.reset()

    def step(self, env_state: State) -> Action:
        return self.agent.step(env_state)

    def search_memory(self, query: str) -> list[str]:
        raise NotImplementedError('Implement this abstract method')
