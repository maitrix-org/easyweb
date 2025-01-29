from reasoners import ReasonerAgent

from easyweb.controller.agent import Agent
from easyweb.controller.state.state import State
from easyweb.core.logger import easyweb_logger as logger
from easyweb.events.action import Action
from easyweb.llm.llm import LLM
from easyweb.runtime.plugins import (
    PluginRequirement,
)
from easyweb.runtime.tools import RuntimeTool


class ReasonerAgentFull(Agent):
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
        if 'gpt-4o-mini' in llm.model_name:
            self.config_name = 'easyweb_mini_world_model'
        else:
            self.config_name = 'easyweb_world_model'

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
