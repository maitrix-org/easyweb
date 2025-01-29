from reasoners import ReasonerAgent

from fast_web.controller.agent import Agent
from fast_web.controller.state.state import State
from fast_web.core.logger import fast_web_logger as logger
from fast_web.events.action import Action
from fast_web.llm.llm import LLM
from fast_web.runtime.plugins import (
    PluginRequirement,
)
from fast_web.runtime.tools import RuntimeTool


class ReasonerAgentFast(Agent):
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
            self.config_name = 'fast_web_mini'
        else:
            self.config_name = 'fast_web'

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
