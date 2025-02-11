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
        - llm (Any): The llm to be used by this agent
        """
        super().__init__(llm)

        self.config_name = 'easyweb'
        if 'gpt-4o-mini' in llm.model_name:
            self.config_name = 'easyweb_mini'

        if 'o1' in llm.model_name or 'o3-mini' in llm.model_name:
            llm = {
                'default': LLM(
                    model='gpt-4o', api_key=llm.api_key, base_url=llm.base_url
                ),
                'policy': llm,
            }
        elif 'deepseek-reasoner' in llm.model_name:
            llm = {
                'default': LLM(
                    model='deepseek/deepseek-chat',
                    api_key=llm.api_key,
                    base_url=llm.base_url,
                ),
                'policy': llm,
            }

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
