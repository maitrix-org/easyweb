import ast
import json
import os
import random
import time
from typing import Any, Dict, List

from browsergym.utils.obs import flatten_axtree_to_str
from openai import OpenAI

from opendevin.controller.agent import Agent
from opendevin.controller.state.state import State
from opendevin.core.logger import llm_output_logger
from opendevin.core.logger import opendevin_logger as logger
from opendevin.events.action import (
    Action,
    AgentFinishAction,
    BrowseInteractiveAction,
    MessageAction,
)
from opendevin.events.event import EventSource
from opendevin.events.observation import BrowserOutputObservation
from opendevin.llm.llm import LLM
from opendevin.runtime.plugins import (
    PluginRequirement,
)
from opendevin.runtime.tools import RuntimeTool

from . import prompt_factory

USE_NAV = (
    os.environ.get('USE_NAV', 'true') == 'true'
)  # only disable NAV actions when running webarena and miniwob benchmarks
USE_CONCISE_ANSWER = (
    os.environ.get('USE_CONCISE_ANSWER', 'false') == 'true'
)  # only return concise answer when running webarena and miniwob benchmarks

if not USE_NAV and USE_CONCISE_ANSWER:
    EVAL_MODE = True  # disabled NAV actions and only return concise answer, for webarena and miniwob benchmarks\
else:
    EVAL_MODE = False

MAX_TOKENS = 32768  # added
OUTPUT_BUFFER = 1100  # added
# DEFAULT_BROWSER = 'https://www.google.com'  # added
DEFAULT_BROWSER = None

DUMP_FOLDER = '../prompt-logs-policy'
# DUMP_FOLDER = None

client = OpenAI()

ENCODER_KEY, POLICY_KEY, EFF_KEY = 'encoder', 'policy', 'effectuator'
KEY_LST = [ENCODER_KEY, POLICY_KEY, EFF_KEY]

# class ParseError(Exception):
#     pass


class PolicyAgent(Agent):
    VERSION = '1.0'
    """
    An agent that interacts with the browser.
    """

    sandbox_plugins: list[PluginRequirement] = []
    runtime_tools: list[RuntimeTool] = [RuntimeTool.BROWSER]

    def __init__(
        self,
        llm: LLM,
    ) -> None:
        """
        Initializes a new instance of the BrowsingAgent class.

        Parameters:
        - llm (LLM): The llm to be used by this agent
        """
        super().__init__(llm)
        # define a configurable action space, with chat functionality, web navigation, and webpage grounding using accessibility tree and HTML.
        # see https://github.com/ServiceNow/BrowserGym/blob/main/core/src/browsergym/core/action/highlevel.py for more details
        # action_subsets = ['chat', 'bid']
        # if USE_NAV:
        #     action_subsets.append('nav')
        # self.action_space = HighLevelActionSet(
        #     subsets=action_subsets,
        #     strict=False,  # less strict on the parsing of the actions
        #     multiaction=False,  # enable to agent to take multiple actions at once
        # )
        assert isinstance(llm, dict)
        assert set(llm.keys()) == set(KEY_LST)

        self.temperature = 0.0
        self.max_retry = 4

        self.reset()

        self.dump_counter = 0
        self.llm_dict = llm

    # # added
    # def count_tokens(self, messages, module):
    #     return self.llm_dict[module].get_token_count(messages)

    # def reduce_ax_tree(self, ax, goal_token):
    #     low, high = 0, len(ax)

    #     while low < high:
    #         mid = (low + high + 1) // 2
    #         if self.count_tokens([{'role': 'user', 'content': ax[:mid]}]) <= goal_token:
    #             low = mid
    #         else:
    #             high = mid - 1

    #     return ax[:low]

    # def truncate_messages(self, messages, max_tokens):
    #     if self.count_tokens(messages) > max_tokens:
    #         tree_start = messages[-1]['content'].find('AXSTART')
    #         tree_end = messages[-1]['content'].find('AXEND')

    #         no_ax = (
    #             messages[-1]['content'][0:tree_start]
    #             + messages[-1]['content'][tree_end:]
    #         )
    #         ax = messages[-1]['content'][tree_start + len('AXSTART') : tree_end]

    #         new_message = {'role': 'user', 'content': no_ax}
    #         tmp_messages = []
    #         tmp_messages.append(messages[0])
    #         tmp_messages.append(new_message)

    #         no_ax_token = self.count_tokens(tmp_messages)
    #         goal_token = max_tokens - no_ax_token
    #         reduced_ax = self.reduce_ax_tree(ax, goal_token)

    #         processed_content = (
    #             messages[-1]['content'][0:tree_start]
    #             + reduced_ax
    #             + messages[-1]['content'][tree_end:]
    #         )
    #         messages[-1]['content'] = processed_content

    #         # print(self.count_tokens(messages))
    #         # print(messages[-1]['content'])
    #         assert self.count_tokens(messages) <= max_tokens
    #         return messages
    #     else:
    #         return messages

    def reset(self) -> None:
        """
        Resets the Browsing Agent.
        """
        super().reset()
        self.cost_accumulator = 0
        self.error_accumulator = 0

        self.history = []
        self.action_history: List[tuple[str, str]] = []
        self.obs_history: List[Dict[str, Any]] = []
        self.full_output: str = ''
        self.full_output_dict: Dict[str, Any] = {}

    def parse_response(self, response: str, thought: str) -> Action:
        # thought = ''
        action_str = response

        # handle send message to user function call in BrowserGym
        msg_content = ''
        for sub_action in action_str.split('\n'):
            if 'send_msg_to_user(' in sub_action:
                tree = ast.parse(sub_action)
                args = tree.body[0].value.args  # type: ignore
                msg_content = args[0].value

        return BrowseInteractiveAction(
            browser_actions=action_str,
            thought=thought,
            browsergym_send_msg_to_user=msg_content,
        )

    def get_llm_output(self, prompt, module, temperature=0.0):
        tmp = self.temperature
        self.temperature = temperature
        messages = []
        messages.append({'role': 'user', 'content': prompt})

        try:
            response = self.llm_dict[module].completion(
                messages=messages,
                # messages=truncated_messages,  # added
                temperature=self.temperature,
                stop=None,
            )
            answer = response['choices'][0]['message']['content'].strip()
            self.log_cost(response, module)
        except ValueError:
            # Likely due to maximum retry. We catch it here to be able to return
            # the list of messages for further analysis
            return None

        self.temperature = tmp
        return answer

    def _encoder(self, current_obs, history, goal):
        encoder_prompt = prompt_factory.get_encoder_prompt(current_obs, history, goal)
        answer = self.get_llm_output(encoder_prompt, ENCODER_KEY)

        if DUMP_FOLDER is not None:
            with open(f'{DUMP_FOLDER}/{self.dump_counter}-encoder.txt', 'w') as f:
                f.write(encoder_prompt + answer)
            self.dump_counter += 1

        return answer

    def _policy(self, state, history, goal):
        policy_prompt = prompt_factory.get_policy_prompt(state, history, goal)
        answer = self.get_llm_output(policy_prompt, POLICY_KEY)

        if DUMP_FOLDER is not None:
            with open(f'{DUMP_FOLDER}/{self.dump_counter}-policy.txt', 'w') as f:
                f.write(policy_prompt + answer)
            self.dump_counter += 1

        return answer

    def _effectuator(self, state, instruction, current_obs, history, goal):
        effectuator_prompt = prompt_factory.get_effectuator_prompt(
            state, instruction, current_obs, history, goal
        )
        answer = self.get_llm_output(effectuator_prompt, EFF_KEY)

        if DUMP_FOLDER is not None:
            with open(f'{DUMP_FOLDER}/{self.dump_counter}-effectuator.txt', 'w') as f:
                f.write(effectuator_prompt + answer)
            self.dump_counter += 1

        return answer

    def step(self, env_state: State) -> Action:
        """
        Performs one step using the Browsing Agent.
        This includes gathering information on previous steps and prompting the model to make a browsing command to execute.

        Parameters:
        - env_state (State): used to get updated info

        Returns:
        - BrowseInteractiveAction(browsergym_command) - BrowserGym commands to run
        - MessageAction(content) - Message action to run (e.g. ask for clarification)
        - AgentFinishAction() - end the interaction
        - StartPlanningAction(eta_seconds) - Indicates start of planning
        """

        # Set default first action
        # if DEFAULT_BROWSER is not None and len(self.actions) == 0:
        #     time.sleep(4)
        #     action = "goto('{}')".format(DEFAULT_BROWSER)
        #     self.actions.append(action)
        #     return BrowseInteractiveAction(
        #         browser_actions=action, thought='Open default browser'
        #     )
        # actions = self.actions
        # if DEFAULT_BROWSER is not None:
        #     actions = actions[1:]

        last_obs, last_action, return_action = self.process_control_flow(env_state)
        if return_action is not None:
            return return_action

        current_obs, return_action = self.parse_current_obs(last_obs)
        if return_action is not None:
            return return_action

        self.current_obs = current_obs
        self.obs_history.append(current_obs)
        self.add_to_log('obs', current_obs['clean_axtree_txt'])
        self.last_action = (
            self.action_history[-1]
            if len(self.action_history) > 0
            else ('No action taken so far', '')
        )

        self.current_state = self._encoder(self.current_obs, self.history, self.goal)
        self.current_instruction = self._policy(
            self.current_state, self.history, self.goal
        )
        self.current_action = self._effectuator(
            self.current_state,
            self.current_instruction,
            self.current_obs,
            self.history,
            self.goal,
        )

        self.full_output = ''
        self.full_output_dict = {}
        self.full_output_dict['obs'] = current_obs
        self.add_to_log('state', self.current_state)
        self.add_to_log('instruction', self.current_instruction)
        self.add_to_log('action', self.current_action)

        llm_output_logger.info(self.full_output)
        self.full_output_dict['full_output'] = self.full_output
        self.full_output_json = json.dumps(self.full_output_dict)

        self.history.append(
            (
                self.current_obs,
                self.current_state,
                self.current_instruction,
                self.current_action,
            )
        )
        return self.parse_response(self.current_action, self.full_output_json)

    def process_control_flow(self, env_state: State) -> Action:
        goal = env_state.get_current_user_intent()
        if goal is None:
            goal = env_state.inputs['task']
        self.goal = goal

        # messages: List[str] = []
        prev_actions: List[str] = []
        last_obs = None
        last_action = None

        # if EVAL_MODE and len(env_state.history) == 1:
        if len(env_state.history) == 1:
            # for webarena and miniwob++ eval, we need to retrieve the initial observation already in browser env
            # initialize and retrieve the first observation by issuing an noop OP
            # For non-benchmark browsing, the browser env starts with a blank page, and the agent is expected to first navigate to desired websites
            time.sleep(10 + random.random() * 5)
            return (
                last_obs,
                last_action,
                BrowseInteractiveAction(browser_actions='noop()'),
            )

        for prev_action, obs in env_state.history:
            # Go through the history to get the last action
            if isinstance(prev_action, BrowseInteractiveAction):
                # Create a list of past actions
                prev_actions.append(prev_action.browser_actions)
                last_obs = obs
                last_action = prev_action
            elif (
                isinstance(prev_action, MessageAction)
                and prev_action.source == EventSource.AGENT
            ):
                # agent has responded, task finish.
                return (
                    last_obs,
                    last_action,
                    AgentFinishAction(outputs={'content': prev_action.content}),
                )

        if EVAL_MODE:
            prev_actions = prev_actions[1:]  # remove the first noop action

        # prev_action_str = '\n'.join(prev_actions)
        # if the final BrowserInteractiveAction exec BrowserGym's send_msg_to_user,
        # we should also send a message back to the user in OpenDevin and call it a day
        if (
            isinstance(last_action, BrowseInteractiveAction)
            and last_action.browsergym_send_msg_to_user
        ):
            # Here the browser interaction action from BrowserGym can also include a message to the user
            # When we see this browsergym action we should use a MessageAction from OpenDevin
            return (
                last_obs,
                last_action,
                MessageAction(last_action.browsergym_send_msg_to_user),
            )

        return last_obs, last_action, None

    def parse_current_obs(self, last_obs):
        cur_axtree_txt = ''
        error_prefix = ''
        current_obs = {}

        if isinstance(last_obs, BrowserOutputObservation):
            # The browser output observation belongs to OpenDevin
            if last_obs.error:
                # add error recovery prompt prefix
                error_prefix = f'IMPORTANT! Last action is incorrect:\n{last_obs.last_browser_action}\nThink again with the current observation of the page.\n'
            try:
                cur_axtree_txt = flatten_axtree_to_str(
                    last_obs.axtree_object,
                    extra_properties=last_obs.extra_element_properties,
                    with_clickable=True,
                    filter_visible_only=True,
                )
                # {'scrollTop': 0, 'windowHeight': 720, 'documentHeight': 720, 'remainingPixels': 0}
                # cur_axtree_txt = (
                #     f"URL {last_obs.url}\n"
                #     f"Scroll Position: {last_obs.scroll_position['scrollTop']}, "
                #     f"Window Height: {last_obs.scroll_position['windowHeight']}, "
                #     f"Webpage Height: {last_obs.scroll_position['documentHeight']}, "
                #     f"Remaining Pixels: {last_obs.scroll_position['remainingPixels']}\n"
                # ) + cur_axtree_txt
                scroll_progress = (
                    1
                    - last_obs.scroll_position['remainingPixels']
                    / last_obs.scroll_position['documentHeight']
                )
                cur_axtree_txt = (
                    f"URL {last_obs.url}\n"
                    f"Scroll Position: {last_obs.scroll_position['scrollTop']}, "
                    f"Window Height: {last_obs.scroll_position['windowHeight']}, "
                    f"Webpage Height: {last_obs.scroll_position['documentHeight']}, "
                    f"Remaining Pixels: {last_obs.scroll_position['remainingPixels']}, "
                    f"Scrolling Progress: {scroll_progress:.1%}\n"
                ) + cur_axtree_txt
                logger.info(last_obs.scroll_position)
            except Exception as e:
                logger.error(
                    'Error when trying to process the accessibility tree: %s', e
                )
                return current_obs, MessageAction('Error encountered when browsing.')

        if error_prefix:
            self.error_accumulator += 1
            if self.error_accumulator > 20:
                return current_obs, MessageAction(
                    'Too many errors encountered. Task failed.'
                )

        ### Above is record keeping by world model

        clean_axtree_lines = []
        num_static_text_lines = 0
        max_static_text_lines = 10
        for line in cur_axtree_txt.split('\n'):
            if line.strip().startswith('StaticText') or line.strip().startswith(
                'ListMarker'
            ):
                num_static_text_lines += 1
            else:
                num_static_text_lines = 0

            if num_static_text_lines <= max_static_text_lines:
                clean_axtree_lines.append(line)
        clean_axtree_txt = '\n'.join(clean_axtree_lines)

        # current_obs = {
        #     'axtree_txt': clean_axtree_txt,
        #     'raw_axtree_txt': cur_axtree_txt,
        #     # 'axtree_txt': "AXSTART "+cur_axtree_txt+" AXEND",
        #     'last_action_error': error_prefix,
        #     'goal': self.goal,
        # }
        current_obs = {
            'clean_axtree_txt': clean_axtree_txt,
            'raw_axtree_txt': cur_axtree_txt,
            # 'axtree_txt': "AXSTART "+cur_axtree_txt+" AXEND",
            'error_prefix': error_prefix,
            'goal': self.goal,
        }
        return current_obs, None

    def add_to_log(self, key, value):
        key_title = key.replace('_', ' ').title()
        logger.info(f'*{key_title}*: {value}')
        self.full_output += f'*{key_title}*: {value}\n'
        self.full_output_dict[key] = value

    def search_memory(self, query: str) -> list[str]:
        raise NotImplementedError('Implement this abstract method')

    def log_cost(self, response, module):
        # TODO: refactor to unified cost tracking
        try:
            cur_cost = self.llm_dict[module].completion_cost(response)
        except Exception:
            cur_cost = 0
        self.cost_accumulator += cur_cost
        logger.info(
            'Cost: %.2f USD | Accumulated Cost: %.2f USD',
            cur_cost,
            self.cost_accumulator,
        )
