import ast
import json
import os
import random
import time
import traceback
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
from .utils import (
    ParseError,
    parse_html_tags_raise,
)

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

DUMP_FOLDER = '../prompt-logs-onepass'
if not os.path.exists(DUMP_FOLDER):
    os.makedirs(DUMP_FOLDER)
# DUMP_FOLDER = None

client = OpenAI()


# class ParseError(Exception):
#     pass


class OnepassAgent(Agent):
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
        print(self.llm.max_input_tokens)
        print(self.llm.max_output_tokens)
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
        self.temperature = 0.0
        self.max_retry = 4

        self.reset()

        self.dump_counter = 0

    # added
    def count_tokens(self, messages):
        return self.llm.get_token_count(messages)

    def reduce_ax_tree(self, ax, goal_token):
        low, high = 0, len(ax)

        while low < high:
            mid = (low + high + 1) // 2
            if self.count_tokens([{'role': 'user', 'content': ax[:mid]}]) <= goal_token:
                low = mid
            else:
                high = mid - 1

        return ax[:low]

    def truncate_messages(self, messages, max_tokens):
        if self.count_tokens(messages) > max_tokens:
            tree_start = messages[-1]['content'].find('AXSTART')
            tree_end = messages[-1]['content'].find('AXEND')

            no_ax = (
                messages[-1]['content'][0:tree_start]
                + messages[-1]['content'][tree_end:]
            )
            ax = messages[-1]['content'][tree_start + len('AXSTART') : tree_end]

            new_message = {'role': 'user', 'content': no_ax}
            tmp_messages = []
            tmp_messages.append(messages[0])
            tmp_messages.append(new_message)

            no_ax_token = self.count_tokens(tmp_messages)
            goal_token = max_tokens - no_ax_token
            reduced_ax = self.reduce_ax_tree(ax, goal_token)

            processed_content = (
                messages[-1]['content'][0:tree_start]
                + reduced_ax
                + messages[-1]['content'][tree_end:]
            )
            messages[-1]['content'] = processed_content

            # print(self.count_tokens(messages))
            # print(messages[-1]['content'])
            assert self.count_tokens(messages) <= max_tokens
            return messages
        else:
            return messages

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

    def retry(
        self,
        messages,
        parser,
        n_retry=4,
        log=True,
        min_retry_wait_time=60,
        rate_limit_max_wait_time=60 * 30,
        override_llm=False,
    ):
        tries = 0
        rate_limit_total_delay = 0
        while tries < n_retry and rate_limit_total_delay < rate_limit_max_wait_time:
            if not override_llm:
                # truncated_messages = self.truncate_messages(
                #     messages, MAX_TOKENS - OUTPUT_BUFFER
                # )  # added
                response = self.llm.completion(
                    messages=messages,
                    # messages=truncated_messages,  # added
                    temperature=self.temperature,
                    top_p=0.7,
                    stop=None,
                )
                answer = response['choices'][0]['message']['content'].strip()
            else:
                tmp_llm = 'gpt-4o'
                logger.info('Overriding LLM with ' + tmp_llm)
                response = client.chat.completions.create(
                    model=tmp_llm,
                    messages=messages,
                    temperature=self.temperature,
                    stop=None,
                )

                answer = response.choices[0].message.content.strip()

            # with open("/home/demo/jinyu/prompts/last_answer.txt", "w") as f:
            #     f.write(answer)

            messages.append({'role': 'assistant', 'content': answer})

            value, valid, retry_message = parser(answer)
            if valid:
                self.log_cost(response)
                return value

            tries += 1
            if log:
                msg = f'Query failed. Retrying {tries}/{n_retry}.\n[LLM]:\n{answer}\n[User]:\n{retry_message}'
                logger.info(msg)
            messages.append({'role': 'user', 'content': retry_message})

        raise ValueError(f'Could not parse a valid value after {n_retry} retries.')

    def get_llm_output_new(
        self, prompt, output_keys, with_system_prompt=True, temperature=0.0
    ):
        tmp = self.temperature
        self.temperature = temperature
        # logger.info(system_msg)
        messages = []
        if with_system_prompt:
            system_msg = prompt_factory.get_system_prompt()
            messages.append({'role': 'system', 'content': system_msg})
        messages.append({'role': 'user', 'content': prompt})

        def parser(text):
            try:
                ans_dict = parse_html_tags_raise(
                    text,
                    keys=output_keys,
                    merge_multiple=True,
                )
            except ParseError as e:
                return None, False, str(e)
            return ans_dict, True, ''

        try:
            ans_dict = self.retry(messages, parser, n_retry=self.max_retry)
            ans_dict['n_retry'] = (len(messages) - 3) / 2
        except ValueError as e:
            # Likely due to maximum retry. We catch it here to be able to return
            # the list of messages for further analysis
            ans_dict = {k: None for k in output_keys}
            ans_dict['err_msg'] = str(e)
            ans_dict['stack_trace'] = traceback.format_exc()
            ans_dict['n_retry'] = self.max_retry

        ans_dict['messages'] = messages
        ans_dict['prompt'] = prompt

        self.temperature = tmp
        return ans_dict

    def _onepass(self, current_obs, history, goal):
        onepass_prompt = prompt_factory.get_onepass_prompt(current_obs, history, goal)
        answer_dict = self.get_llm_output_new(
            onepass_prompt, ['state', 'instruction', 'action'], temperature=1.0
        )

        if DUMP_FOLDER is not None:
            with open(f'{DUMP_FOLDER}/{self.dump_counter}-onepass.json', 'w') as f:
                json.dump(answer_dict, f, indent=4)
            self.dump_counter += 1

        return answer_dict['state'], answer_dict['instruction'], answer_dict['action']

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
        self.current_state, self.current_instruction, self.current_action = (
            self._onepass(self.current_obs, self.history, self.goal)
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

    def log_cost(self, response):
        # TODO: refactor to unified cost tracking
        try:
            cur_cost = self.llm.completion_cost(response)
        except Exception:
            cur_cost = 0
        self.cost_accumulator += cur_cost
        logger.info(
            'Cost: %.2f USD | Accumulated Cost: %.2f USD',
            cur_cost,
            self.cost_accumulator,
        )
