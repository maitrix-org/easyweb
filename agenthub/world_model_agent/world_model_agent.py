import ast
import itertools
import json
import math

# import multiprocessing
# import multiprocess as mp
import multiprocessing.dummy as mp_dummy
import os
import random
import time
import traceback
from typing import Any, Dict, List, Optional

import numpy as np
from browsergym.core.action.highlevel import HighLevelActionSet
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

from .prompt import MyMainPrompt
from .utils import ParseError

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


client = OpenAI()


# class ParseError(Exception):
#     pass


def sample_action(obs_history, states, strategies, explanations, actions, policy):
    main_prompt = MyMainPrompt(
        obs_history=obs_history,
        states=states,
        strategies=strategies,
        explanations=explanations,
        actions=actions,
    )
    strategy, summary = policy(main_prompt)
    return strategy, summary


def sample_action_reward(
    obs_history, states, strategies, explanations, actions, action_reward
):
    main_prompt = MyMainPrompt(
        obs_history=obs_history,
        states=states,
        strategies=strategies,
        explanations=explanations,
        actions=actions,
    )
    fast_reward, think, response = action_reward(main_prompt)
    return fast_reward, think, response


class WorldModelAgent(Agent):
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
        action_subsets = ['chat', 'bid']
        if USE_NAV:
            action_subsets.append('nav')
        self.action_space = HighLevelActionSet(
            subsets=action_subsets,
            strict=False,  # less strict on the parsing of the actions
            multiaction=False,  # enable to agent to take multiple actions at once
        )
        self.temperature = 0.0
        self.max_retry = 4

        self.reset()

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

        self.actions: List[str] = []
        self.explanations: List[str] = []
        self.obs_history: List[Dict[str, Any]] = []
        self.states: List[str] = []
        self.evaluations: List[str] = []
        self.strategies: List[Optional[str]] = []
        self.active_strategy: Optional[str] = None
        self.full_output: str = ''
        self.full_output_dict: Dict[str, Any] = {}
        self.active_strategy_turns: int = 0

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
                truncated_messages = self.truncate_messages(
                    messages, MAX_TOKENS - OUTPUT_BUFFER
                )  # added
                response = self.llm.completion(
                    # messages=messages,
                    messages=truncated_messages,  # added
                    temperature=self.temperature,
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

    def get_llm_output(self, prompt, parse_func, output_keys, override_llm=False):
        #         system_msg = f"""\
        # # Instructions
        # Review the current state of the page and all other information to find the best possible next action to accomplish your goal. Your answer will be interpreted and executed by a program, make sure to follow the formatting instructions.

        # # Goal:
        # {self.goal}
        # Stop and ask the user when contact or personal information is required to proceed
        # with the task.

        # # Action Space
        # {self.action_space.describe(with_long_description=False, with_examples=True)}

        # # Note 1
        # You should not attempt to visit the following domains as they will block your entry:
        # - Reddit: www.reddit.com
        # - Zillow: www.zillow.com
        # - StreetEasy: www.streeteasy.com

        # # Note 2
        # You should not attempt to enter personal information unless the user tells you to.
        # If you encounter a situation where you need to enter personal information, stop and
        # ask the user to supply it.
        # """
        #         addition = """
        # Stop and ask the user when their own information is required to proceed with the task, like name, phone, email, login credentials, and more. Do not stop if the information is what you need to search for."""
        system_msg = f"""\
# Instructions
Review the current state of the page and all other information to find the best possible next action to accomplish your goal. Your answer will be interpreted and executed by a program, make sure to follow the formatting instructions.

# Goal:
{self.goal}

# Goal Tips
- Stop and ask the user when their personal information (e.g., name, phone, email, login credentials) is required to proceed. Do not stop if the information is only needed for a search.
- When searching for information online, visit multiple websites for comprehensive information before responding.
- Avoid messaging the user with information during the exploration. Save your notes internally and provide a comprehensive final answer. Only message the user if you are unable to find specific information, explaining what you have done so far.

# Action Space
{self.action_space.describe(with_long_description=False, with_examples=True)}

# Action Tips
- Always enclose string inputs in 'single quotes', including bid inputs.
- If the corresponding bid is not visible, scroll down until it appears.
- Your response will be executed as a Python function call, so ensure it adheres to the format and argument data type specifications defined in the action space.

# Domain Blacklist
Do not visit the following domains as they will block your entry:
- www.reddit.com
- www.zillow.com
- www.streeteasy.com
- www.apartmentfinder.com
- www.quora.com
- www.expedia.com
- www.tripadvisor.com
- www.ticketmaster.com
- www.indeed.com
- www.walmart.com
- www.newegg.com
- www.realtor.com
- www.glassdoor.com
- www.seatgeek.com
- www.vividseats.com
If you accidentally enter any of these websites, go back or revisit Google to try other websites.

# Browsing Tips
- Interact only with elements on the current page starting with bracketed IDs; others are for information or out of view.
- Scroll up and down if more information is needed.
- Respond to dialogs immediately to proceed. Accept cookies, select "No Thanks" for insurance offers, and click "Continue" or "Select" button if relevant boxes are filled out.
- You can only open one tab at a time. You can only interact with elements starting with bids; the rest are for information only or out of view.
- If you are blocked by CAPTCHA, consider going back to the previous page or restart your search.
- If you use go_back() repeatedly but cannot go back to the previous page, consider going to www.google.com to restart your browsing.
- "Alh6id" and "lst-ib" are not valid element IDs.

# Combobox Tips
- If a combobox is not clickable, clicking on the section containing it may bring up a dialog with options to select from.
- If a combobox has autocomplete, filling it with text may bring up a dialog with related options that you can click on to select from.
"""

        # - If you enter arithmetic expressions into Google Search, you will see a calculator tool showing the computed result. The term "Alh6id" is not a valid id you can enter.
        #         # Browsing Tips
        # - Interact only with elements on the current page starting with bracketed IDs; others are for information or out of view.
        # - Scroll up and down if more information is needed.
        # - Respond to dialogs immediately to proceed. These typically appear at the end of the page and might not have the label "dialog". Accept cookies, select "No Thanks" for insurance offers, and click "Continue" or "Select" button if relevant boxes are filled out.
        # - You can only open one tab at a time. You can only interact with elements starting with bids; the rest are for information only or out of view.
        # - If you are blocked by CAPTCHA, consider going back to the previous page or restart your search.
        # - If you have trouble with go_back(), consider going back to www.google.com.

        # # Combobox Tips
        # - If a combobox is not clickable, clicking on the section containing it may bring up a dialog with options to select from.
        # - If a combobox has autocomplete, first fill it with text, which may bring up a dialog with related options. Then click on one of the options to select it.
        #         scrolling_prompt = """
        # scroll(delta_x: float, delta_y: float)
        #     Examples:
        #         scroll(0, 200)

        #         scroll(-50.2, -100.5)
        # """
        #         system_msg = system_msg.replace(scrolling_prompt, '')

        focus_prompt = """
focus(bid: str)
    Examples:
        focus('b455')"""
        system_msg = system_msg.replace(focus_prompt, '')

        select_option_prompt = """
select_option(bid: str, options: str | list[str])
    Examples:
        select_option('48', 'blue')

        select_option('48', ['red', 'green', 'blue'])"""
        system_msg = system_msg.replace(select_option_prompt, '')

        hover_prompt = """
hover(bid: str)
    Examples:
        hover('b8')"""
        system_msg = system_msg.replace(hover_prompt, '')
        # logger.info(system_msg)
        messages = []
        messages.append({'role': 'system', 'content': system_msg})
        messages.append({'role': 'user', 'content': prompt})

        # with open("/home/demo/jinyu/prompts/last_prompt.txt", "w") as f:
        #     f.write(prompt)

        def parser(text):
            try:
                ans_dict = parse_func(text)
            except ParseError as e:
                return None, False, str(e)
            return ans_dict, True, ''

        try:
            # ans_dict = self.retry(messages, parser, n_retry=self.max_retry)
            ans_dict = self.retry(
                self.truncate_messages(messages, MAX_TOKENS - OUTPUT_BUFFER),
                parser,
                n_retry=self.max_retry,
                override_llm=override_llm,
            )  # added
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

        return ans_dict

    def encoder(self, main_prompt):
        prompt = main_prompt.get_encoder_prompt()
        # logger.info(prompt)
        ans_dict = self.get_llm_output(
            prompt, main_prompt._parse_encoder_answer, ['state', 'progress']
        )

        think = ans_dict.get('think')
        replan = ans_dict['progress'] in ['finished', 'failed', 'not-sure']

        return ans_dict['state'], ans_dict['progress'], replan, think

    def policy(self, main_prompt):
        prompt = main_prompt.get_policy_prompt()
        # logger.info(main_prompt.states)
        # logger.info(prompt)

        temp = self.temperature
        self.temperature = 1.0
        ans_dict = self.get_llm_output(
            prompt, main_prompt._parse_policy_answer, ['strategy', 'summary']
        )
        self.temperature = temp

        return ans_dict['strategy'], ans_dict['summary']

    def dynamics(self, main_prompt):
        prompt = main_prompt.get_dynamics_prompt()
        # logger.info(prompt)

        ans_dict = self.get_llm_output(
            prompt, main_prompt._parse_dynamics_answer, ['next_state', 'progress']
        )

        is_terminal = ans_dict['progress'] == 'goal-reached'
        return ans_dict['next_state'], ans_dict['progress'], is_terminal

    def action_reward(self, main_prompt):
        prompt = main_prompt.get_action_reward_prompt()
        # logger.info(prompt)

        ans_dict = self.get_llm_output(
            prompt, main_prompt._parse_action_reward_answer, ['response']
        )

        think = ans_dict.get('think')
        response = ans_dict['response']
        reward = (
            -1
            if response == 'away-from-the-goal'
            else 1
            if response == 'towards-the-goal'
            else 0
        )
        return reward, think, response

    def effectuator(self, main_prompt):
        prompt = main_prompt.get_effectuator_prompt()
        # logger.info(prompt)

        ans_dict = self.get_llm_output(
            prompt,
            main_prompt._parse_effectuator_answer,
            ['action', 'explanation', 'summary'],
            override_llm=False,
        )

        return ans_dict['action'], ans_dict['explanation'], ans_dict['summary']

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
        """

        # Set default first action
        # if DEFAULT_BROWSER is not None and len(self.actions) == 0:
        #     time.sleep(4)
        #     action = "goto('{}')".format(DEFAULT_BROWSER)
        #     self.actions.append(action)
        #     return BrowseInteractiveAction(
        #         browser_actions=action, thought='Open default browser'
        #     )
        actions = self.actions
        # if DEFAULT_BROWSER is not None:
        #     actions = actions[1:]

        goal = env_state.get_current_user_intent()
        if goal is None:
            goal = env_state.inputs['task']
        self.goal = goal

        # messages: List[str] = []
        prev_actions: List[str] = []
        cur_axtree_txt = ''
        error_prefix = ''
        last_obs = None
        last_action = None

        if EVAL_MODE and len(env_state.history) == 1:
            # for webarena and miniwob++ eval, we need to retrieve the initial observation already in browser env
            # initialize and retrieve the first observation by issuing an noop OP
            # For non-benchmark browsing, the browser env starts with a blank page, and the agent is expected to first navigate to desired websites
            return BrowseInteractiveAction(browser_actions='noop()')

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
                return AgentFinishAction(outputs={'content': prev_action.content})

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
            return MessageAction(last_action.browsergym_send_msg_to_user)

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
                cur_axtree_txt = (
                    f"URL {last_obs.url}\n"
                    f"Scroll Position: {last_obs.scroll_position['scrollTop']}, "
                    f"Window Height: {last_obs.scroll_position['windowHeight']}, "
                    f"Webpage Height: {last_obs.scroll_position['documentHeight']}, "
                    f"Remaining Pixels: {last_obs.scroll_position['remainingPixels']}\n"
                ) + cur_axtree_txt
                logger.info(last_obs.scroll_position)
            except Exception as e:
                logger.error(
                    'Error when trying to process the accessibility tree: %s', e
                )
                return MessageAction('Error encountered when browsing.')

        if error_prefix:
            self.error_accumulator += 1
            if self.error_accumulator > 20:
                return MessageAction('Too many errors encountered. Task failed.')

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

        current_obs = {
            'axtree_txt': clean_axtree_txt,
            'raw_axtree_txt': cur_axtree_txt,
            # 'axtree_txt': "AXSTART "+cur_axtree_txt+" AXEND",
            'last_action_error': error_prefix,
            'goal': goal,
        }
        self.obs_history.append(current_obs)
        main_prompt = MyMainPrompt(
            obs_history=self.obs_history,
            states=self.states,
            strategies=self.strategies,
            explanations=self.explanations,
            actions=self.actions,
            active_strategy=self.active_strategy,
        )

        state, status, replan, think = self.encoder(main_prompt)
        self.full_output = ''
        self.full_output_dict = {}

        logger.info(f'*State*: {state}')
        logger.info(f'*Replan Reasoning*: {think}')
        logger.info(f'*Replan Status*: {status}')

        self.full_output_dict['obs'] = current_obs
        self.full_output_dict['state'] = state
        self.full_output_dict['replan_reasoning'] = think
        self.full_output_dict['replan_status'] = status

        self.full_output += f'*State*: {state}\n'
        self.full_output += f'*Replan Reasoning*: {think}\n'
        self.full_output += f'*Replan Status*: {status}\n'

        # replan = True
        if len(actions) > 1 and actions[-1] == actions[-2]:
            logger.info('*Action Repeat, Force Replan*')
            replan = True
        elif self.active_strategy is None or self.active_strategy_turns >= 3:
            replan = True

        if replan:
            strategy = self.planning_search(state)
            self.strategies.append(strategy)
            self.active_strategy = strategy
            self.active_strategy_turns = 0
        else:
            self.strategies.append(None)
            self.active_strategy_turns += 1
        logger.info(f'*Active Strategy*: {self.active_strategy}')
        self.full_output += f'*Active Strategy*: {self.active_strategy}\n'

        self.full_output_dict['replan'] = replan
        self.full_output_dict['active_strategy'] = self.active_strategy

        self.states.append(state)
        main_prompt = MyMainPrompt(
            obs_history=self.obs_history,
            states=self.states,
            strategies=self.strategies,
            explanations=self.explanations,
            actions=self.actions,
            active_strategy=self.active_strategy,
            action_space=self.action_space,
        )

        action, explanation, summary = self.effectuator(main_prompt)
        if len(actions) >= 10 and (
            (
                actions[-1]
                == actions[-2]
                == actions[-3]
                == actions[-4]
                == actions[-5]
                == actions[-6]
                == actions[-7]
                == actions[-8]
            )
            or (len(set(actions[-8:])) <= 2)
        ):
            action = "send_msg_to_user('It seems I am stuck. Could you help me out?')"

        logger.info(f'*Action*: {action}')
        # self.full_output += f'*Action*: {action}\n'
        logger.info(f'*Summary*: {summary}')
        self.full_output += f'*Action*: {explanation}\n'
        self.full_output += f'*Summary*: {summary}\n'
        self.full_output_dict['explanation'] = explanation
        self.full_output_dict['action'] = action
        self.full_output_dict['summary'] = summary

        llm_output_logger.info(self.full_output)
        self.full_output_dict['full_output'] = self.full_output

        self.full_output_json = json.dumps(self.full_output_dict)

        time.sleep(random.random() * 5)

        self.actions.append(action)
        self.explanations.append(explanation)
        # return self.parse_response(action, self.full_output)
        return self.parse_response(action, self.full_output_json)

    def planning_search(self, state):
        # Run MCTS Search
        class MCTSNode:
            id_iter = itertools.count()

            @classmethod
            def reset_id(cls):
                cls.id_iter = itertools.count()

            def __init__(
                self,
                state=None,
                status=None,
                action=None,
                reward_think=None,
                reward_answer=None,
                fast_reward=0,
                parent=None,
                is_terminal=False,
            ):
                self.state = state
                self.status = status
                self.action = action
                self.reward_think = reward_think
                self.reward_answer = reward_answer
                self.fast_reward = self.reward = fast_reward
                self.cum_rewards = []
                self.parent = parent
                self.children: List[MCTSNode] = []
                self.is_terminal = is_terminal
                if parent is None:
                    self.depth = 0
                else:
                    self.depth = parent.depth + 1

            @property
            def Q(self):
                if len(self.cum_rewards) == 0:
                    return 0
                return np.mean(self.cum_rewards)

        def _expand(node, path):
            new_states = [n.state for n in path[:] if n.state is not None]
            new_actions = [n.action for n in path[:] if n.action is not None]
            # actions = self.actions
            # if DEFAULT_BROWSER is not None:
            #     actions = actions[1:]
            if node.state is None:
                # print(self.states + new_states)
                # print(self.actions + new_actions)
                main_prompt = MyMainPrompt(
                    obs_history=self.obs_history,
                    states=self.states + new_states,
                    strategies=self.strategies + new_actions,
                    explanations=self.explanations,
                    actions=self.actions,
                )
                node.state, node.status, node.is_terminal = self.dynamics(main_prompt)
                logger.info(f'*Expanded Strategy*: {node.action}')
                logger.info(f'*Next State*: {node.state}')
                logger.info(f'*Status*: {node.status}')

                self.full_output += f'*Expanded Strategy*: {node.action}\n'
                self.full_output += f'*Next State*: {node.state}\n'
                self.full_output += f'*Status*: {node.status}\n'

                new_states.append(node.state)

                # Here is a chance to reset the node reward using things like state transition certainty
                # or state-conditional critic (value function)
                # As a default we just keep using the fast reward
                node.reward = node.fast_reward

                # main_prompt = my_prompting.MyMainPrompt(
                #     obs_history=self.obs_history,
                #     states=self.states + new_states,
                #     actions=actions + new_actions
                # )
                # node.reward, node.is_terminal, node.reward_details = self.critic(main_prompt)
                # TODO (DONE) : Figure out numerical reward logic
            if not node.is_terminal and not _is_terminal_with_depth_limit(node):
                children = []
                # Sample an action space:
                # action_space = {}
                n_actions = 3
                # def sample_action(obs_history, states, strategies, actions, policy):
                #     main_prompt = MyMainPrompt(
                #         obs_history=self.obs_history,
                #         states=self.states + new_states,
                #         strategies=self.strategies + new_actions,
                #         # actions=actions,
                #         actions=self.explanations,
                #     )
                #     strategy = self.policy(main_prompt)
                #     return strategy

                # Create a pool of worker processes
                with mp_dummy.Pool(processes=n_actions) as pool:
                    # Use starmap to pass multiple arguments to the function
                    arguments = [
                        (
                            self.obs_history,
                            self.states + new_states,
                            self.strategies + new_actions,
                            self.explanations,
                            self.actions,
                            self.policy,
                        )
                    ] * n_actions
                    sampled_actions = pool.starmap(sample_action, arguments)

                sampled_actions = list(set(sampled_actions))

                with mp_dummy.Pool(processes=len(sampled_actions)) as pool:
                    arguments = [
                        (
                            self.obs_history,
                            self.states + new_states,
                            self.strategies + new_actions + [action],
                            self.explanations,
                            self.actions,
                            self.action_reward,
                        )
                        for action in sampled_actions
                    ]
                    sampled_action_rewards = pool.starmap(
                        sample_action_reward, arguments
                    )

                # action_space = {action: (fast_reward, think) for action, (fast_reward, think) in}
                for (action, summary), (fast_reward, think, response) in zip(
                    sampled_actions, sampled_action_rewards
                ):
                    logger.info(f'*Strategy Candidate*: {action}')
                    logger.info(f'*Summary*: {summary}')
                    logger.info(f'*Fast Reward Reasoning*: {think}')
                    logger.info(f'*Fast Reward*: {fast_reward}')

                    self.full_output += f'*Strategy Candidate*: {action}\n'
                    self.full_output += f'*Summary*: {summary}\n'
                    self.full_output += f'*Fast Reward Reasoning*: {think}\n'
                    self.full_output += f'*Fast Reward*: {fast_reward}\n'

                    # child = MCTSNode(
                    #     state=None, action=action, parent=node, fast_reward=fast_reward
                    # )
                    child = MCTSNode(
                        state=None,
                        status=None,
                        action=action,
                        reward_think=think,
                        reward_answer=response,
                        fast_reward=fast_reward,
                        parent=node,
                    )
                    # child.action_dict = action_dict
                    # child.fast_reward_dict = fast_reward_dict
                    children.append(child)

                node.children = children

        w_exp = 1
        depth_limit = 3

        def _uct(node):
            uct_term = np.sqrt(
                np.log(len(node.parent.cum_rewards)) / max(1, len(node.cum_rewards))
            )
            logger.info(f'{node.Q} {round(uct_term, 3)}')
            self.full_output += f'{node.Q} {round(uct_term, 3)}\n'
            return node.Q + w_exp * uct_term

        def _is_terminal_with_depth_limit(node):
            return node.is_terminal or node.depth >= depth_limit

        # _uct = lambda node: node.Q + w_exp * np.sqrt(np.log(len(node.parent.cum_rewards)) / max(1, len(node.cum_rewards)))
        # _is_terminal_with_depth_limit = (
        #     lambda node: node.is_terminal or node.depth >= depth_limit
        # )

        N = 3
        root = MCTSNode(state=state, action=None, parent=None, status='init')
        # for n in trange(N, desc='MCTS iteration', leave=True):
        for n in range(N):
            logger.info(f'MCTS iter {n}')
            self.full_output += f'MCTS iter {n}\n'
            # select
            node = root
            path = []
            finished = False
            while not finished:
                path.append(node)
                if (
                    node.children is None
                    or len(node.children) == 0
                    or _is_terminal_with_depth_limit(node)
                ):
                    finished = True
                else:
                    # uct select with fast reward
                    node = max(node.children, key=_uct)

            node = path[-1]
            if not _is_terminal_with_depth_limit(node):
                # expand
                _expand(node, path)
                # simulate
                finished = False
                while not finished:
                    if node.state is None:
                        _expand(node, path)
                    if _is_terminal_with_depth_limit(node) or len(node.children) == 0:
                        finished = True
                    else:
                        fast_rewards = [child.fast_reward for child in node.children]
                        # TODO (DONE): Simulate choice
                        node = node.children[np.argmax(fast_rewards)]
                        path.append(node)
            # backpropagate
            rewards = []
            cum_reward = -math.inf
            for node in reversed(path):
                rewards.append(node.reward)
                cum_reward = np.sum(rewards[::-1])
                node.cum_rewards.append(cum_reward)

        # max reward output strategy
        # dfs on max reward
        def _dfs_max_reward(path):
            cur = path[-1]
            if cur.is_terminal:
                return sum([node.reward for node in path[1:]]), path
            if cur.children is None:
                return -math.inf, path
            visited_children = [x for x in cur.children if x.state is not None]
            if len(visited_children) == 0:
                return -math.inf, path
            return max(
                (_dfs_max_reward(path + [child]) for child in visited_children),
                key=lambda x: x[0],
            )

        output_cum_reward, output_iter = _dfs_max_reward([root])
        action = output_iter[1].action
        logger.info(f'*Selected Strategy*: {action}')
        self.full_output += f'*Selected Strategy*: {action}\n'

        # def __init__(
        #         self,
        #         state=None,
        #         status=None,
        #         action=None,
        #         reward_think=None,
        #         reward_answer=None,
        #         fast_reward=0,
        #         parent=None,
        #         is_terminal=False,

        # Iterate through the tree to build the full_output_dict
        def node_to_dict(node):
            return {
                'state': node.state,
                'status': node.status,
                'strategy': node.action,
                'critique': node.reward_think,
                'evaluation': node.reward_answer,
                'reward': node.reward,
                'q_value': node.Q,
                'children': [node_to_dict(child) for child in node.children],
            }

        self.full_output_dict['mcts_tree'] = node_to_dict(root)

        # print('Selected Action:', action)

        return action

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
