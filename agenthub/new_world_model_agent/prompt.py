import abc
import platform

from browsergym.core.action.base import AbstractActionSet
from browsergym.core.action.highlevel import HighLevelActionSet

from .utils import (
    ParseError,
    parse_html_tags_raise,
)


class PromptElement:
    """Base class for all prompt elements. Prompt elements can be hidden.

    Prompt elements are used to build the prompt. Use flags to control which
    prompt elements are visible. We use class attributes as a convenient way
    to implement static prompts, but feel free to override them with instance
    attributes or @property decorator."""

    _prompt = ''
    _abstract_ex = ''
    _concrete_ex = ''

    def __init__(self, visible: bool = True) -> None:
        """Prompt element that can be hidden.

        Parameters
        ----------
        visible : bool, optional
            Whether the prompt element should be visible, by default True. Can
            be a callable that returns a bool. This is useful when a specific
            flag changes during a shrink iteration.
        """
        self._visible = visible

    @property
    def prompt(self):
        """Avoid overriding this method. Override _prompt instead."""
        return self._hide(self._prompt)

    @property
    def abstract_ex(self):
        """Useful when this prompt element is requesting an answer from the llm.
        Provide an abstract example of the answer here. See Memory for an
        example.

        Avoid overriding this method. Override _abstract_ex instead
        """
        return self._hide(self._abstract_ex)

    @property
    def concrete_ex(self):
        """Useful when this prompt element is requesting an answer from the llm.
        Provide a concrete example of the answer here. See Memory for an
        example.

        Avoid overriding this method. Override _concrete_ex instead
        """
        return self._hide(self._concrete_ex)

    @property
    def is_visible(self):
        """Handle the case where visible is a callable."""
        visible = self._visible
        if callable(visible):
            visible = visible()
        return visible

    def _hide(self, value):
        """Return value if visible is True, else return empty string."""
        if self.is_visible:
            return value
        else:
            return ''

    def _parse_answer(self, text_answer) -> dict:
        if self.is_visible:
            return self._parse_answer(text_answer)
        else:
            return {}


class Shrinkable(PromptElement, abc.ABC):
    @abc.abstractmethod
    def shrink(self) -> None:
        """Implement shrinking of this prompt element.

        You need to recursively call all shrinkable elements that are part of
        this prompt. You can also implement a shriking startegy for this prompt.
        Shrinking is can be called multiple times to progressively shrink the
        prompt until it fits max_tokens. Default max shrink iterations is 20.
        """
        pass


class Trunkater(Shrinkable):
    def __init__(self, visible, shrink_speed=0.3, start_trunkate_iteration=10):
        super().__init__(visible=visible)
        self.shrink_speed = shrink_speed
        self.start_trunkate_iteration = start_trunkate_iteration
        self.shrink_calls = 0
        self.deleted_lines = 0

    def shrink(self) -> None:
        if self.is_visible and self.shrink_calls >= self.start_trunkate_iteration:
            # remove the fraction of _prompt
            lines = self._prompt.splitlines()
            new_line_count = int(len(lines) * (1 - self.shrink_speed))
            self.deleted_lines += len(lines) - new_line_count
            self._prompt = '\n'.join(lines[:new_line_count])
            self._prompt += (
                f'\n... Deleted {self.deleted_lines} lines to reduce prompt size.'
            )

        self.shrink_calls += 1


class HTML(Trunkater):
    def __init__(self, html, visible: bool = True, prefix='') -> None:
        super().__init__(visible=visible, start_trunkate_iteration=5)
        self._prompt = f'\n{prefix}HTML:\n{html}\n'


class AXTree(Trunkater):
    def __init__(
        self, ax_tree, visible: bool = True, coord_type=None, prefix=''
    ) -> None:
        super().__init__(visible=visible, start_trunkate_iteration=10)
        if coord_type == 'center':
            coord_note = """\
Note: center coordinates are provided in parenthesis and are
  relative to the top left corner of the page.\n\n"""
        elif coord_type == 'box':
            coord_note = """\
Note: bounding box of each object are provided in parenthesis and are
  relative to the top left corner of the page.\n\n"""
        else:
            coord_note = ''
        self._prompt = f'\n{prefix}AXTree (you may only interact with elements in this tree):\n{coord_note}{ax_tree}\n'


class Error(PromptElement):
    def __init__(self, error, visible: bool = True, prefix='') -> None:
        super().__init__(visible=visible)
        self._prompt = f'\n{prefix}Error from previous action:\n{error}\n'


class MacNote(PromptElement):
    def __init__(self) -> None:
        super().__init__(visible=platform.system() == 'Darwin')
        self._prompt = '\nNote: you are on mac so you should use Meta instead of Control for Control+C etc.\n'


class GoalInstructions(PromptElement):
    def __init__(self, goal, visible: bool = True) -> None:
        super().__init__(visible)
        self._prompt = f"""\
# Instructions
Review the current state of the page and all other information to find the best possible next action to accomplish your goal. Your answer will be interpreted and executed by a program, make sure to follow the formatting instructions.

## Goal:
{goal}
"""


class ChatInstructions(PromptElement):
    def __init__(self, chat_messages, visible: bool = True) -> None:
        super().__init__(visible)
        self._prompt = """\
# Instructions

You are a UI Assistant, your goal is to help the user perform tasks using a web browser. You can communicate with the user via a chat, in which the user gives you instructions and in which you can send back messages. You have access to a web browser that both you and the user can see, and with which only you can interact via specific commands.

Review the instructions from the user, the current state of the page and all other information to find the best possible next action to accomplish your goal. Your answer will be interpreted and executed by a program, make sure to follow the formatting instructions.

## Chat messages:

"""
        self._prompt += '\n'.join(
            [
                f"""\
 - [{msg['role']}] {msg['message']}"""
                for msg in chat_messages
            ]
        )


class SystemPrompt(PromptElement):
    _prompt = """\
You are an agent trying to solve a web task based on the content of the page and a user instructions. You can interact with the page and explore. Each time you submit an action it will be sent to the browser and you will receive a new page."""


def _get_my_action_space() -> AbstractActionSet:
    # Assume action space type is bid
    action_space = 'bid'
    action_subsets = ['chat', 'nav', 'bid']

    action_space = HighLevelActionSet(
        subsets=action_subsets,
        multiaction=False,
        strict=False,
        demo_mode=True,
    )

    return action_space


class MyActionSpace(PromptElement):
    def __init__(self) -> None:
        super().__init__()
        # self.flags = flags
        self.action_space = _get_my_action_space()

        # self._prompt = f"# Action space:\n{self.action_space.describe()}{MacNote().prompt}\n"
        self._prompt = f'# Action space:\n{self.action_space.describe(with_long_description=False, with_examples=True)}\n'
        self._abstract_ex = f"""
<action>
{self.action_space.example_action(abstract=True)}
</action>
"""
        self._concrete_ex = f"""
<action>
{self.action_space.example_action(abstract=False)}
</action>
"""

    def _parse_answer(self, text_answer):
        ans_dict = parse_html_tags_raise(
            text_answer, keys=['action'], merge_multiple=True
        )

        try:
            # just check if action can be mapped to python code but keep action as is
            # the environment will be responsible for mapping it to python
            self.action_space.to_python_code(ans_dict['action'])
        except Exception as e:
            raise ParseError(
                f'Error while parsing action\n: {e}\n'
                'Make sure your answer is restricted to the allowed actions.'
            )

        return ans_dict


class MyMainPrompt(PromptElement):
    def __init__(
        self,
        obs_history,
        states,
        strategies,
        explanations,
        actions,
        active_strategy=None,
        action_space=None,
    ):
        super().__init__()
        # Include all states + actions from the history. Ignore obs_history for now
        self.obs_history = obs_history
        self.states = states
        self.strategies = strategies
        self.explanations = explanations
        self.actions = actions
        self.active_strategy = active_strategy
        self.action_space = action_space

        self.history = self.get_history(
            obs_history, states, strategies, explanations, actions
        )
        # self.instructions = self.get_goal_instruction(obs_history[-1]["goal"])

        # Several modes
        # 1. len(obs_history) == len(states) + 1 == len(actions) + 1: encoding, use just the obs
        # 2. len(obs_history) == len(states) == len(actions) + 1: policy, use obs + state
        # 3. len(obs_history) == len(states) == len(actions): first forward dynamics, use obs + state + action
        # 4. len(states) == len(actions) > len(obs_history): other forward dynamics, use state + action
        # 4. len(states) == len(actions) > len(obs_history): action value, use state + action
        # 4. len(states) == len(actions) + 1 > len(obs_history): critic, use state
        # 5. len(states) == len(actions) + 1 > len(obs_history): rollout policy, use state

        if len(obs_history) == len(states) + 1 and len(states) == len(strategies):
            # encoding, use just the obs
            self.obs = self.get_obs(obs_history[-1])
            window = min(3, len(states))
            self.history = self.get_history(
                obs_history,
                states[-window:],
                strategies[-window:],
                explanations[-window:],
                actions[-window:],
            )
            # self.history = self.get_history(
            #     obs_history, states[:], strategies[:], explanations[:], actions[:]
            # )
        elif (
            len(obs_history) == len(states)
            and len(states) == len(strategies) + 1
            and len(strategies) == len(actions)
        ):
            # strategy, use the obs and the state
            # also the first step in dynamics
            self.obs = self.get_obs_state(obs_history[-1], states[-1])
            self.history = self.get_history(
                obs_history, states[:-1], strategies[:], explanations[:], actions[:]
            )
        elif (
            len(obs_history) == len(states)
            and len(states) == len(strategies)
            and len(strategies) == len(actions) + 1
        ):
            # policy, use obs, state, and strategy
            self.obs = self.get_obs_state_strat(
                obs_history[-1], states[-1], active_strategy
            )
            self.history = self.get_history(
                obs_history, states[:-1], strategies[:-1], explanations[:], actions[:]
            )
        elif (
            len(obs_history) <= len(states)
            and len(states) == len(strategies)
            and len(strategies) >= len(actions) + 1
        ):
            # forward dynamics, use state + strat
            # action value, use state + strat
            self.obs = self.get_state_strat(states[-1], strategies[-1])
        elif len(obs_history) < len(states) and len(states) == len(strategies) + 1:
            # rollout strategy, use state
            self.obs = self.get_state(states[-1])
            self.history = self.get_history(
                obs_history, states[:-1], strategies[:], explanations[:], actions[:]
            )

        # self.action_space = MyActionSpace()

    #     def get_goal_instruction(self, goal):
    #         prompt = f"""\
    # # Instructions
    # Review the current state of the page and all other information to find the best
    # possible next action to accomplish your goal. Your answer will be interpreted
    # and executed by a program, make sure to follow the formatting instructions.

    # ## Goal:
    # {goal}
    # """
    #         return prompt

    def get_history(self, obs_history, states, strategies, explanations, actions):
        # assert len(obs_history) == len(states) or len(obs_history) == len(states) + 1
        # assert len(obs_history) == len(actions) + 1
        # assert len(states) == len(actions) or len(states) == len(actions) + 1
        assert len(states) == len(strategies) or len(states) == len(strategies) + 1

        self.history_steps = []

        for i in range(1, len(states) + 1):
            history_step = self.get_history_step(
                None,
                states[i - 1],
                strategies[i - 1]
                if (len(strategies) > 0 and i <= len(strategies))
                else None,
                explanations[i - 1]
                if (len(explanations) > 0 and i <= len(explanations))
                else None,
                actions[i - 1] if (len(actions) > 0 and i <= len(actions)) else None,
            )

            self.history_steps.append(history_step)

        prompts = ['\n# History of interaction with the task:\n']
        for i, step in enumerate(self.history_steps):
            num_steps_away = len(self.history_steps) - i
            # prompts.append(f'## Step {i}')
            prompts.append(f'## {num_steps_away} Steps Earlier')
            prompts.append(step)
        return '\n'.join(prompts) + '\n'

    def get_history_step(self, current_obs, state, strategy, explanation, action):
        if current_obs is not None:
            self.ax_tree = AXTree(
                current_obs['axtree_txt'],
                visible=True,
                coord_type=False,
                prefix='\n#### Accessibility tree:\n',
            )
            self.error = Error(
                current_obs['last_action_error'],
                visible=current_obs['last_action_error'],
                prefix='#### ',
            )
            # # self.observation = f"{self.ax_tree.prompt}{self.error.prompt}"
            self.observation = f'{self.error.prompt}{self.ax_tree.prompt}'
        self.state = state
        self.strategy = strategy
        self.explanation = explanation
        self.action = action

        prompt = ''
        if current_obs is not None:
            prompt += f'\n### Observation:\n{self.observation}\n\n'
        else:
            prompt += f'\n### State:\n{self.state}\n'

        if strategy is not None:
            prompt += f'\n### Strategy:\n{self.strategy}\n'

        if action is not None:
            prompt += f'\n### Action:\n{self.explanation}\n{self.action}\n'

        return prompt

    def get_obs(self, obs):
        # self.html = HTML(obs["pruned_html"],
        #                  visible=True,
        #                  prefix="## ")
        self.ax_tree = AXTree(
            obs['axtree_txt'],
            visible=True,
            coord_type=False,
            prefix='## ',
        )
        self.error = Error(
            obs['last_action_error'],
            visible=obs['last_action_error'],
            prefix='## ',
        )
        # self.screenshot = obs["screenshot"]

        # return f"\n# Observation of current step:\n{self.error.prompt}{self.ax_tree.prompt}\n"
        # return f"\n# Observation of current step:\n{self.html.prompt}{self.ax_tree.prompt}{self.error.prompt}\n\n"
        return f'\n# Observation of current step:\nAXSTART{self.ax_tree.prompt}AXEND{self.error.prompt}\n\n'

    def get_obs_state(self, obs, state):
        return self.get_obs(obs) + f'\n## Current State:\n{state}\n'

    def get_obs_state_strat(self, obs, state, strategy):
        return self.get_obs_state(obs, state) + f'\n## Current Strategy:\n{strategy}\n'

    def get_state(self, state):
        return f'\n## Current State:\n{state}\n\n'

    def get_state_strat(self, state, strategy):
        return self.get_state(state) + f'\n## Current Strategy:\n{strategy}\n'

    # def add_screenshot(self, prompt):
    #     if isinstance(prompt, str):
    #         prompt = [{'type': 'text', 'text': prompt}]
    #     img_url = BrowserEnv.image_to_jpg_base64_url(self.screenshot)
    #     prompt.append({'type': 'image_url', 'image_url': img_url})

    #     return prompt

    def get_effectuator_prompt(self) -> str:
        prompt = f"""\
{self.history}\
{self.obs}\
"""

        prompt += """

# Abstract Example

Here is an abstract version of the answer with description of the content of each tag. Make sure you follow this structure, but replace the content with your answer:
<explanation>
Describe what the action to be taken is trying to do using a single concise sentence. Break down the active strategy into individual, manageable actions. Avoid long, complex search terms. Focus on the single action. Use first-person perspective like "I am doing something." If you encounter trouble using the search button, try hitting enter on the search box instead. If you fail to click on something, try scrolling down by 500 pixels first. If an element is no longer visible, try scrolling up by 500 pixels. Use clear and simple language to describe your action.
</explanation>

<action>
Based on the current observation, state, active strategy, and action history, select one single action to be executed. Use only one action at a time. You must not enclose bid inputs in [brackets]. Interact only with elements in the current step observation. Your response will be executed as a Python function call, so ensure it adheres to the format and argument data type specifications defined in the action space.
</action>

<summary>
Based on the explanation, summarize the explanation down to with 5 words.
</summary>
"""

        prompt += """

# Concrete Example

Here is a concrete example of how to format your answer. Make sure to follow the template by wrapping with proper html starting and closing tags:
<explanation>
I am filling out the textbox for Date with 'example with "quotes"'
</explanation>

<action>
fill('32-12', 'example with "quotes"')
</action>

<summary>
Fill textbox for date.
</summary>
"""

        # prompt = self.add_screenshot(prompt)

        return prompt

    def get_encoder_prompt(self) -> str:
        prompt = f"""\
{self.history}\
{self.obs}\
## Active Strategy:
{self.active_strategy}
"""

        prompt += """

# Abstract Example

Here is an abstract version of the answer with description of the content of each tag. Make sure you follow this structure, but replace the content with your answer:
<state>
Summarize the current state of the webpage observation, focusing on the most recent action you took and any errors encountered. Note any dialogs, progress indicators, or significant changes such as items in your cart or sites visited. Describe the impact of your previous action on the webpage, including any new interactive elements. Include any inferred information that may help achieve the goal. Information from steps earlier are for reference only. Focus on objective description of the current observation and any inferences you can draw from it. Report any error messages displayed. Do not include your next planned actions; focus solely on providing an objective summary.
</state>\

<progress>
Evaluate your most recent action, the current state of the task, and your active strategy. Categorize the situation into one of four categories based on the progress of your strategy:
1. "finished" - Your strategy has been successfully executed, and you will plan the next step.
2. "in-progress" - Your strategy is still ongoing, and further actions are required.
3. "not-sure" - It's unclear whether your strategy has been executed successfully, and you need to reassess your plan.
4. "failed" - Your strategy was unsuccessful, and you need to develop a new plan.
Be cautious when assigning the "in-progress" label. If uncertain, choose "not-sure" instead.
</progress>
"""

        prompt += """

# Concrete Example

Here is a concrete example of how to format your answer. Make sure to follow the template by wrapping with proper html starting and closing tags:
<state>
The previous action resulted in a timeout error, indicating no changes were made to the page. Thus far, I have visited ABC.com and DEF.com, discovering information G and H, respectively. The current page contains a dialog prompting whether to add protection, offering coverage options and the choices "Add Protection" or "No Thanks". A link indicates "1 item in cart", revealing a cellphone in the cart with a subtotal of $345. I searched for a 5-night hotel stay, but results only showed availability for a 6-night stay, suggesting a 5-night stay is unavailable. The page displays:
- An empty textbox in the middle labeled "Date", indicating it is likely for date input.
- A button below labeled "Submit" positioned below the textbox, suggesting it submits the date entered in the textbox.
Additionally, there are:
- A notification below the button displaying "Error: Invalid date format" when attempting to submit the date.
- A dropdown menu labeled "Room Type" containing options "Single", "Double", and "Suite".
- A section showing "Total Price: $0.00", implying the total cost updates dynamically based on selections.
The page did not display any new errors after the latest action, apart from the timeout issue.
I clicked the "Submit" button on the booking form, but no confirmation message appeared, and the page did not change. There is no indication whether the submission was successful or not. I need to reassess the page for any subtle changes or possible errors.
</state>\

<progress>
not-sure
</progress>
"""

        #         The previous action resulted in a timeout error, indicating no changes were made to the page. Thus far, I have visited ABC.com and DEF.com, discovering information G and H, respectively. The current page contains a dialog with id 789 prompting whether to add protection, offering coverage options and the choices "Add Protection" or "No Thanks". A link with id 234 indicates "1 item in cart", revealing a cellphone in the cart with a subtotal of $345. I searched for a 5-night hotel stay, but results only showed availability for a 6-night stay, suggesting a 5-night stay is unavailable. The page displays:
        # - An empty textbox with id 123 labeled "Date", indicating it is likely for date input.
        # - A button with id 456 labeled "Submit" positioned below textbox 123, suggesting it submits the date entered in the textbox.
        # Additionally, there are:
        # - A notification with id 101 displaying "Error: Invalid date format" when attempting to submit the date.
        # - A dropdown menu with id 102 labeled "Room Type" containing options "Single", "Double", and "Suite".
        # - A section with id 103 showing "Total Price: $0.00", implying the total cost updates dynamically based on selections.
        # The page did not display any new errors after the latest action, apart from the timeout issue.
        # I clicked the "Submit" button on the booking form, but no confirmation message appeared, and the page did not change. There is no indication whether the submission was successful or not. I need to reassess the page for any subtle changes or possible errors.

        # foo = 'Include details such as accessibility tree id when describing elements on the page.'

        # prompt = self.add_screenshot(prompt)

        return prompt

    def get_policy_prompt(self) -> str:
        prompt = f"""\
{self.history}\
{self.obs}\
"""

        prompt += """

# Abstract Example

Here is an abstract version of the answer with description of the content of each tag. Make sure you follow this structure, but replace the content with your answer:
<strategy>
Given that previous actions have been completed and the environment has transitioned to the current inferred state, describe the next action to achieve the goal. Break down the goal into clear, manageable steps. Avoid using phrases such as "To accomplish the goal," "I will," "To proceed," or "Assume the previous strategies have been carried out." Refrain from mentioning specific element IDs as they may change during execution. Limit your response to one sentence and include any details that help select the correct action. Be creative and propose novel methods to achieve the goal. Avoid creating accounts without user permission or providing personal information.
</strategy>

<summary>
Given the strategy/action that you just came up with, summarize the strategy/action down to within 5 words in 5 DIFFERENT ways, and choose ONE of them to respond.
</summary>

<explanation>
Use first person to describe what you plan to do next. Begin with the phrase "My next step is".
</explanation>
"""

        prompt += """

# Concrete Example

Here is a concrete example of how to format your answer. Make sure to follow the template by wrapping with proper html starting and closing tags:
<strategy>
Click through the form fields to explore available options and ensure all mandatory fields are completed.
</strategy>

<summary>
Exploring, ensure mandatory fields completed.
</summary>

<explanation>
I am clicking through the form fields to explore available options and ensure all mandatory fields are completed.
</explanation>
"""

        # prompt = self.add_screenshot(prompt)

        return prompt

    def get_dynamics_prompt(self) -> str:
        if len(self.obs_history) == len(self.states):
            self.obs = self.get_obs_state_strat(
                self.obs_history[-1], self.states[-1], self.strategies[-1]
            )
        prompt = f"""\
{self.obs}\
"""

        prompt += """
# Abstract Example

Here is an abstract version of the answer with description of the content of each tag. Make sure you follow this structure, but replace the content with your
answer:
<next_state>
Assume the environment is at the current inferred state and your proposed strategy has been applied. Predict the new state of the webpage after executing each part of the proposed strategy. Describe the expected changes in page content, any new information relevant to your goal, and how interactive elements might change. Pay close attention to how the details of elements will be altered. Identify the new elements you can interact with on the updated webpage.
</next_state>\

<progress>
Evaluate the previous and current states of the browser environment. Classify your status into one of three categories based on your progress towards the goal:

"in-progress" - You are still working towards achieving the goal.
"not-sure" - It's unclear if the goal has been achieved.
"goal-reached" - You have successfully completed the goal.
</progress>
"""

        prompt += """
# Concrete Example

Here is a concrete example of how to format your answer. Make sure to follow the template by wrapping with proper html starting and closing tags:
<next_state>
A dropdown menu appears below the textbox, listing various predefined options based on the initial input.
The textbox now contains the text "quote" and displays suggested completions as a dropdown list.
The page shows:
- A textbox labeled "quote" filled with the text "quote on."
- A dropdown list beneath the textbox containing items such as "quote on insurance," "quote on travel," and "quote on booking," each selectable.
- A "Submit" button below the textbox, which can be clicked to submit the selected or typed input to the backend.
- A notification area above the form indicating any immediate feedback or errors from previous actions, currently displaying "Please complete all required fields."
- Additional form fields dynamically appearing based on previous selections, such as a date picker or additional textboxes for more specific information related to the chosen quote option.
I have filled out most of the booking form but still need to select a room type before submission. The form fields are populated, but the "Submit" button remains inactive until all required fields are completed.
</next_state>\

<progress>
in-progress
</progress>
"""

        # prompt = self.add_screenshot(prompt)

        return prompt

    def get_action_reward_prompt(self) -> str:
        if len(self.obs_history) == len(self.states):
            self.obs = self.get_obs_state_strat(
                self.obs_history[-1], self.states[-1], self.strategies[-1]
            )
        prompt = f"""\
{self.obs}\
"""

        prompt += """
# Abstract Example

Here is an abstract version of the answer with description of the content of each tag. Make sure you follow this structure, but replace the content with your answer:
<think>
Observe your current state and proposed strategy in the browser environment, classify the proposed strategy into one of four categories based on progress towards your goal. The categories are:

1. "towards-the-goal" - You are moving closer to achieving the goal.
2. "not-sure" - It's unclear if the action are helping reach the goal.
3. "away-from-the-goal" - Your actions are diverting from the goal.

Explain your reasoning here.
</think>\

<response>
"towards-the-goal", "not-sure", or "away-from-the-goal"
You should be extra-careful when assigning "towards-the-goal" labels. If you are unsure, please select "not-sure" instead.
</response>
"""

        prompt += """
# Concrete Example

Here is a concrete example of how to format your answer. Make sure to follow the template by wrapping with proper html starting and closing tags:
<think>
The proposed action clicks the "Submit" button with 123 without filling out the form above it. It will likely encounter an error, moving away from the goal.
</think>\

<response>
away-from-the-goal
</response>
"""

        return prompt

    def _parse_effectuator_answer(self, text_answer):
        ans_dict = {}
        ans_dict.update(
            parse_html_tags_raise(
                text_answer,
                keys=['action', 'explanation', 'summary'],
                merge_multiple=True,
            )
        )

        try:
            # just check if action can be mapped to python code but keep action as is
            # the environment will be responsible for mapping it to python
            self.action_space.to_python_code(ans_dict['action'])
        except Exception as e:
            raise ParseError(
                f'Error while parsing action\n: {e}\n'
                'Make sure your answer is restricted to the allowed actions.'
            )

        return ans_dict

    def _parse_encoder_answer(self, text_answer):
        ans_dict = {}
        ans_dict.update(
            parse_html_tags_raise(
                text_answer, optional_keys=['think'], merge_multiple=True
            )
        )
        ans_dict.update(
            parse_html_tags_raise(
                text_answer, keys=['state', 'progress'], merge_multiple=True
            )
        )
        return ans_dict

    def _parse_policy_answer(self, text_answer):
        ans_dict = {}
        ans_dict.update(
            parse_html_tags_raise(
                text_answer,
                keys=['strategy', 'summary', 'explanation'],
                merge_multiple=True,
            )
        )
        return ans_dict

    def _parse_dynamics_answer(self, text_answer):
        ans_dict = {}
        ans_dict.update(
            parse_html_tags_raise(
                text_answer, keys=['next_state', 'progress'], merge_multiple=True
            )
        )
        return ans_dict

    def _parse_action_reward_answer(self, text_answer):
        ans_dict = {}
        ans_dict.update(
            parse_html_tags_raise(
                text_answer, optional_keys=['think'], merge_multiple=True
            )
        )
        ans_dict.update(
            parse_html_tags_raise(text_answer, keys=['response'], merge_multiple=True)
        )
        return ans_dict


if __name__ == '__main__':
    pass
    # html_template = """
    # <html>
    # <body>
    # <div>
    # Hello World.
    # Step {}.
    # </div>
    # </body>
    # </html>
    # """

    # OBS_HISTORY = [
    #     {
    #         'goal': 'do this and that',
    #         'pruned_html': html_template.format(1),
    #         'axtree_txt': '[1] Click me',
    #         'last_action_error': '',
    #     },
    #     {
    #         'goal': 'do this and that',
    #         'pruned_html': html_template.format(2),
    #         'axtree_txt': '[1] Click me',
    #         'last_action_error': '',
    #     },
    #     {
    #         'goal': 'do this and that',
    #         'pruned_html': html_template.format(3),
    #         'axtree_txt': '[1] Click me',
    #         'last_action_error': 'Hey, there is an error now',
    #     },
    # ]
    # ACTIONS = ["click('41')", "click('42')"]
    # MEMORIES = ['memory A', 'memory B']
    # THOUGHTS = ['thought A', 'thought B']

    # flags = Flags(
    #     use_html=True,
    #     use_ax_tree=True,
    #     use_thinking=True,
    #     use_error_logs=True,
    #     use_past_error_logs=True,
    #     use_history=True,
    #     use_action_history=True,
    #     use_memory=True,
    #     use_diff=True,
    #     html_type='pruned_html',
    #     use_concrete_example=True,
    #     use_abstract_example=True,
    #     multi_actions=True,
    # )

    # print(
    #     MainPrompt(
    #         obs_history=OBS_HISTORY,
    #         actions=ACTIONS,
    #         memories=MEMORIES,
    #         thoughts=THOUGHTS,
    #         step=0,
    #         flags=flags,
    #     ).prompt
    # )
