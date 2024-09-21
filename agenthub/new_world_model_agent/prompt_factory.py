from .prompt import AXTree, Error


def get_system_prompt(goal, action_space):
    system_msg = f"""\
# Goal:
{goal}

# Goal Tips:
- If the user's name, phone, email, login credentials, etc. are required to proceed, stop and ask the user.
- When searching for information online, prioritize Google and Wikipedia to avoid problems with browser compatibility.
- Avoid messaging the user while using the browser. Instead provide a comprehensive final answer when you are finished. Only message the user if you are unable to find specific information, explaining what you have done so far.

# Action Space
{action_space.describe(with_long_description=False, with_examples=True)}

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
- Respond to dialogs immediately to proceed. These typically appear at the end of the page and might not have the label "dialog". Accept cookies, select "No Thanks" for insurance offers, and click "Continue" or "Select" button if relevant boxes are filled out.
- You can only open one tab at a time. You can only interact with elements starting with bids; the rest are for information only or out of view.
- If you are blocked by CAPTCHA, go back to the previous page or restart your search.
- If you use go_back() repeatedly but cannot go back to the previous page, consider going to www.google.com to restart your browsing.
"""

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

    return system_msg


def _get_obs_prompt(obs):
    ax_tree = AXTree(
        obs['axtree_txt'],
        visible=True,
        coord_type=False,
        prefix='## ',
    )
    axtree_prompt = ax_tree.prompt
    if len(axtree_prompt) == 0:
        axtree_prompt = '(The webpage is blank)'
    error = Error(
        obs['last_action_error'],
        visible=obs['last_action_error'],
        prefix='## ',
    )

    return f'\n{error.prompt}\n{axtree_prompt}\n\n'


def _get_plan_step_text(action, target_state):
    target_state_text = _get_state_text(target_state)
    return f'### Strategy:\n{action}\n### Target State:\n{target_state_text}'


def _get_plan_prompt(plan):
    # The structure of the plan can be understood as [(action, target_state), (action, target_state), ...]
    # If you think about it, an instruction can be parsed as a precondition, an action, and a postcondition
    # The plan can be understood as a sequence of such instructions
    # The plan can be used to infer the current state of the task
    # We may be able to represent the plan in two ways: one only using the first step, and the other using all steps
    plan_prompt = ''
    for i, (action, target_state) in enumerate(plan):
        plan_prompt += (
            f'\n## Next Step {i+1}:\n{_get_plan_step_text(action, target_state)}\n\n'
        )
    if len(plan_prompt) == 0:
        plan_prompt = 'No plan has been made yet.'
    return plan_prompt


def get_obs_summary_prompt(current_obs, last_action, plan):
    obs_prompt = _get_obs_prompt(current_obs)
    explanation, action = last_action
    plan_prompt = _get_plan_prompt(plan)

    prompt = f"""\
# Plan for Achieving Goal:
{plan_prompt}

# Previous Action:
{explanation}
{action}

# Current Webpage:
{obs_prompt}

# Abstract Example:

Here is an abstract version of the answer with description of the content of each tag. Make sure you follow this structure, but replace the content with your answer:
<summary>
Describe this part of the webpage, such as what website this is, scroll position, the content of the accessibility tree relevant to the goal and plan for achieving goal. Focus on an objective description and do not make any new plans.
</summary>
"""
    return prompt, ['summary']


def _get_state_text(state_dict):
    state_text = ''
    for key in ['content', 'progress', 'reflection']:
        value = state_dict.get(key, 'None')
        if value != 'None':
            state_text += f'### {key.capitalize()}: {value}\n'
    return state_text


def _get_state_memory(state_history):
    state_memory = []
    for i, state_dict in enumerate(state_history):
        # state = {'content': 'None', 'progress': 'None', 'reflection': 'None', 'task_completion': 'None'}
        state_text = _get_state_text(state_dict)
        state_step_prompt = f'## Step {i+1}:\n{state_text}'
        state_memory.append(state_step_prompt)
    state_memory = '\n\n'.join(state_memory)
    if len(state_memory) == 0:
        state_memory = 'No interaction has happened yet.'
    return state_memory


def get_state_prompt(current_obs, obs_summary, last_action, state_history, plan):
    obs_prompt = _get_obs_prompt(current_obs)
    explanation, action = last_action
    plan_prompt = _get_plan_prompt(plan)
    state_memory = _get_state_memory(state_history)

    shared_prompt = f"""\
# Interaction History:
{state_memory}

# Plan for Achieving Goal:
{plan_prompt}

# Previous Action:
{explanation}
{action}

# Current Webpage:
{obs_prompt}

# Webpage Description:
{obs_summary}

# Abstract Example:
Here is an abstract version of the answer with description of the content of each tag. Make sure you follow this structure, but replace the content with your answer:
"""
    content_prompt = (
        shared_prompt
        + """\
<memory>
Describe any new content you should remember, such as information in the accessibility tree or summary relevant to the goal and plan. If there is nothing to add, reply "None".
</memory>
"""
    )
    progress_prompt = (
        shared_prompt
        + """\
<progress>
Describe any new things you have done, such as any specific values entered, websites visited, or information found. Consider how the user who supplied the goal might react to what you did. Focus on an objective description. Avoid making new plans even if at the beginning of planning. If there is no progress, reply "None".
</progress>
"""
    )
    reflection_prompt = (
        shared_prompt
        + """\
<reflection>
Reflect on any mistakes made, and if you need to correct anything to carry out your plan concisely in one or two sentences. Focus on reflecting on the current plan. Do not try to make any new plan. Avoid starting phrases like "Reflecting on the current plan." If there are no mistakes, reply "None".
</reflection>
"""
    )
    task_completion_prompt = (
        shared_prompt
        + """\
<think>
You should think about whether your current plan is completed or not. Base your answer on the provided plan, consider the similarity between the target state and the current state. Avoid making your own plan. If it is completed, reply "finished". If it is not completed, reply "not-finished". If you are not sure, reply "not-sure". If your plan is no longer reasonable in the current state, reply "replan".
</think>

<completion>
"finished", "not-finished", "not-sure", "replan"
</completion>
"""
    )
    prompt_answer_key_dict = {
        content_prompt: ['memory'],
        progress_prompt: ['progress'],
        reflection_prompt: ['reflection'],
        task_completion_prompt: ['think', 'completion'],
    }
    return prompt_answer_key_dict


def _get_plan_memory(partial_plan):
    plan_memory = []
    for i in range(1, len(partial_plan)):
        state_dict = partial_plan[i - 1][1]
        state_text = _get_state_text(state_dict)
        action = partial_plan[i][0]
        plan_memory.append(
            f'## Step {i}:\n### State:\n{state_text}\n### Strategy:\n{action}'
        )
    plan_memory = '\n\n'.join(plan_memory)
    if len(plan_memory) == 0:
        plan_memory = 'Beginning of planning.'
    return plan_memory


def get_strategy_prompt(partial_plan, state_history):
    # The strategy is a high-level plan that can be used to achieve the goal
    # The strategy can be inferred from the state history and the goal
    # The strategy can be represented as a sequence of high-level actions
    # The strategy can be used to guide the agent to achieve the goal
    state_memory = _get_state_memory(state_history)
    plan_memory = _get_plan_memory(partial_plan)
    current_state_dict = partial_plan[-1][1]
    current_state_text = _get_state_text(current_state_dict)

    strategy_prompt = f"""\
# Interaction History:
{state_memory}

# Plan So Far:
{plan_memory}

# Current Step of the Plan:

## Current State:
{current_state_text}

# Abstract Example:
Here is an abstract version of the answer with description of the content of each tag. Make sure you follow this structure, but replace the content with your answer:
<think>
Given the current state, what is the logical next step in your plan to achieve the goal? This step should continue the plan so far and be achievable using a few actions. Avoid repeating previous steps.
</think>\

<strategy>
Describe your next step succinctly. Avoid using phrases such as "To accomplish the goal," "I will," "To proceed," "Assume the previous strategies have been carried out," or "The next step is." Limit your response to one sentence. Avoid mentioning specific element IDs.
</strategy>
"""
    # """Begin with a verb. """
    return strategy_prompt, ['think', 'strategy']


def get_next_state_prompt(current_strategy, partial_plan, state_history):
    # The next state can be inferred from the current state and the strategy
    # The next state can be used to guide the agent to achieve the goal
    # The next state can be represented as a dictionary with content, progress, reflection, and task_completion

    state_memory = _get_state_memory(state_history)
    plan_memory = _get_plan_memory(partial_plan)
    current_state_dict = partial_plan[-1][1]
    current_state_text = _get_state_text(current_state_dict)

    shared_prompt = f"""\
# Interaction History:
{state_memory}

# Plan So Far:
{plan_memory}

# Current Step of the Plan:

## Current State:
{current_state_text}

## Current Strategy:
{current_strategy}

# Abstract Example:
Here is an abstract version of the answer with description of the content of each tag. Make sure you follow this structure, but replace the content with your answer:
"""

    content_prompt = (
        shared_prompt
        + """\
<memory>
Describe any new content you expect to have encountered after executing the current strategy. If there will be nothing to add, reply "None".
</memory>
"""
    )
    progress_prompt = (
        shared_prompt
        + """\
<progress>
Describe any new things you expect to have done, such as any specific values entered, websites visited, or information found, after executing the current strategy. Consider how the user who supplied the goal might react to the outcome. Avoid starting phrases like "After executing the current strategy". If there will be no progress, reply "None".
</progress>
"""
    )
    reflection_prompt = (
        shared_prompt
        + """\
<reflection>
Reflect on any mistakes you expect to have made, and if you expect to be able to correct anything to achieve the goal, after executing the current strategy. Respond concisely in one or two sentences. If there will be no mistakes, reply "None".
</reflection>
"""
    )
    task_completion_prompt = (
        shared_prompt
        + """\
<think>
Consider whether your goal will have been achieved or not after executing the current strategy. If you expect it to be achieved, reply "finished". If it is not achieved, reply "not-finished". If you are not sure, reply "not-sure".
</think>

<completion>
"finished", "not-finished", "not-sure"
</completion>
"""
    )
    prompt_answer_key_dict = {
        content_prompt: ['memory'],
        progress_prompt: ['progress'],
        reflection_prompt: ['reflection'],
        task_completion_prompt: ['think', 'completion'],
    }
    return prompt_answer_key_dict


def get_action_reward_prompt(current_strategy, state_history, partial_plan):
    # The action reward can be inferred from the current state and the goal
    # The action reward can be used to guide the agent to achieve the goal
    # The action reward can be represented as a tuple of reward, think, and response

    state_memory = _get_state_memory(state_history)
    plan_memory = _get_plan_memory(partial_plan)
    current_state_dict = partial_plan[-1][1]
    current_state_text = _get_state_text(current_state_dict)

    action_reward_prompt = f"""\
# Interaction History:
{state_memory}

# Plan So Far:
{plan_memory}

# Current Step of the Plan:

## Current State:
{current_state_text}

## Current Strategy:
{current_strategy}

# Abstract Example:
Here is an abstract version of the answer with description of the content of each tag. Make sure you follow this structure, but replace the content with your answer:
<think>
Based on your current state, classify the current strategy into one of four categories based on progress towards your goal after the strategy is executed. The categories are:

1. "towards-the-goal" - You are moving closer to achieving the goal.
2. "goal-achieved" - You will achieve the goal after the current strategy is executed.
2. "not-sure" - It's unclear if the strategy are helping reach the goal.
3. "away-from-the-goal" - Your actions are diverting from the goal.

Explain your reasoning here.
</think>

<response>
"towards-the-goal", "goal-achieved", "not-sure", or "away-from-the-goal"
If you are unsure, please select "not-sure" instead.
</response>
"""
    return action_reward_prompt, ['think', 'response']


def get_action_prompt(
    current_obs, current_state_dict, last_action, plan, state_history
):
    state_memory = _get_state_memory(state_history)
    obs_prompt = _get_obs_prompt(current_obs)
    obs_summary = current_state_dict['summary']
    current_state_text = _get_state_text(current_state_dict)
    action, explanation = last_action
    current_plan_step_text = _get_plan_step_text(*plan[0])
    future_plan_prompt = _get_plan_prompt(plan[1:])
    future_plan_prompt = future_plan_prompt.replace('No plan', 'No future plan')

    action_prompt = f"""\
# Interaction History:
{state_memory}

# Previous Action
{explanation}
{action}

# Current Webpage:
{obs_prompt}

# Current State:

## Webpage Summary:
{obs_summary}

{current_state_text}

# Current Plan:
{current_plan_step_text}

# Future Plans:
{future_plan_prompt}

# Abstract Example

Here is an abstract version of the answer with description of the content of each tag. Make sure you follow this structure, but replace the content with your answer:
<action>
Select one single action in order to carry out the current plan towards the target state. Break down the plan into individual, manageable actions. Use only one action at a time. You must not enclose bid inputs in [brackets]. Your response will be executed as a Python function call, so ensure it adheres to the format and argument data type specifications defined in the action space.
</action>\

<explanation>
Describe the action to be taken using a single concise sentence. Focus on the single action. Use clear and simple language to describe your action.
</explanation>
"""
    return action_prompt, ['action', 'explanation']
