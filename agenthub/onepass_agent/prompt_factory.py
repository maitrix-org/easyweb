from datetime import datetime


def get_history_prompt(history, include_action=True):
    history_steps = []
    for i, (obs, state, instruction, action) in enumerate(history):
        step = f"""\
## Step {i + 1}:\

### State:
{state}\

### Instruction:
{instruction}\
"""
        if include_action:
            step += f"""
### Action:
{action}\
"""
        history_steps.append(step)

    history_prompt = '\n\n'.join(history_steps)
    if len(history_prompt) > 0:
        return history_prompt
    else:
        return 'Beginning of task'


def _get_axtree_prompt(obs):
    if len(obs['error_prefix']) > 0 and not obs['clean_axtree_txt'].startswith(
        obs['error_prefix']
    ):
        axtree = f"{obs['error_prefix']}\n" + obs['clean_axtree_txt']
    else:
        axtree = obs['clean_axtree_txt']
    return axtree


def get_onepass_prompt(current_obs, history, goal):
    # current_datetime = datetime.now().strftime('%a, %b %d, %Y %H:%M:%S')
    current_datetime = datetime.strptime(
        '2024-10-12-20:32:02', '%Y-%m-%d-%H:%M:%S'
    ).strftime('%b %d, %Y %H:%M:%S')
    current_datetime_prompt = f'\n# Current Date and Time:\n{current_datetime}\n\n'

    axtree = _get_axtree_prompt(current_obs)
    history_prompt = get_history_prompt(history)

    example_prompt = f"""
{current_datetime_prompt}\

# Goal:
{goal}

# History:
{history_prompt}

# Step {len(history) + 1}:
## Observation:
{axtree}

## Answer:
"""
    return example_prompt


def get_system_prompt():
    return """# Instructions
Review the current state of the page and all other information to find the best possible next action to accomplish your goal. Your answer will be interpreted and executed by a program, make sure to follow the formatting instructions.

# Action Space

16 different types of actions are available.

noop(wait_ms: float = 1000)

send_msg_to_user(text: str)

scroll(delta_x: float, delta_y: float)

fill(bid: str, value: str)

click(bid: str, button: Literal['left', 'middle', 'right'] = 'left', modifiers: list[typing.Literal['Alt', 'Control', 'Meta', 'Shift']] = [])

dblclick(bid: str, button: Literal['left', 'middle', 'right'] = 'left', modifiers: list[typing.Literal['Alt', 'Control', 'Meta', 'Shift']] = [])

press(bid: str, key_comb: str)

clear(bid: str)

drag_and_drop(from_bid: str, to_bid: str)

upload_file(bid: str, file: str | list[str])

go_back()

go_forward()

goto(url: str)

Only a single action can be provided at once.
"""
