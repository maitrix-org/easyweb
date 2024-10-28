encoder_prompt_template = """\
{memory}

# Observation:
{observation}

# State:
Summarize the current state of the webpage observation, focusing on the most \
recent action you took and any errors encountered. Note any dialogs, progress \
indicators, or significant changes such as items in your cart or sites visited. \
Describe the impact of your previous action on the webpage, including any new \
interactive elements. Include any inferred information that may help achieve \
the goal. Information from steps earlier are for reference only. Focus on \
objective description of the current observation and any inferences you can \
draw from it. Report any error messages displayed. Do not include your next \
planned actions; focus solely on providing an objective summary.

Wrap your response in the tag <state> and </state>.\
"""

policy_prompt_template = """\
{memory}

# Current State:
{state}

# Intent:
Describe the action the assistant should take next to carry out the user's \
instruction. \
Avoid using phrases such as "To accomplish the goal," "I will," "To \
proceed.". Avoid ending with phrases like "to execute the search." \
Describe one action at a time and avoid combining multiple steps. \
Refrain from mentioning specific element IDs as they may change \
during execution. Limit your response to one phrase and include any details \
that help select the correct action. Be creative and propose novel \
methods to achieve the goal. Avoid creating accounts without user \
permission or providing personal information. A concrete example \
would be "Go to the home page of Google Flights."

Wrap your response in the tag <intent> and </intent>.\
"""

actor_prompt_template = """\
{memory}

# Observation:
{observation}

# Current State:
{state}

# Current Intent:
{intent}

# Action:
Choose an API call that will carry out the intent when executed in the webpage. \
Use only one action at a time. You must not enclose bid inputs in [brackets] but instead in 'single quotes'. \
Interact only with elements in the current step observation. Your response \
will be executed as a Python function call, so ensure it adheres to the format \
and argument data type specifications defined in the action space.

Wrap your response in the tag <action> and </action>.\
"""
