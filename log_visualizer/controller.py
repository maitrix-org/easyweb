import json
from glob import glob

import gradio as gr
from PIL import Image
from session import TestSession

MAX_TABS = 30


def load_history(log_selection):
    messages = json.load(open(log_selection, 'r'))
    self = TestSession()
    for message in messages:
        self._read_message(message, verbose=False)

    chat_history = [[None, '\n\n'.join(self.action_messages)]]

    tabs = []
    start_url = 'about:blank'
    blank = Image.new('RGB', (1280, 720), (255, 255, 255))
    placeholder = '<placeholder>'

    # print(self.browser_history)
    tabs = []
    urls = []
    screenshots = []
    sub_tabs = []

    # urls = [start_url] + [x[1] for x in self.browser_history]
    # screenshots = [blank] + [x[0] for x in self.browser_history]
    browser_history = [(blank, start_url)] + self.browser_history
    # observations = self.observations
    # states = self.states
    # instructions = self.instructions
    # actions = self.actions

    observations = []
    states = []
    instructions = []
    actions = []

    for i in range(MAX_TABS):
        visible = i < len(self.browser_history)
        with gr.Tab(f'Step {i + 1}', visible=visible) as tab:
            with gr.Group():
                browser_step = (
                    browser_history[i]
                    if i < len(browser_history)
                    else (blank, start_url)
                )
                # content = urls[i] if i < len(urls) else start_url
                url = gr.Textbox(
                    browser_step[1], label='URL', interactive=False, max_lines=1
                )
                # content = screenshots[i] if i < len(screenshots) else blank
                screenshot = gr.Image(
                    browser_step[0], interactive=False, label='Webpage'
                )
                urls.append(url)
                screenshots.append(screenshot)

                with gr.Tab('Observation') as obs_tab:
                    content = (
                        self.observations[i]
                        if i < len(self.observations)
                        else placeholder
                    )
                    observation = gr.Textbox(
                        content,
                        interactive=False,
                        lines=20,
                        max_lines=30,
                    )
                    observations.append(observation)
                    sub_tabs.append(obs_tab)
                with gr.Tab('State') as state_tab:
                    content = self.states[i] if i < len(self.states) else placeholder
                    state = gr.Textbox(
                        content,
                        interactive=False,
                        lines=20,
                        max_lines=30,
                    )
                    states.append(state)
                    sub_tabs.append(state_tab)
                with gr.Tab('Instruction') as instruction_tab:
                    content = (
                        self.instructions[i]
                        if i < len(self.instructions)
                        else placeholder
                    )
                    instruction = gr.Textbox(
                        content,
                        interactive=False,
                        lines=20,
                        max_lines=30,
                    )
                    instructions.append(instruction)
                    sub_tabs.append(instruction_tab)
                with gr.Tab('Action') as action_tab:
                    content = self.actions[i] if i < len(self.actions) else placeholder
                    action = gr.Textbox(
                        content,
                        interactive=False,
                        lines=20,
                        max_lines=30,
                    )
                    actions.append(action)
                    sub_tabs.append(action_tab)

            tabs.append(tab)

    # print(len(tabs))
    return (
        [chat_history]
        + tabs
        + urls
        + screenshots
        + sub_tabs
        + observations
        + states
        + instructions
        + actions
    )


def select_log_dir(log_dir_selection):
    log_list = list(reversed(sorted(glob(f'./{log_dir_selection}/*.json'))))
    return gr.Dropdown(
        log_list,
        value=None,
        interactive=True,
        label='Log',
        info='Choose the log to visualize',
    )


def refresh_log_selection(log_dir_selection):
    log_list = list(reversed(sorted(glob(f'./{log_dir_selection}/*.json'))))
    return gr.Dropdown(
        log_list,
        value=None,
        interactive=True,
        label='Log',
        info='Choose the log to visualize',
    )
