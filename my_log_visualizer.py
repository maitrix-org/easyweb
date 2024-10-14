import base64
import json
import os
from glob import glob
from io import BytesIO

import gradio as gr
import plotly.graph_objects as go
from PIL import Image, UnidentifiedImageError

from my_frontend import (
    LABEL_LEN,
    LINE_LEN,
    Node,
    parse_and_visualize,
    process_string,
    visualize_tree_plotly,
)

api_key = os.environ.get('OPENAI_API_KEY')


def parse_log_onestep(log_file):
    count = 0
    nodes = {}
    current_node = None
    root = None
    in_state = False
    state_info = ''
    log_string = log_file
    lines = log_string.strip().split('\n')

    for line in lines:
        if line.startswith('*State*'):
            in_state = True
            state_info = (
                state_info + process_string(line.split(': ')[1], LINE_LEN) + '<br>'
            )

        if (
            in_state
            and not (line.startswith('*State*'))
            and not (line.startswith('*Replan Reasoning*'))
        ):
            state_info = state_info + process_string(line, LINE_LEN) + '<br>'

        if line.startswith('*Replan Reasoning*'):
            in_state = False
            current_node = Node(count, 'null', state_info, 'Init', 0.0, None)
            if root is None:
                root = current_node
            nodes[count] = current_node
            count += 1
            state_info = ''

        if line.startswith('*Replan Status*'):
            status = process_string(line.split(': ')[1], LINE_LEN)
            current_node.status = status

        if line.startswith('*Active Strategy*'):
            strat_info = process_string(line.split(': ')[1], LINE_LEN)

        if line.startswith('*Action*'):
            action_info = process_string(line.split(': ')[1], LINE_LEN)

        if line.startswith('*Summary'):
            summary = process_string(line.split(': ')[1], LABEL_LEN)
            nodes[count] = Node(
                count,
                strat_info + '<br>' + '<b>Grounding</b>: ' + action_info,
                'null',
                'null',
                0.0,
                None,
            )
            nodes[count].set_summary(summary)
            current_node.children.append(nodes[count])
            nodes[count].parent = current_node
            count += 1

    return root, nodes


def parse_and_visualize_onestep(log_file):
    root, nodes = parse_log_onestep(log_file)
    # print(root, nodes)
    fig = visualize_tree_plotly(root, nodes)
    return fig


class TestSession:
    def __init__(self):
        self.token = self.status = self.agent_state = self.figure = None
        self.action_messages = []
        self.browser_history = []
        self.figures = []
        self.webpages = []

    def _read_message(self, message, verbose=True):
        printable = {}
        if message.get('token'):
            self.token = message['token']
            self.status = message['status']
            printable = message
        elif message.get('observation') == 'agent_state_changed':
            self.agent_state = message['extras']['agent_state']
            printable = message
        elif 'action' in message:
            self.action_messages.append(message['message'])
            if message['action'] == 'browse_interactive':
                self._update_figure(message)
                self.figures.append(self.figure)

                webpage = self._load_webpage(message)
                self.webpages.append(webpage)
            printable = message
        elif 'extras' in message and 'screenshot' in message['extras']:
            image_data = base64.b64decode(message['extras']['screenshot'])
            try:
                screenshot = Image.open(BytesIO(image_data))
                url = message['extras']['url']
                printable = {
                    k: v for k, v in message.items() if k not in ['extras', 'content']
                }
                self.browser_history.append((screenshot, url))
            except UnidentifiedImageError:
                err_msg = (
                    'Failure to receive screenshot, likely due to a server-side error.'
                )
                self.action_messages.append(err_msg)
        if verbose:
            print(printable)

    def _update_figure(self, message):
        if (
            ('args' in message)
            and ('thought' in message['args'])
            and (message['args']['thought'].find('MCTS') != -1)
        ):
            # print('Update figure')
            if message['args']['thought'].startswith('{'):
                log_content = json.loads(message['args']['thought'])['full_output']
            else:
                log_content = message['args']['thought']
            # print(log_content)
            self.figure = parse_and_visualize(log_content)
            # figure.show()
        elif (
            'args' in message
            and 'thought' in message['args']
            and 'State' in message['args']['thought']
        ):
            if message['args']['thought'].startswith('{'):
                log_content = json.loads(message['args']['thought'])['full_output']
            else:
                log_content = message['args']['thought']
            # self.figure = parse_and_visualize_onestep(message['args']['thought'])
            # self.figure = parse_and_visualize_onestep(log_content)

    def _load_webpage(self, message):
        webpage = ''
        print('Start load webpage')
        print(message)
        if (
            ('args' in message)
            and ('thought' in message['args'])
            and message['args']['thought'].startswith('{')
        ):
            print('Load webpage')
            planning_record = json.loads(message['args']['thought'])
            if 'obs' in planning_record:
                print('Planning record')
                webpage = planning_record['obs']['clean_axtree_txt']
        return webpage


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
    plots = []
    webpages = []

    browser_history = [(blank, start_url)] + self.browser_history
    print(len(browser_history))
    for i in range(len(self.figures)):
        img, txt = browser_history[min(i, len(browser_history) - 1)]
        figure = self.figures[i]
        webpage = self.webpages[i]
        with gr.Tab(f'Step {i+1}', visible=True) as tab:
            with gr.Group():
                url = gr.Textbox(txt, label='URL', interactive=False, max_lines=1)
                screenshot = gr.Image(img, interactive=False, label='Webpage')
                with gr.Tab('Plot') as plot_tab:
                    plot = gr.Plot(figure, label='Agent Reasoning Process')
                    sub_tabs.append(plot_tab)
                with gr.Tab('Webpage') as obs_tab:
                    webpage = gr.Textbox(
                        webpage, interactive=False, lines=20, max_lines=30
                    )
                    sub_tabs.append(obs_tab)

                urls.append(url)
                screenshots.append(screenshot)
                plots.append(plot)
                webpages.append(webpage)

            tabs.append(tab)
        if len(tabs) > max_tabs:
            tabs = tabs[1:]

    while len(tabs) < max_tabs:
        with gr.Tab(f'Step {len(tabs)+1}', visible=False) as tab:
            with gr.Group():
                url = gr.Textbox(start_url, label='URL', interactive=False, max_lines=1)
                screenshot = gr.Image(blank, interactive=False, label='Webpage')
                with gr.Tab('Plot') as plot_tab:
                    plot = gr.Plot(go.Figure(), label='Agent Reasoning Process')
                    sub_tabs.append(plot_tab)
                with gr.Tab('Webpage') as obs_tab:
                    webpage = gr.Textbox(
                        placeholder, interactive=False, lines=20, max_lines=30
                    )
                    sub_tabs.append(obs_tab)

                urls.append(url)
                screenshots.append(screenshot)
                plots.append(plot)
                webpages.append(webpage)

            tabs.append(tab)

    # print(len(tabs))
    return [chat_history] + tabs + urls + screenshots + sub_tabs + plots + webpages


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


if __name__ == '__main__':
    # log_list = list(reversed(sorted(glob('./frontend_logs/*.json'))))

    with gr.Blocks() as demo:
        title = gr.Markdown('# FastAgent Log Visualizer')
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                with gr.Group():
                    log_dir_options = glob('**/*_logs/', recursive=True)
                    default_logdir = log_dir_options[-1]
                    log_list = list(
                        reversed(sorted(glob(f'./{default_logdir}/*.json')))
                    )
                    log_dir_selection = gr.Dropdown(
                        log_dir_options, value=default_logdir, label='Log Directory'
                    )
                    log_selection = gr.Dropdown(
                        log_list,
                        value=None,
                        interactive=True,
                        label='Log',
                        info='Choose the log to visualize',
                    )
                    chatbot = gr.Chatbot()
                refresh = gr.Button('Refresh Log List')

            with gr.Column(scale=2):
                start_url = 'about:blank'
                blank = Image.new('RGB', (1280, 720), (255, 255, 255))
                placeholder = '<placeholder>'

                max_tabs = 100
                tabs = []
                urls = []
                screenshots = []
                sub_tabs = []
                plots = []
                webpages = []
                while len(tabs) < max_tabs:
                    with gr.Tab(f'Step {len(tabs)+1}', visible=(len(tabs) == 0)) as tab:
                        with gr.Group():
                            url = gr.Textbox(
                                start_url, label='URL', interactive=False, max_lines=1
                            )
                            screenshot = gr.Image(
                                blank, interactive=False, label='Webpage'
                            )
                            with gr.Tab('Plot') as plot_tab:
                                plot = gr.Plot(
                                    go.Figure(), label='Agent Reasoning Process'
                                )
                                sub_tabs.append(plot_tab)
                            with gr.Tab('Webpage') as obs_tab:
                                webpage = gr.Textbox(
                                    placeholder,
                                    interactive=False,
                                    lines=20,
                                    max_lines=30,
                                )
                                sub_tabs.append(obs_tab)

                            urls.append(url)
                            screenshots.append(screenshot)
                            plots.append(plot)
                            webpages.append(webpage)

                        tabs.append(tab)
                # print(len(tabs))

        log_dir_selection.select(select_log_dir, log_dir_selection, log_selection)
        log_selection.select(
            load_history,
            log_selection,
            [chatbot] + tabs + urls + screenshots + sub_tabs + plots + webpages,
        )
        refresh.click(refresh_log_selection, log_dir_selection, log_selection)

    demo.queue()
    demo.launch(share=True)
