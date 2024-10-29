import base64
import json
import os
import time
from io import BytesIO

import gradio as gr
import networkx as nx
import plotly.graph_objects as go
import websocket
from PIL import Image, UnidentifiedImageError

# from openai import OpenAI

default_api_key = os.environ.get('OPENAI_API_KEY')
LINE_LEN = 100
LABEL_LEN = 20
WIDTH = 18
HEIGHT = 4
RADIUS = 1


class Node:
    def __init__(self, state, in_action, state_info, status, reward, parent):
        self.state = state
        self.in_action = in_action
        self.state_info = state_info
        self.status = status
        self.parent = parent
        self.children = []
        self.reward = reward
        self.Q = 0.0
        self.uct = 0.0
        self.summary = 'Start Planning'

    def set_summary(self, summary):
        self.summary = summary


class OpenDevinSession:
    def __init__(
        self,
        agent,
        port,
        model,
        language='en',
        api_key=default_api_key,
    ):
        self.model = model
        self.agent = agent
        self.language = language
        self.api_key = api_key
        self.port = port

        self.figure = None

        self._reset()

    def initialize(self, as_generator=False):
        self.agent_state = None
        if self.ws:
            self._close()
        self.ws = websocket.WebSocket()
        self.ws.connect(f'ws://127.0.0.1:{self.port}/ws')

        payload = {
            'action': 'initialize',
            'args': {
                'LLM_MODEL': self.model,
                'AGENT': self.agent,
                'LANGUAGE': self.language,
                'LLM_API_KEY': self.api_key,
            },
        }
        self.ws.send(json.dumps(payload))

        while self.agent_state != 'init':
            message = self._get_message()
            if message.get('token'):
                self.token, self.status = message['token'], message['status']
            elif message.get('observation') == 'agent_state_changed':
                self.agent_state = message['extras']['agent_state']
                if as_generator:
                    yield self.agent_state
        print(f'{self.agent} Initialized')

    def pause(self):
        if self.agent_state != 'running':
            raise ValueError('Agent not running, nothing to pause')
        print('Pausing')

        payload = {'action': 'change_agent_state', 'args': {'agent_state': 'paused'}}
        self.ws.send(json.dumps(payload))

        self.agent_state = 'pausing'

    def resume(self):
        if self.agent_state != 'paused':
            raise ValueError('Agent not paused, nothing to resume')
        print('Resuming')

        payload = {'action': 'change_agent_state', 'args': {'agent_state': 'running'}}
        self.ws.send(json.dumps(payload))

        self.agent_state = 'resuming'

    def run(self, task):
        if self.agent_state not in ['init', 'running', 'pausing', 'resuming', 'paused']:
            raise ValueError(
                'Agent not initialized. Please run the initialize() method first'
            )

        if task is not None:
            payload = {'action': 'message', 'args': {'content': task}}
            self.ws.send(json.dumps(payload))

        while self.agent_state not in ['finished', 'paused']:
            message = self._get_message()
            self._read_message(message)

            # self._update_figure(message)

            print(self.agent_state)
            yield message

    def _get_message(self):
        # try:
        response = self.ws.recv()
        try:
            message = json.loads(response)
            message_size = len(str(message))
            print(f'Received message of size: {message_size}')
        except json.decoder.JSONDecodeError as e:
            print(e)
            print(response)
            message = {
                'action': 'error',
                'message': 'Received JSON response cannot be parsed. Skipping..',
                'response': response,
            }

        self.raw_messages.append(message)
        # print(list(message.keys()))
        return message
        # except json.decoder.JSONDecodeError as e:
        #     return {}

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
            # print(message)
            if message['action'] != 'browse_interactive':
                self.action_messages.append(message['message'])
            elif self.agent == 'WorldModelAgent':
                full_output_dict = json.loads(message['args']['thought'])
                if full_output_dict['active_strategy'] != self.last_active_strategy:
                    self.last_active_strategy = full_output_dict['active_strategy']
                    self.action_history.append((0, self.last_active_strategy))
                self.action_history.append((1, full_output_dict['summary']))
            else:
                self.action_messages.append(message['message'])
                self.action_history.append((0, message['message']))
            # printable = message
            printable = {k: v for k, v in message.items() if k not in 'args'}
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
            # log_content = message['args']['thought']
            # self.figure = parse_and_visualize(log_content)

            planning_record = json.loads(message['args']['thought'])
            self.figure = parse_and_visualize(planning_record['full_output'])

    def _reset(self, agent_state=None):
        self.token, self.status = None, None
        self.ws, self.agent_state = None, agent_state
        self.is_paused = False
        self.raw_messages = []
        self.browser_history = []
        self.action_history = []
        self.last_active_strategy = ''
        self.action_messages = []
        self.figure = go.Figure()

    def _close(self):
        print(f'Closing connection {self.token}')
        if self.ws:
            self.ws.close()
        now = time.time()
        from datetime import datetime

        os.makedirs('frontend_logs', exist_ok=True)

        # Get current date and time
        now = datetime.now()
        # Format date and time
        formatted_now = now.strftime('%Y-%m-%d-%H:%M:%S')
        formatted_model = self.model.replace('/', '-')
        output_path = (
            f'frontend_logs/{formatted_now}_{self.agent}_{formatted_model}_steps.json'
        )
        print('Saving log to', output_path)
        json.dump(self.raw_messages, open(output_path, 'w'))
        self._reset()

    def __del__(self):
        self._close()


def process_string(string, line_len):
    preformat = string.split('\n')
    final = []
    for sentence in preformat:
        formatted = []
        for i in range(0, len(sentence), line_len):
            splitted = sentence[i : i + line_len]
            if (
                i + line_len < len(sentence)
                and sentence[i + line_len].isalnum()
                and splitted[-1].isalnum()
            ):
                formatted.append(splitted + '-')
            else:
                formatted.append(splitted)
        final.append('<br>'.join(formatted))

    return '\n'.join(final)


def update_Q(node):
    if len(node.children) == 0:
        node.Q = node.reward
        return node.reward
    else:
        total_Q = node.reward
        for child in node.children:
            if child.status != 'Init' and child.status != 'null':
                total_Q += update_Q(child)
        node.Q = total_Q
        return node.Q


def parse_log(log_file):
    count = 0
    nodes = {}
    current_node = None
    root = None
    chosen_node = -1
    in_next_state = False
    next_state = ''
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

        if line.startswith('*Strategy Candidate*'):
            strat_info = process_string(line.split(': ')[1], LINE_LEN)

        if line.startswith('*Summary*'):
            summary = process_string(line.split(': ')[1], LABEL_LEN)

        if line.startswith('*Fast Reward*'):
            reward = float(line.split(': ')[1])
            nodes[count] = Node(count, strat_info, 'null', 'null', reward, None)
            nodes[count].set_summary(summary)
            current_node.children.append(nodes[count])
            nodes[count].parent = current_node
            count += 1

        if line.startswith('*Expanded Strategy*'):
            expanded_strat = process_string(line.split(': ')[1], LINE_LEN)
            for node_num, node in nodes.items():
                if node.in_action == expanded_strat:
                    chosen_node = node_num

        if line.startswith('*Next State*'):
            in_next_state = True
            next_state = (
                next_state + process_string(line.split(': ')[1], LINE_LEN) + '<br>'
            )

        if (
            in_next_state
            and not (line.startswith('*Next State*'))
            and not (line.startswith('*Status*'))
        ):
            next_state = next_state + process_string(line, LINE_LEN) + '<br>'

        if line.startswith('*Status*'):
            status = process_string(line.split(': ')[1], LINE_LEN)
            nodes[chosen_node].state_info = next_state
            nodes[chosen_node].status = status
            current_node = nodes[chosen_node]
            chosen_node = -1
            in_next_state = False
            next_state = ''

    update_Q(root)
    return root, nodes


def visualize_tree_plotly(root, nodes):
    G = nx.DiGraph()

    def add_edges(node):
        for child in node.children:
            G.add_edge(node.state, child.state)
            add_edges(child)

    def get_nodes_by_level(node, level, level_nodes):
        if level not in level_nodes:
            level_nodes[level] = []
        level_nodes[level].append(node)
        for child in node.children:
            get_nodes_by_level(child, level + 1, level_nodes)

    add_edges(root)

    level_nodes = {}
    get_nodes_by_level(root, 0, level_nodes)

    highest_q_nodes = set()
    for level, nodes_at_level in level_nodes.items():
        highest_q_node = max(nodes_at_level, key=lambda x: x.Q)
        highest_q_nodes.add(highest_q_node.state)

    pos = horizontal_hierarchy_pos(G, root.state)
    edge_x = []
    edge_y = []

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=2, color='white'),
        hoverinfo='none',
        mode='lines',
        showlegend=False,
    )

    node_x = []
    node_y = []
    hover_texts = []
    colors = []
    shapes = []
    annotations = []
    width, height, radius = WIDTH, HEIGHT, RADIUS

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        hover_text = (
            f'<b>State {node}</b><br>'
            f'<b>Reward:</b> {nodes[node].reward}<br>'
            f'<b>Q:</b> {nodes[node].Q}<br>'
            f'<b>In Action:</b> {"<br>" + nodes[node].in_action if nodes[node].in_action != "null" else nodes[node].in_action}<br>'
            f'<b>State Info:</b> {"<br>" + nodes[node].state_info if nodes[node].state_info != "null" else nodes[node].state_info}<br>'
            f'<b>Status:</b> {nodes[node].status}'
        )
        hover_texts.append(hover_text)

        annotations.append(
            dict(
                x=x,
                y=y,
                text=nodes[node].summary,
                xref='x',
                yref='y',
                showarrow=False,
                font=dict(family='Arial', size=12, color='black'),
                align='center',
            )
        )

        if node in highest_q_nodes:
            colors.append('pink')
        else:
            colors.append('#FFD700')

        custom_path = (
            f'M{x - width / 2 + radius},{y - height / 2} '
            f'L{x + width / 2 - radius},{y - height / 2} '
            f'Q{x + width / 2},{y - height / 2} {x + width / 2},{y - height / 2 + radius} '
            f'L{x + width / 2},{y + height / 2 - radius} '
            f'Q{x + width / 2},{y + height / 2} {x + width / 2 - radius},{y + height / 2} '
            f'L{x - width / 2 + radius},{y + height / 2} '
            f'Q{x - width / 2},{y + height / 2} {x - width / 2},{y + height / 2 - radius} '
            f'L{x - width / 2},{y - height / 2 + radius} '
            f'Q{x - width / 2},{y - height / 2} {x - width / 2 + radius},{y - height / 2} '
            f'Z'
        )

        shapes.append(
            dict(
                xref='x',
                yref='y',
                type='path',
                path=custom_path,
                fillcolor='pink' if node in highest_q_nodes else '#FFD700',
                line_color='black',
            )
        )

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers',
        hoverinfo='text',
        text=hover_texts,
        hoverlabel=dict(font=dict(size=16)),
        marker=dict(showscale=False, color=colors, size=0),
        showlegend=False,
    )

    agent_choice_trace = go.Scatter(
        x=[None],
        y=[None],
        mode='markers',
        marker=dict(
            size=10,
            color='pink',
            line=dict(width=2),
        ),
        showlegend=True,
        name='Agent Choice',
    )

    candidate_trace = go.Scatter(
        x=[None],
        y=[None],
        mode='markers',
        marker=dict(
            size=10,
            color='#FFD700',
            line=dict(width=2),
        ),
        showlegend=True,
        name='Candidate',
    )

    fig = go.Figure(
        data=[
            edge_trace,
            node_trace,
            agent_choice_trace,
            candidate_trace,
        ],  # label_trace,
        layout=go.Layout(
            title='Agent Thinking Process',
            titlefont_size=16,
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                visible=False,
                range=[-10, 80],
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                visible=False,
                range=[0, 40],
            ),
            width=920,
            height=540,
            shapes=shapes,
            annotations=annotations,
            dragmode='pan',
        ),
    )
    fig.update_layout(
        modebar_remove=[
            'zoom2d',
            'zoomIn2d',
            'zoomOut2d',
            'autoScale2d',
            'resetScale2d',
            'lasso2d',
            'select2d',
        ]
    )
    return fig


def horizontal_hierarchy_pos(G, root, height=110, hor_gap=20.0, hor_loc=0, ycenter=20):
    pos = _horizontal_hierarchy_pos(G, root, height, hor_gap, hor_loc, ycenter)
    return pos


def _horizontal_hierarchy_pos(
    G,
    root,
    height=110,
    hor_gap=20.0,
    hor_loc=0,
    ycenter=20.0,
    pos=None,
    parent=None,
    parsed=None,
):
    if pos is None:
        pos = {root: (hor_loc, ycenter)}
    if parsed is None:
        parsed = []

    pos[root] = (hor_loc, ycenter)
    children = list(G.neighbors(root))
    if not isinstance(G, nx.DiGraph) and parent is not None:
        children.remove(parent)
    if len(children) != 0:
        dy = height / len(children)
        if dy < HEIGHT:
            dy = HEIGHT
        else:
            nexty = ycenter - height / 2 - dy / 2

        for child in children:
            nexty += dy
            pos = _horizontal_hierarchy_pos(
                G,
                child,
                height=dy,
                hor_gap=hor_gap,
                hor_loc=hor_loc + hor_gap,
                ycenter=nexty,
                pos=pos,
                parent=root,
                parsed=parsed,
            )
    return pos


def parse_and_visualize(log_file):
    root, nodes = parse_log(log_file)
    fig = visualize_tree_plotly(root, nodes)
    return fig


def user(user_message, history):
    return '', history + [[user_message, None]]


def get_status(agent_state):
    if agent_state == 'loading':
        status = 'Agent Status: 🟡 Loading'
    elif agent_state == 'init':
        status = 'Agent Status: 🟢 Initialized'
    elif agent_state == 'running':
        status = 'Agent Status: 🟢 Running'
    elif agent_state == 'pausing':
        status = 'Agent Status: 🟢 Pausing'
    elif agent_state == 'paused':
        status = 'Agent Status: 🟡 Paused'
    elif agent_state == 'resuming':
        status = 'Agent Status: 🟡 Resuming'
    elif agent_state == 'finished':
        status = 'Agent Status: 🟢 Finished'
    elif agent_state == 'stopped':
        status = 'Agent Status: 🔴 Stopped'
    elif agent_state is None:
        status = 'Agent Status: 🔴 Inactive'
    else:
        status = f'Agent Status: 🔴 {agent_state}'

    return status


def get_action_history_markdown(action_history):
    text = ''
    for level, line in action_history:
        text += '  ' * level + '* ' + line + '\n'
    # print(text)
    return text


def get_messages(
    chat_history,
    action_messages,
    browser_history,
    session,
    status,
    agent_selection,
    model_selection,
    api_key,
):
    model_selection = model_display2name[model_selection]
    print('Get Messages', session.agent_state)
    if len(chat_history) > 0:
        if chat_history[-1][1] is None:
            user_message = chat_history[-1][0]
            chat_history[-1][1] = ''
        else:
            user_message = None
            chat_history[-1][1] = chat_history[-1][1].strip() + '\n\n'
    else:
        user_message = None

    if (
        session.agent_state is None or session.agent_state in ['paused', 'finished']
    ) and user_message is None:
        clear = gr.Button('Clear', interactive=True)
        if len(chat_history) > 0:
            chat_history[-1][1] = '\n\n'.join(action_messages)
        status = get_status(session.agent_state)
        screenshot, url = browser_history[-1]

        if session.figure:
            figure = session.figure
        else:
            figure = go.Figure()

        action_history = get_action_history_markdown(session.action_history)
        action_history = action_history if action_history else 'No Action Taken Yet'

        yield (
            chat_history,
            screenshot,
            url,
            action_messages,
            browser_history,
            session,
            status,
            clear,
            figure,
            action_history,
        )
    else:
        clear = gr.Button('Clear', interactive=False)
        if session.agent_state not in ['init', 'running', 'pausing', 'resuming']:
            session.agent = agent_selection
            # session.model = model_port_config[model_selection]["provider"] + '/' + model_selection
            session.model = model_selection
            if model_requires_key[model_selection]:
                session.api_key = api_key
            elif model_port_config[model_selection].get('default_key', None):
                session.api_key = model_port_config[model_selection].get(
                    'default_key', None
                )
            else:
                session.api_key = ''

            print('API Key:', session.api_key)
            # session.api_key = (
            #     api_key if len(api_key) > 0 else 'token-abc123'
            # )  # token-abc123
            action_messages = []
            browser_history = browser_history[:1]
            for agent_state in session.initialize(as_generator=True):
                status = get_status(agent_state)
                screenshot, url = browser_history[-1]

                if session.figure:
                    figure = session.figure
                else:
                    figure = go.Figure()

                action_history = get_action_history_markdown(session.action_history)
                action_history = (
                    action_history if action_history else 'No Action Taken Yet'
                )

                yield (
                    chat_history,
                    screenshot,
                    url,
                    action_messages,
                    browser_history,
                    session,
                    status,
                    clear,
                    figure,
                    action_history,
                )

        for message in session.run(user_message):
            clear = gr.Button('Clear', interactive=(session.agent_state == 'finished'))
            status = get_status(session.agent_state)
            while len(session.action_messages) > len(action_messages):
                diff = len(session.action_messages) - len(action_messages)
                action_messages.append(session.action_messages[-diff])
                # chat_history[-1][1] += session.action_messages[-diff] + '\n\n'
                chat_history[-1][1] = '\n\n'.join(action_messages)
            while len(session.browser_history) > (len(browser_history) - 1):
                diff = len(session.browser_history) - (len(browser_history) - 1)
                browser_history.append(session.browser_history[-diff])
            screenshot, url = browser_history[-1]

            if session.figure:
                figure = session.figure
            else:
                figure = go.Figure()

            action_history = get_action_history_markdown(session.action_history)
            action_history = action_history if action_history else 'No Action Taken Yet'

            yield (
                chat_history,
                screenshot,
                url,
                action_messages,
                browser_history,
                session,
                status,
                clear,
                figure,
                action_history,
            )


def clear_page(browser_history, session):
    browser_history = browser_history[:1]
    current_screenshot, current_url = browser_history[-1]
    session._close()
    status = get_status(session.agent_state)
    # pause_resume = gr.Button("Pause", interactive=False)
    return (
        None,
        'Pause',
        False,
        current_screenshot,
        current_url,
        [],
        browser_history,
        session,
        status,
        go.Figure(),
        'No Action Taken Yet',
    )


def check_requires_key(model_selection, api_key):
    model_real_name = model_display2name[model_selection]
    requires_key = model_requires_key[model_real_name]
    if requires_key:
        api_key = gr.Textbox(
            api_key, label='API Key', placeholder='Your API Key', visible=True
        )
    else:
        api_key = gr.Textbox(
            api_key, label='API Key', placeholder='Your API Key', visible=False
        )
    return api_key


def pause_resume_task(is_paused, session, status):
    if not is_paused and session.agent_state == 'running':
        session.pause()
        is_paused = True
    elif is_paused and session.agent_state == 'paused':
        session.resume()
        is_paused = False

    button = 'Resume' if is_paused else 'Pause'
    status = get_status(session.agent_state)
    return button, is_paused, session, status


def toggle_options(visible):
    new_visible = not visible
    toggle_text = 'Hide Advanced Options' if new_visible else 'Show Advanced Options'
    return (
        # gr.update(visible=new_visible),
        gr.update(visible=new_visible),
        new_visible,
        gr.update(value=toggle_text),
    )


current_dir = os.path.dirname(__file__)
print(os.path.dirname(__file__))

default_port = 5000
with open(os.path.join(current_dir, 'Makefile')) as f:
    while True:
        line = f.readline()
        if 'BACKEND_PORT' in line:
            default_port = int(line.split('=')[1].strip())
            break
        if not line:
            break
default_agent = 'WorldModelAgent'

global model_port_config
model_port_config = {}
with open(os.path.join(current_dir, 'model_port_config.json')) as f:
    model_port_config = json.load(f)
# model_list = list(model_port_config.keys())
# model_list = [cfg.get('display_name', model) for model, cfg in model_port_config.items()]
global model_display2name
model_display2name = {
    cfg.get('display_name', model): model for model, cfg in model_port_config.items()
}
model_list = list(model_display2name.keys())
global model_requires_key
model_requires_key = {
    model: cfg.get('requires_key', False) for model, cfg in model_port_config.items()
}

default_model = model_list[0]
for model, cfg in model_port_config.items():
    if cfg.get('default', None):
        default_model = cfg.get('display_name', model)
        break

default_api_key = os.environ.get('OPENAI_API_KEY')

with gr.Blocks() as demo:
    title = gr.Markdown('# OpenQ')
    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            with gr.Group():
                agent_selection = gr.Dropdown(
                    [
                        'DummyWebAgent',
                        'WorldModelAgent',
                        'NewWorldModelAgent',
                        'FewShotWorldModelAgent',
                        'OnepassAgent',
                        'PolicyAgent',
                        'WebPlanningAgent',
                        'AgentModelAgent',
                    ],
                    value='AgentModelAgent',
                    interactive=True,
                    label='Agent',
                    # info='Choose your own adventure partner!',
                )
                model_selection = gr.Dropdown(
                    model_list,
                    value=default_model,
                    interactive=True,
                    label='Backend LLM',
                    # info='Choose the model you would like to use',
                )
                api_key = check_requires_key(default_model, default_api_key)

                chatbot = gr.Chatbot()
            with gr.Group():
                with gr.Row():
                    msg = gr.Textbox(container=False, show_label=False, scale=7)
                    submit = gr.Button(
                        'Submit',
                        variant='primary',
                        scale=1,
                        min_width=150,
                    )
                    submit_triggers = [msg.submit, submit.click]
            with gr.Row():
                toggle_button = gr.Button('Hide Advanced Options')
                pause_resume = gr.Button('Pause')
                clear = gr.Button('Clear')

            status = gr.Markdown('Agent Status: 🔴 Inactive')

        # with gr.Column(scale=2):
        with gr.Column(scale=2, visible=True) as visualization_column:
            # with gr.Group():
            #     start_url = 'about:blank'
            #     url = gr.Textbox(
            #         start_url, label='URL', interactive=False, max_lines=1
            #     )
            #     blank = Image.new('RGB', (1280, 720), (255, 255, 255))
            #     screenshot = gr.Image(blank, interactive=False, label='Webpage')
            #     plot = gr.Plot(go.Figure(), label='Agent Planning Process')

            with gr.Tab('Web Browser') as browser_tab:
                with gr.Group():
                    start_url = 'about:blank'
                    url = gr.Textbox(
                        start_url, label='URL', interactive=False, max_lines=1
                    )
                    blank = Image.new('RGB', (1280, 720), (255, 255, 255))
                    screenshot = gr.Image(blank, interactive=False, label='Webpage')

            with gr.Tab('Planning Process') as planning_tab:
                plot = gr.Plot(go.Figure(), label='Agent Planning Process')

            with gr.Tab('Action History') as history_tab:
                action_history = gr.Markdown('No Action Taken Yet')

    action_messages = gr.State([])
    browser_history = gr.State([(blank, start_url)])
    session = gr.State(
        OpenDevinSession(agent=default_agent, port=default_port, model=default_model)
    )
    options_visible = gr.State(True)
    toggle_button.click(
        toggle_options,
        inputs=[options_visible],
        outputs=[
            visualization_column,
            options_visible,
            toggle_button,
        ],  # advanced_options_group
        queue=False,
    )
    is_paused = gr.State(False)
    # chat_msg = msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False)
    chat_msg = gr.events.on(
        submit_triggers, user, [msg, chatbot], [msg, chatbot], queue=False
    )
    bot_msg = chat_msg.then(
        get_messages,
        [
            chatbot,
            action_messages,
            browser_history,
            session,
            status,
            agent_selection,
            model_selection,
            api_key,
        ],
        [
            chatbot,
            screenshot,
            url,
            action_messages,
            browser_history,
            session,
            status,
            clear,
            plot,
            action_history,
        ],
        concurrency_limit=10,
    )
    (
        pause_resume.click(
            pause_resume_task,
            [is_paused, session, status],
            [pause_resume, is_paused, session, status],
            queue=False,
        ).then(
            get_messages,
            [
                chatbot,
                action_messages,
                browser_history,
                session,
                status,
                agent_selection,
                model_selection,
                api_key,
            ],
            [
                chatbot,
                screenshot,
                url,
                action_messages,
                browser_history,
                session,
                status,
                clear,
                plot,
                action_history,
            ],
            concurrency_limit=10,
        )
    )
    clear.click(
        clear_page,
        [browser_history, session],
        [
            chatbot,
            pause_resume,
            is_paused,
            screenshot,
            url,
            action_messages,
            browser_history,
            session,
            status,
            plot,
            action_history,
        ],
        queue=False,
    )

    model_selection.select(
        check_requires_key, [model_selection, api_key], api_key, queue=False
    )

if __name__ == '__main__':
    # demo.queue(default_concurrency_limit=5)
    demo.queue()
    demo.launch(share=False)
