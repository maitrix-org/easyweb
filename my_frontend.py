# from openai import OpenAI
import argparse
import base64
import json
import os
import queue
import time
from datetime import datetime
from io import BytesIO

import gradio as gr  # type: ignore

# import networkx as nx
# import plotly.graph_objects as go  # type: ignore
import requests
import websocket
from bs4 import BeautifulSoup
from PIL import Image, UnidentifiedImageError

parser = argparse.ArgumentParser(description='Specify the number of backends to use.')
parser.add_argument(
    '--num-backends',
    type=int,
    default=1,
    help='The number of backends to initialize (default: 1)',
)
args = parser.parse_args()

backend_ports = [5000 + i for i in range(args.num_backends)]
default_api_key = os.environ.get('OPENAI_API_KEY')
# LINE_LEN = 100
# LABEL_LEN = 20
# WIDTH = 18
# HEIGHT = 4
# RADIUS = 1


class BackendManager:
    def __init__(self, backend_ports):
        self.available_ports = queue.Queue()
        for port in backend_ports:
            self.available_ports.put(port)

    def acquire_backend(self):
        try:
            # Wait indefinitely until a port becomes available
            port = self.available_ports.get(block=True)
            print(f'Acquired backend on port {port}')
            return port
        except Exception as e:
            print(f'Error acquiring backend: {e}')
            return None

    def release_backend(self, port):
        try:
            self.available_ports.put(port, block=True)
            print(f'Released backend on port {port}')
        except Exception as e:
            print(f'Error releasing backend: {e}')


backend_manager = BackendManager(backend_ports)
# class Node:
#     def __init__(self, state, in_action, state_info, status, reward, parent):
#         self.state = state
#         self.in_action = in_action
#         self.state_info = state_info
#         self.status = status
#         self.parent = parent
#         self.children = []
#         self.reward = reward
#         self.Q = 0.0
#         self.uct = 0.0
#         self.summary = 'Start Planning'

#     def set_summary(self, summary):
#         self.summary = summary


class FastWebSession:
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
        self.output_path = ''

        # self.figure = None

        self._reset()

    def initialize(self, as_generator=False):
        # create an output path that is global to all functions called within the FastWebSession class, so that it can be referred back to later
        # this code is copied from _close() function
        now = time.time()
        os.makedirs('frontend_logs', exist_ok=True)

        # Get current date and time
        now = datetime.now()
        # Format date and time
        formatted_now = now.strftime('%Y-%m-%d-%H:%M:%S')
        formatted_model = self.model.replace('/', '-')
        self.output_path = (
            f'frontend_logs/{formatted_now}_{self.agent}_{formatted_model}_steps.json'
        )

        self.agent_state = None
        if self.ws:
            self._reset()
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

    # def pause(self):
    #     if self.agent_state != 'running':
    #         raise ValueError('Agent not running, nothing to pause')
    #     print('Pausing')

    #     payload = {'action': 'change_agent_state', 'args': {'agent_state': 'paused'}}
    #     self.ws.send(json.dumps(payload))

    #     self.agent_state = 'pausing'

    def stop(self):
        if self.agent_state != 'running':
            raise ValueError('Agent not running, nothing to stop')
        print('Stopping')

        payload = {'action': 'change_agent_state', 'args': {'agent_state': 'stopped'}}
        self.ws.send(json.dumps(payload))

        self.agent_state = 'stopped'
        # self.save_log()
        self._reset

    # def resume(self):
    #     if self.agent_state != 'paused':
    #         raise ValueError('Agent not paused, nothing to resume')
    #     print('Resuming')

    #     payload = {'action': 'change_agent_state', 'args': {'agent_state': 'running'}}
    #     self.ws.send(json.dumps(payload))

    #     self.agent_state = 'resuming'

    def run(self, task):
        if self.agent_state not in ['init', 'running', 'pausing', 'resuming', 'paused']:
            raise ValueError(
                'Agent not initialized. Please run the initialize() method first'
            )

        if task is not None:
            payload = {'action': 'message', 'args': {'content': task}}
            self.ws.send(json.dumps(payload))

        while self.agent_state not in ['finished', 'paused', 'stopped']:
            message = self._get_message()
            self._read_message(message)

            # self._update_figure(message)

            print(self.agent_state)
            yield message
        if self.agent_state != 'stopped':
            backend_manager.release_backend(self.port)

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

    # def _update_figure(self, message):
    #     if (
    #         ('args' in message)
    #         and ('thought' in message['args'])
    #         and (message['args']['thought'].find('MCTS') != -1)
    #     ):
    #         # log_content = message['args']['thought']
    #         # self.figure = parse_and_visualize(log_content)

    #         planning_record = json.loads(message['args']['thought'])
    #         self.figure = parse_and_visualize(planning_record['full_output'])

    def _reset(self, agent_state=None):
        self.token, self.status = None, None
        self.ws, self.agent_state = None, agent_state
        # self.is_paused = False
        self.raw_messages = []
        self.browser_history = []
        self.action_history = []
        self.last_active_strategy = ''
        self.action_messages = []
        # self.figure = go.Figure()
        # session = gr.State(None)

    # changed the creation of the output to above. _close() may now be an unneccesary function, with the addition of save_log
    # def _close(self):
    #     self.save_log()
    #     self._reset()

    # partly copied from _close() function, but is activated immediately when the session moves to "finished"
    def save_log(self):
        print(f'Closing connection {self.token}')
        if self.ws:
            self.ws.close()

        if self.output_path:
            print('Saving log to', self.output_path)
            json.dump(self.raw_messages, open(self.output_path, 'w'))
        # print(self.output_path)


# opens the existing file that was saved, and adds {user_feedback: x} at the top.


def save_user_feedback(stars, session):
    path = session.output_path
    # print("other output path", path)

    if stars == 'No Action Taken Yet':
        return
    if int(stars) >= 1 and int(stars) <= 5:
        try:
            with open(path, 'r') as file:
                f = json.load(file)
            f.insert(0, {'user feedback: ': stars})
            json.dump(f, open(path, 'w'))
            print('User feedback saved!')
        except Exception:
            print("Couldn't find output log: " + str(path) + '.')


# def process_string(string, line_len):
#     preformat = string.split('\n')
#     final = []
#     for sentence in preformat:
#         formatted = []
#         for i in range(0, len(sentence), line_len):
#             splitted = sentence[i : i + line_len]
#             if (
#                 i + line_len < len(sentence)
#                 and sentence[i + line_len].isalnum()
#                 and splitted[-1].isalnum()
#             ):
#                 formatted.append(splitted + '-')
#             else:
#                 formatted.append(splitted)
#         final.append('<br>'.join(formatted))

#     return '\n'.join(final)


# def update_Q(node):
#     if len(node.children) == 0:
#         node.Q = node.reward
#         return node.reward
#     else:
#         total_Q = node.reward
#         for child in node.children:
#             if child.status != 'Init' and child.status != 'null':
#                 total_Q += update_Q(child)
#         node.Q = total_Q
#         return node.Q


# def parse_log(log_file):
#     count = 0
#     nodes = {}
#     current_node = None
#     root = None
#     chosen_node = -1
#     in_next_state = False
#     next_state = ''
#     in_state = False
#     state_info = ''

#     log_string = log_file
#     lines = log_string.strip().split('\n')

#     for line in lines:
#         if line.startswith('*State*'):
#             in_state = True
#             state_info = (
#                 state_info + process_string(line.split(': ')[1], LINE_LEN) + '<br>'
#             )

#         if (
#             in_state
#             and not (line.startswith('*State*'))
#             and not (line.startswith('*Replan Reasoning*'))
#         ):
#             state_info = state_info + process_string(line, LINE_LEN) + '<br>'

#         if line.startswith('*Replan Reasoning*'):
#             in_state = False
#             current_node = Node(count, 'null', state_info, 'Init', 0.0, None)
#             if root is None:
#                 root = current_node
#             nodes[count] = current_node
#             count += 1
#             state_info = ''

#         if line.startswith('*Strategy Candidate*'):
#             strat_info = process_string(line.split(': ')[1], LINE_LEN)

#         if line.startswith('*Summary*'):
#             summary = process_string(line.split(': ')[1], LABEL_LEN)

#         if line.startswith('*Fast Reward*'):
#             reward = float(line.split(': ')[1])
#             nodes[count] = Node(count, strat_info, 'null', 'null', reward, None)
#             nodes[count].set_summary(summary)
#             current_node.children.append(nodes[count])
#             nodes[count].parent = current_node
#             count += 1

#         if line.startswith('*Expanded Strategy*'):
#             expanded_strat = process_string(line.split(': ')[1], LINE_LEN)
#             for node_num, node in nodes.items():
#                 if node.in_action == expanded_strat:
#                     chosen_node = node_num

#         if line.startswith('*Next State*'):
#             in_next_state = True
#             next_state = (
#                 next_state + process_string(line.split(': ')[1], LINE_LEN) + '<br>'
#             )

#         if (
#             in_next_state
#             and not (line.startswith('*Next State*'))
#             and not (line.startswith('*Status*'))
#         ):
#             next_state = next_state + process_string(line, LINE_LEN) + '<br>'

#         if line.startswith('*Status*'):
#             status = process_string(line.split(': ')[1], LINE_LEN)
#             nodes[chosen_node].state_info = next_state
#             nodes[chosen_node].status = status
#             current_node = nodes[chosen_node]
#             chosen_node = -1
#             in_next_state = False
#             next_state = ''

#     update_Q(root)
#     return root, nodes


# def visualize_tree_plotly(root, nodes):
#     G = nx.DiGraph()

#     def add_edges(node):
#         for child in node.children:
#             G.add_edge(node.state, child.state)
#             add_edges(child)

#     def get_nodes_by_level(node, level, level_nodes):
#         if level not in level_nodes:
#             level_nodes[level] = []
#         level_nodes[level].append(node)
#         for child in node.children:
#             get_nodes_by_level(child, level + 1, level_nodes)

#     add_edges(root)

#     level_nodes = {}
#     get_nodes_by_level(root, 0, level_nodes)

#     highest_q_nodes = set()
#     for level, nodes_at_level in level_nodes.items():
#         highest_q_node = max(nodes_at_level, key=lambda x: x.Q)
#         highest_q_nodes.add(highest_q_node.state)

#     pos = horizontal_hierarchy_pos(G, root.state)
#     edge_x = []
#     edge_y = []

#     for edge in G.edges():
#         x0, y0 = pos[edge[0]]
#         x1, y1 = pos[edge[1]]
#         edge_x.append(x0)
#         edge_x.append(x1)
#         edge_x.append(None)
#         edge_y.append(y0)
#         edge_y.append(y1)
#         edge_y.append(None)

#     edge_trace = go.Scatter(
#         x=edge_x,
#         y=edge_y,
#         line=dict(width=2, color='white'),
#         hoverinfo='none',
#         mode='lines',
#         showlegend=False,
#     )

#     node_x = []
#     node_y = []
#     hover_texts = []
#     colors = []
#     shapes = []
#     annotations = []
#     width, height, radius = WIDTH, HEIGHT, RADIUS

#     for node in G.nodes():
#         x, y = pos[node]
#         node_x.append(x)
#         node_y.append(y)
#         hover_text = (
#             f'<b>State {node}</b><br>'
#             f'<b>Reward:</b> {nodes[node].reward}<br>'
#             f'<b>Q:</b> {nodes[node].Q}<br>'
#             f'<b>In Action:</b> {"<br>" + nodes[node].in_action if nodes[node].in_action != "null" else nodes[node].in_action}<br>'
#             f'<b>State Info:</b> {"<br>" + nodes[node].state_info if nodes[node].state_info != "null" else nodes[node].state_info}<br>'
#             f'<b>Status:</b> {nodes[node].status}'
#         )
#         hover_texts.append(hover_text)

#         annotations.append(
#             dict(
#                 x=x,
#                 y=y,
#                 text=nodes[node].summary,
#                 xref='x',
#                 yref='y',
#                 showarrow=False,
#                 font=dict(family='Arial', size=12, color='black'),
#                 align='center',
#             )
#         )

#         if node in highest_q_nodes:
#             colors.append('pink')
#         else:
#             colors.append('#FFD700')

#         custom_path = (
#             f'M{x - width / 2 + radius},{y - height / 2} '
#             f'L{x + width / 2 - radius},{y - height / 2} '
#             f'Q{x + width / 2},{y - height / 2} {x + width / 2},{y - height / 2 + radius} '
#             f'L{x + width / 2},{y + height / 2 - radius} '
#             f'Q{x + width / 2},{y + height / 2} {x + width / 2 - radius},{y + height / 2} '
#             f'L{x - width / 2 + radius},{y + height / 2} '
#             f'Q{x - width / 2},{y + height / 2} {x - width / 2},{y + height / 2 - radius} '
#             f'L{x - width / 2},{y - height / 2 + radius} '
#             f'Q{x - width / 2},{y - height / 2} {x - width / 2 + radius},{y - height / 2} '
#             f'Z'
#         )

#         shapes.append(
#             dict(
#                 xref='x',
#                 yref='y',
#                 type='path',
#                 path=custom_path,
#                 fillcolor='pink' if node in highest_q_nodes else '#FFD700',
#                 line_color='black',
#             )
#         )

#     node_trace = go.Scatter(
#         x=node_x,
#         y=node_y,
#         mode='markers',
#         hoverinfo='text',
#         text=hover_texts,
#         hoverlabel=dict(font=dict(size=16)),
#         marker=dict(showscale=False, color=colors, size=0),
#         showlegend=False,
#     )

#     agent_choice_trace = go.Scatter(
#         x=[None],
#         y=[None],
#         mode='markers',
#         marker=dict(
#             size=10,
#             color='pink',
#             line=dict(width=2),
#         ),
#         showlegend=True,
#         name='Agent Choice',
#     )

#     candidate_trace = go.Scatter(
#         x=[None],
#         y=[None],
#         mode='markers',
#         marker=dict(
#             size=10,
#             color='#FFD700',
#             line=dict(width=2),
#         ),
#         showlegend=True,
#         name='Candidate',
#     )

#     fig = go.Figure(
#         data=[
#             edge_trace,
#             node_trace,
#             agent_choice_trace,
#             candidate_trace,
#         ],  # label_trace,
#         layout=go.Layout(
#             title='Agent Thinking Process',
#             titlefont_size=16,
#             showlegend=True,
#             hovermode='closest',
#             margin=dict(b=20, l=5, r=5, t=40),
#             xaxis=dict(
#                 showgrid=False,
#                 zeroline=False,
#                 visible=False,
#                 range=[-10, 80],
#             ),
#             yaxis=dict(
#                 showgrid=False,
#                 zeroline=False,
#                 visible=False,
#                 range=[0, 40],
#             ),
#             width=920,
#             height=540,
#             shapes=shapes,
#             annotations=annotations,
#             dragmode='pan',
#         ),
#     )
#     fig.update_layout(
#         modebar_remove=[
#             'zoom2d',
#             'zoomIn2d',
#             'zoomOut2d',
#             'autoScale2d',
#             'resetScale2d',
#             'lasso2d',
#             'select2d',
#         ]
#     )
#     return fig


# def horizontal_hierarchy_pos(G, root, height=110, hor_gap=20.0, hor_loc=0, ycenter=20):
#     pos = _horizontal_hierarchy_pos(G, root, height, hor_gap, hor_loc, ycenter)
#     return pos


# def _horizontal_hierarchy_pos(
#     G,
#     root,
#     height=110,
#     hor_gap=20.0,
#     hor_loc=0,
#     ycenter=20.0,
#     pos=None,
#     parent=None,
#     parsed=None,
# ):
#     if pos is None:
#         pos = {root: (hor_loc, ycenter)}
#     if parsed is None:
#         parsed = []

#     pos[root] = (hor_loc, ycenter)
#     children = list(G.neighbors(root))
#     if not isinstance(G, nx.DiGraph) and parent is not None:
#         children.remove(parent)
#     if len(children) != 0:
#         dy = height / len(children)
#         if dy < HEIGHT:
#             dy = HEIGHT
#         else:
#             nexty = ycenter - height / 2 - dy / 2

#         for child in children:
#             nexty += dy
#             pos = _horizontal_hierarchy_pos(
#                 G,
#                 child,
#                 height=dy,
#                 hor_gap=hor_gap,
#                 hor_loc=hor_loc + hor_gap,
#                 ycenter=nexty,
#                 pos=pos,
#                 parent=root,
#                 parsed=parsed,
#             )
#     return pos


# def parse_and_visualize(log_file):
#     root, nodes = parse_log(log_file)
#     fig = visualize_tree_plotly(root, nodes)
#     return fig


def get_status(agent_state):
    if agent_state == 'loading':
        status = 'Agent Status: 🟡 Loading'
    elif agent_state == 'init':
        status = 'Agent Status: 🟢 Initialized'
    elif agent_state == 'running':
        status = 'Agent Status: 🟢 Running'
    # elif agent_state == 'pausing':
    #     status = 'Agent Status: 🟢 Pausing'
    # elif agent_state == 'paused':
    #     status = 'Agent Status: 🟡 Paused'
    # elif agent_state == 'resuming':
    #     status = 'Agent Status: 🟡 Resuming'
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
    options_visible,
):
    # upvote = gr.Button('👍 Upvote', interactive=(session.agent_state == 'finished'))
    # downvote = gr.Button('👎 Downvote', interactive=(session.agent_state == 'finished'))
    model_selection = model_display2name[model_selection]
    user_message = None
    if len(chat_history) > 0:
        # check to see if user has sent a message previously
        if chat_history[-1]['role'] == 'user':
            user_message = chat_history[-1]['content']

    # Initialize a new session if it doesn't exist
    if session is None or session.agent_state in ['finished', 'paused']:
        new_session = FastWebSession(
            agent=agent_selection,
            port=backend_manager.acquire_backend(),
            model=model_selection,
            api_key=api_key if model_requires_key[model_selection] else default_api_key,
        )
        session = new_session
    stop_flag = session.agent_state is not None and session.agent_state == 'stopped'

    if (
        session.agent_state is None
        or session.agent_state in ['paused', 'finished', 'stopped']
    ) and user_message is None:
        clear = gr.Button('🗑️ Clear', interactive=True)
        status = get_status(session.agent_state)

        screenshot, url = browser_history[-1]
        feedback = gr.Button('Submit Feedback', visible=False)
        stars = gr.Textbox(elem_id='dummy_textbox', value=-1)
        submit = gr.Button(
            'Submit',
            variant='primary',
            scale=1,
            min_width=150,
            visible=session.agent_state != 'running',
        )
        stop = gr.Button('Stop', visible=session.agent_state == 'running')
        # if session.figure:
        #     figure = session.figure
        # else:
        #     figure = go.Figure()

        # action_history = get_action_history_markdown(session.action_history)
        # action_history = action_history if action_history else 'No Action Taken Yet'
        yield (
            chat_history,
            screenshot,
            url,
            action_messages,
            browser_history,
            session,
            status,
            clear,
            feedback,
            stars,
            options_visible,
            # upvote,
            # downvote,
            submit,
            stop,
        )
    else:
        # make sure that the buttons and stars aren't shown yet
        clear = gr.Button('🗑️ Clear', interactive=False)
        feedback = gr.Button('Submit Feedback')
        stars = gr.Textbox(elem_id='dummy_textbox', value=-1)
        if session.agent_state not in [
            'init',
            'running',
            'pausing',
            'resuming',
        ]:
            if stop_flag:
                stop_flag = False
                clear = gr.Button('Clear', interactive=False)
                screenshot, url = browser_history[-1]
                session._reset()
                chat_history = chat_history[-1:]
                action_messages = []

                submit = gr.Button(
                    'Submit',
                    variant='primary',
                    scale=1,
                    min_width=150,
                    visible=session.agent_state != 'running',
                )
                stop = gr.Button('Stop', visible=session.agent_state == 'running')
                #!!! options_visible
                yield (
                    chat_history,
                    screenshot,
                    url,
                    [],
                    browser_history,
                    session,
                    # None,
                    status,
                    clear,
                    # go.Figure(),
                    'No Action Taken Yet',
                    options_visible,
                    submit,
                    stop,
                )

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

                submit = gr.Button(
                    'Submit',
                    variant='primary',
                    scale=1,
                    min_width=150,
                    visible=session.agent_state != 'running',
                )
                stop = gr.Button('Stop', visible=session.agent_state == 'running')
                # if session.figure:
                #     figure = session.figure
                # else:
                #     figure = go.Figure()

                # action_history = get_action_history_markdown(session.action_history)
                # action_history = (
                #     action_history if action_history else 'No Action Taken Yet'
                # )
                yield (
                    chat_history,
                    screenshot,
                    url,
                    action_messages,
                    browser_history,
                    session,
                    status,
                    clear,
                    feedback,
                    stars,
                    options_visible,
                    # upvote,
                    # downvote,
                    submit,
                    stop,
                )

        website_counter = 0
        message_list = []
        for message in session.run(user_message):
            # if "content" in message.keys():
            # print("MESSAGEEEE", message['message'])
            message_list.append(message['message'])
            # else:
            # print("No content:", message)
            # this is so that the swingout (hide/show browser) will only pop out right when the user submits something
            if website_counter == 1:
                options_visible = True
            # only enable the stars and feedback if the session.agent_state == finished
            clear = gr.Button(
                '🗑️ Clear', interactive=(session.agent_state == 'finished')
            )
            feedback = gr.Button(
                'Submit Feedback', visible=(session.agent_state == 'finished')
            )
            # upvote = gr.Button('👍 Upvote', interactive=(session.agent_state == 'finished'))
            # downvote = gr.Button('👎 Downvote', interactive=(session.agent_state == 'finished'))
            if message.get('action', '') in ['message', 'finish']:
                # chat_history.append(
                #     gr.ChatMessage(role='assistant', content=message.get('message', '(Empty Message)'))
                # )
                chat_history.append(gr.ChatMessage(role='assistant', content=''))
                assistant_message = message.get('message', '(Empty Message)')
                assistant_message_chars = []
                for i, char in enumerate(assistant_message):
                    assistant_message_chars.append(char)
                    updated_message = ''.join(assistant_message_chars)
                    if (i + 1) % 5 == 0 or i == len(assistant_message) - 1:
                        chat_history[-1] = gr.ChatMessage(
                            role='assistant', content=updated_message
                        )
                        time.sleep(0.01)

                        yield (
                            chat_history,
                            screenshot,
                            url,
                            action_messages,
                            browser_history,
                            session,
                            status,
                            clear,
                            feedback,
                            stars,
                            options_visible,
                            # upvote,
                            # downvote,
                            submit,
                            stop,
                        )
            elif (
                session.agent == 'ReasonerWebAgent'
                and message.get('action', '') == 'browse_interactive'
                and message.get('args', {}).get('thought', '')
            ):
                full_output_dict = json.loads(message['args']['thought'])
                plan = full_output_dict.get('plan')
                if plan:
                    # chat_history.append(
                    #     gr.ChatMessage(role='assistant', content=plan)
                    # )
                    chat_history.append(gr.ChatMessage(role='assistant', content=''))
                    assistant_message = plan
                    assistant_message_chars = []
                    for i, char in enumerate(assistant_message):
                        assistant_message_chars.append(char)
                        updated_message = ''.join(assistant_message_chars)
                        if (i + 1) % 5 == 0 or i == len(assistant_message) - 1:
                            chat_history[-1] = gr.ChatMessage(
                                role='assistant', content=updated_message
                            )
                            time.sleep(0.01)

                            yield (
                                chat_history,
                                screenshot,
                                url,
                                action_messages,
                                browser_history,
                                session,
                                status,
                                clear,
                                feedback,
                                stars,
                                options_visible,
                                # upvote,
                                # downvote,
                                submit,
                                stop,
                            )
            elif (
                session.agent == 'BrowsingAgent'
                and message.get('action', '') == 'browse_interactive'
                and message.get('args', {}).get('thought', '')
            ):
                thought = message['args']['thought']
                chat_history.append(gr.ChatMessage(role='assistant', content=thought))
            # Stream End
            # if user_message:
            #     chat_history.append(gr.ChatMessage(role='user', content=user_message))
            if session.agent_state == 'finished':  # Stream End
                # Add the last output message once it is finished
                # if len(message_list) >= 3:
                #     if message_list[-3] != '':
                #         chat_history.append(
                #             gr.ChatMessage(role='assistant', content=message_list[-3])
                #         )
                #     if message_list[-2] != '':
                #         chat_history.append(
                #             gr.ChatMessage(role='assistant', content=message_list[-2])
                #         )
                #     if message_list[-2] == '' and message_list[-3] == '':
                #         chat_history.append(
                #             gr.ChatMessage(
                #                 role='assistant',
                #                 content='Task Complete! How can I assist you next?',
                #             )
                #         )

                # Stream End: Handle the end-of-session UI updates
                stars = gr.Textbox(elem_id='dummy_textbox', value=0)
                session.save_log()
                # stars = gr.Textbox(elem_id='dummy_textbox', value=0)
                # session.save_log()
            status = get_status(session.agent_state)
            while len(session.action_messages) > len(action_messages):
                diff = len(session.action_messages) - len(action_messages)
                action_messages.append(session.action_messages[-diff])
                # create sites_visited list from browser_history, use it in display history
                sites_visited = []
                website_counter = 0
                for item in browser_history:
                    website_counter += 1
                    sites_visited.append(item[1])

                chat_history = display_history(
                    chat_history, sites_visited, action_messages
                )
            while len(session.browser_history) > (len(browser_history) - 1):
                diff = len(session.browser_history) - (len(browser_history) - 1)
                browser_history.append(session.browser_history[-diff])
            screenshot, url = browser_history[-1]
            # if session.figure:
            #     figure = session.figure
            # else:
            #     figure = go.Figure()

            # action_history = get_action_history_markdown(session.action_history)
            # action_history = action_history if action_history else 'No Action Taken Yet'
            submit = gr.Button(
                'Submit',
                variant='primary',
                scale=1,
                min_width=150,
                visible=session.agent_state != 'running',
            )
            stop = gr.Button('Stop', visible=session.agent_state == 'running')
            yield (
                chat_history,
                screenshot,
                url,
                action_messages,
                browser_history,
                session,
                status,
                clear,
                feedback,
                stars,
                options_visible,
                # upvote,
                # downvote,
                submit,
                stop,
            )


def clear_page(browser_history, session, feedback):
    feedback = gr.Button('Submit Feedback', visible=False)
    browser_history = browser_history[:1]
    current_screenshot, current_url = browser_history[-1]

    if session is not None:
        session._reset()
        status = get_status(session.agent_state)
    else:
        status = get_status(None)
    # pause_resume = gr.Button("Pause", interactive=False)
    return (
        None,
        # 'Pause',
        # False,
        current_screenshot,
        current_url,
        [],
        browser_history,
        # None,  # Reset session to None
        session,
        status,
        feedback,
    )


def check_requires_key(model_selection, api_key):
    model_real_name = model_display2name[model_selection]
    requires_key = model_requires_key[model_real_name]
    if requires_key:
        api_key = gr.Textbox(
            api_key,
            label='API Key',
            placeholder='Your API Key',
            visible=True,
            max_lines=2,
        )
    else:
        api_key = gr.Textbox(
            api_key,
            label='API Key',
            placeholder='Your API Key',
            visible=False,
            max_lines=2,
        )
    return api_key


# def pause_resume_task(is_paused, session, status):
#     if not is_paused and session.agent_state == 'running':
#         session.pause()
#         is_paused = True
#     elif is_paused and session.agent_state == 'paused':
#         session.resume()
#         is_paused = False

#     button = 'Resume' if is_paused else 'Pause'
#     status = get_status(session.agent_state)
#     return button, is_paused, session, status


# for display history, this is the dropdown box that shows up
def display_history(history, messages_history, action_messages):
    # parse everything into a string so that it is in one message instead of multiple, for the dropdown effect
    links_string = ''
    # count total links for the title
    total_links = 0
    # fix the issue of multiple titles in a row
    previous_titles = ['']
    for message in messages_history:
        # try and get the title, if it doesn't work, just use the previous message
        try:
            url = message
            # print('URL', url)
            response = requests.get(url)
            soup = BeautifulSoup(response.content, 'html.parser')
            title = soup.title.string
        except Exception:
            title = message
        # check for duplicate entries in a row
        if title != previous_titles[-1]:
            links_string += f'<a href="{message}" style="float: left;" target="_blank">{title}</a>\n'
            previous_titles.append(title)
            total_links += 1
    # add total links to title
    if total_links == 1:
        history_title = 'Searched 1 site'
    else:
        history_title = 'Searched ' + str(total_links) + ' sites'
    # replace the last message unless it is the user's message
    # sometimes represented as a dictionary, sometimes as a gr.ChatMessage() class. Not really sure when is which.
    # if it is a gr.ChatMessage(), need to reference differently from dictionary
    if 'goto' in action_messages[-1]:
        history_title = 'Browsing ' + message + '...'

    # The links_string should be appended to the last message that does not come from the assistant
    # If something like the links_string already exists, replace it with the new one
    # Otherwise, append it to history
    last_non_assistant_message_idx = 0
    for i, chat_message in enumerate(history):
        if not isinstance(chat_message, dict) and chat_message.role != 'assistant':
            last_non_assistant_message_idx = i
        elif isinstance(chat_message, dict) and chat_message['role'] != 'assistant':
            last_non_assistant_message_idx = i

    # links_string_idx = last_non_assistant_message_idx + 1
    links_string_idx = last_non_assistant_message_idx + 1
    # after_last_nonassistant_message_exists and this message is already links_string
    # if links_string_idx < len(history):
    #     print(history[links_string_idx])
    if links_string_idx < len(history) and (
        (
            isinstance(history[links_string_idx], dict)
            and isinstance(history[links_string_idx]['metadata'], dict)
            and history[links_string_idx]['metadata'].get('title')
        )
        or (
            not isinstance(history[links_string_idx], dict)
            and isinstance(history[links_string_idx].metadata, dict)
            and history[links_string_idx].metadata.get('title')
        )
    ):
        history[links_string_idx] = gr.ChatMessage(
            role='assistant',
            content=(links_string),
            metadata={'title': history_title},
        )
    else:
        history.insert(
            links_string_idx,
            gr.ChatMessage(
                role='assistant',
                content=(links_string),
                metadata={'title': history_title},
            ),
        )

    # if not isinstance(history[-1], dict):
    #     if history[-1].metadata is None or history[-1].role != 'assistant':
    #         history.append(
    #             gr.ChatMessage(
    #                 role='assistant',
    #                 content=(links_string),
    #                 metadata={'title': history_title},
    #             )
    #         )
    #     else:
    #         history[-1] = gr.ChatMessage(
    #             role='assistant',
    #             content=(links_string),
    #             metadata={'title': history_title},
    #         )
    # # this else exists just in case it is a dictionary:
    # else:
    #     if history[-1]['metadata'] is None or history[-1]['role'] != 'assistant':
    #         history.append(
    #             gr.ChatMessage(
    #                 role='assistant',
    #                 content=(links_string),
    #                 metadata={'title': history_title},
    #             )
    #         )
    #     else:
    #         history[-1] = gr.ChatMessage(
    #             role='assistant',
    #             content=(links_string),
    #             metadata={'title': history_title},
    #         )

    # return history returns the chatbot itself
    return history


# replaced previous function called user() which basically processes the user input into the gr.Chatbot class
def process_user_message(user_message, history):
    # return '', history + [[user_message, None]]
    if not user_message.strip():
        return '', history
    chat_message = gr.ChatMessage(role='user', content=user_message)
    history.append(chat_message)
    return '', history


def stop_task(session):
    if session.agent_state == 'running':
        session.stop()
        # session._reset()
    # clear everything on resubmit
    # save to session logs
    # pull and merge
    # when its running, disable submit, enable stop like chatgpt
    # change submit to stop button when running?
    # dont submit if nothing in box
    # session.save_log()
    status = get_status(session.agent_state)
    clear = gr.Button('Clear', interactive=True)
    return session, status, clear
    # return None, status, clear


# toggle hiding and showing the browser. IfClick is basically because I call this function sometimes without the user specifically clicking on the button.
def toggle_options(visible, ifClick):
    if ifClick:
        new_visible = not visible
    else:
        new_visible = visible
    toggle_text = '🔍 Hide Browser' if new_visible else '🔍 Show Browser'
    return (
        gr.update(visible=new_visible),
        new_visible,
        gr.update(value=toggle_text),
    )


# if __name__ == '__main__':
#     # demo.queue(default_concurrency_limit=5)
#     demo.queue()
#     demo.launch(share=True)


current_dir = os.path.dirname(__file__)
print(os.path.dirname(__file__))

# default_port = 5000
# with open(os.path.join(current_dir, 'Makefile')) as f:
#     while True:
#         line = f.readline()
#         if 'BACKEND_PORT' in line:
#             default_port = int(line.split('=')[1].strip())
#             break
#         if not line:
#             break
# default_agent = 'WorldModelAgent'
# default_agent = 'AgentModelAgent'
# default_agent = 'ModularWebAgent'
default_agent = 'ReasonerWebAgent'

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

# default_model = model_list[0]
default_model = 'gpt-4o'
for model, cfg in model_port_config.items():
    if cfg.get('default', None):
        default_model = cfg.get('display_name', model)
        break

# default_api_key = os.environ.get('OPENAI_API_KEY')
current_dir = os.path.dirname(__file__)

with open(os.path.join(current_dir, 'default_api_key.txt'), 'r') as fr:
    default_api_key = fr.read().strip()

# Define the custom HTML for the 5-star rating system
html_content = """
<div style="display: none;" id="feedback" class = "block svelte-5y6bt2 padded">
    <h2>How did we do?</h2>
    <div id="stars" style="font-size: 2rem; color: #ffd700;">
        <span onclick="setRating(1)">★</span>
        <span onclick="setRating(2)">★</span>
        <span onclick="setRating(3)">★</span>
        <span onclick="setRating(4)">★</span>
        <span onclick="setRating(5)">★</span>
    </div>
    <h3 id="confirmation-text"></h3>
</div>
"""

# JavaScript to handle the star rating functionality
js_code = """
async () => {
    let currentRating = -1; // To store the current rating
    let submitted = false;

    globalThis.setRating = (stars) => {
        currentRating = stars; // Update the current rating
        // document.getElementById("rating-text").innerText = `Your rating: ${stars} stars`;

        // Highlight stars up to the selected rating
        let starElements = document.getElementById("stars").children;
        if (!submitted){
            for (let i = 0; i < starElements.length; i++) {
                starElements[i].style.color = i < stars ? "#ffd700" : "gray";
            }
        }
    }

    //this is triggered when the submit button is clicked
    globalThis.submitRating = () => {
        const confirmationText = document.getElementById("confirmation-text");
        if (currentRating > 0) {
            confirmationText.innerText = `Thank you for your feedback.`;
            document.getElementById("submit-button").style.display = "none";
            console.log(currentRating);
        } else {
            confirmationText.innerText = "Please select a rating before submitting.";
        }
        submitted = true;
        return currentRating;
    }

    //show the stars by setting their display to inline-block
    globalThis.showStars = () => {
        document.getElementById("feedback").style.display = "inline-block";
        document.getElementById("feedback").scrollIntoView({ behavior: "smooth" });
    }

    //hide the stars by setting their display to none
    globalThis.hideStars = () => {
        document.getElementById("feedback").style.display = "none";
    }

}
"""

# different so that the function can be called by gradio elements in the python code
get_rating = """
function(){
    let currentRating = submitRating();
    return currentRating;
}
"""

# make this in python for the clear button
hide_stars = """
function(){
    hideStars();
}
"""
# random css for other formatting and whatnot
# background-color: #f4f4f7;
css = """
#submit-button{
    width: 200px;
}
#feedback{
    padding-left: 20px;
    max-width: 230px;
    padding-bottom: 20px;
}
#confirmation-text{
    margin-top: 8px;
}
"""

# def vote(upvote):
#     if upvote:
#         print('Upvoted!')
#     else:
#         print('Downvoted.')

with gr.Blocks(css=css) as demo:
    action_messages = gr.State([])
    # session = gr.State(
    #     FastWebSession(agent=default_agent, port=default_port, model=default_model)
    # )
    session = gr.State(None)
    title = gr.Markdown('# 🚀 OpenQ: An Open-Source LLM-Powered Web Agent')
    # header = gr.Markdown('''## How it works:''')
    tutorial1 = gr.Markdown("""- 🔑 **Choose** an **Agent**, an **LLM**, and provide an **API Key** if required.
                            - 💬 **Ask the Agent** to perform advanced web-related tasks. **For example:**
                            - "What were box office values of the Star Wars films in the prequel and sequel trilogies?"
                            - "Can you search for a round-trip flight from Los Angeles to Tokyo in business class?"
                            - ✍️ **Share your feedback** using the form below once the Agent completes its task!""")

    privacy_title = gr.Markdown(
        """❗️**Important: Data submitted may be used for research purposes. Please avoid uploading confidential or personal information. User prompts and feedback are logged.**"""
    )
    with gr.Row(equal_height=False):
        with gr.Column(scale=2):
            with gr.Group():
                with gr.Row():
                    agent_selection = gr.Dropdown(
                        [
                            # 'DummyWebAgent',
                            'BrowsingAgent',
                            # 'WorldModelAgent',
                            # 'NewWorldModelAgent',
                            # 'FewShotWorldModelAgent',
                            # 'OnepassAgent',
                            # 'PolicyAgent',
                            # 'WebPlanningAgent',
                            # 'AgentModelAgent',
                            # 'ModularWebAgent',
                            'ReasonerWebAgent',
                        ],
                        value=default_agent,
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

            # change to be type=messages, which converts the messages inputted from tuples to gr.ChatMessage class
            chatbot = gr.Chatbot(type='messages', height=320)
            with gr.Row():
                rating_html = gr.HTML(html_content)
                # dummy textbox that isn't shown in order to store the value which can be referred to by both HTML and gradio
                stars = gr.Textbox(elem_id='dummy_textbox', value=-1, visible=False)
                # when the stars dummy textbox is changed, trigger all of this
                stars.change(None, None, None, js='() => {showStars()}')
                stars.change(save_user_feedback, inputs=[stars, session])
                # Load the JavaScript code to initialize the interactive stars
                demo.load(None, None, None, js=js_code)
            # feedback button, in a different row.
            feedback = gr.Button(
                'Submit Feedback',
                variant='secondary',
                elem_id='submit-button',
                visible=False,
            )
            feedback.click(None, inputs=None, outputs=stars, js=get_rating)
            with gr.Group():
                with gr.Row():
                    msg = gr.Textbox(container=False, show_label=False, scale=7)

                    submit = gr.Button(
                        'Submit',
                        variant='primary',
                        scale=1,
                        min_width=150,
                    )
                    stop = gr.Button('Stop', visible=False)
                    submit_triggers = [msg.submit, submit.click]
        with gr.Column(scale=4, visible=False) as visualization_column:
            # with gr.Group():
            #     start_url = 'about:blank'
            #     url = gr.Textbox(
            #         start_url, label='URL', interactive=False, max_lines=1
            #     )
            #     blank = Image.new('RGB', (1280, 720), (255, 255, 255))
            #     screenshot = gr.Image(blank, interactive=False, label='Webpage')
            #     plot = gr.Plot(go.Figure(), label='Agent Planning Process')
            with gr.Group():
                # starting url can be changed
                start_url = 'about:blank'
                url = gr.Textbox(start_url, label='URL', interactive=False, max_lines=1)
                blank = Image.new('RGB', (1280, 720), (255, 255, 255))
                screenshot = gr.Image(blank, interactive=False, label='Webpage')

    with gr.Row():
        toggle_button = gr.Button('🔍 Show Browser')
        # pause_resume = gr.Button('Pause')
        clear = gr.Button('Clear')
    # with gr.Row():
    #     rating_html = gr.HTML(html_content)
    #     # dummy textbox that isn't shown in order to store the value which can be referred to by both HTML and gradio
    #     stars = gr.Textbox(elem_id='dummy_textbox', value=-1, visible=False)
    #     # when the stars dummy textbox is changed, trigger all of this
    #     stars.change(None, None, None, js='() => {showStars()}')
    #     stars.change(save_user_feedback, inputs=[stars, session])
    #     # Load the JavaScript code to initialize the interactive stars
    #     demo.load(None, None, None, js=js_code)
    # # feedback button, in a different row.
    # feedback = gr.Button(
    #     'Submit Feedback', variant='secondary', elem_id='submit-button', visible=False
    # )
    # feedback.click(None, inputs=None, outputs=stars, js=get_rating)
    status = gr.Markdown('Agent Status: 🔴 Inactive')
    browser_history = gr.State([(blank, start_url)])
    options_visible = gr.State(False)
    # upvote.click(vote, inputs=[gr.State(True)])
    # downvote.click(vote, inputs=[gr.State(False)])
    options_visible.change(
        toggle_options,
        inputs=[options_visible, gr.State(False)],
        outputs=[
            visualization_column,
            options_visible,
            toggle_button,
        ],
        queue=False,
    )
    toggle_click = toggle_button.click(
        toggle_options,
        inputs=[options_visible, gr.State(True)],
        outputs=[
            visualization_column,
            options_visible,
            toggle_button,
        ],
        queue=False,
    )
    # is_paused = gr.State(False)
    # chat_msg = msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False)
    chat_msg = gr.events.on(
        submit_triggers,
        process_user_message,
        [msg, chatbot],
        [msg, chatbot],
        queue=False,
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
            options_visible,
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
            feedback,
            stars,
            options_visible,
            # upvote,
            # downvote,
            submit,
            stop,
        ],
        concurrency_limit=args.num_backends,
    )
    # (
    #     pause_resume.click(
    #         pause_resume_task,
    #         [is_paused, session, status],
    #         [pause_resume, is_paused, session, status],
    #         queue=False,
    #     ).then(
    #         get_messages,
    #         [
    #             chatbot,
    #             action_messages,
    #             browser_history,
    #             session,
    #             status,
    #             agent_selection,
    #             model_selection,
    #             api_key,
    #         ],
    #         [
    #             chatbot,
    #             screenshot,
    #             url,
    #             action_messages,
    #             browser_history,
    #             session,
    #             status,
    #             clear,
    #             feedback,
    #             stars,
    #         ],
    #         concurrency_limit=10,
    #     )
    # )
    (
        stop.click(
            stop_task,
            [session],
            [session, status, clear],
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
                options_visible,
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
                feedback,
                stars,
                options_visible,
                # upvote,
                # downvote,
                submit,
                stop,
            ],
            concurrency_limit=args.num_backends,
        )
    )
    (
        clear.click(
            clear_page,
            [browser_history, session, feedback],
            [
                chatbot,
                # pause_resume,
                # is_paused,
                screenshot,
                url,
                action_messages,
                browser_history,
                session,
                status,
                feedback,
            ],
            queue=False,
        ).then(fn=None, js=hide_stars)
    )
    model_selection.select(
        check_requires_key, [model_selection, api_key], api_key, queue=False
    )


if __name__ == '__main__':
    # demo.queue(default_concurrency_limit=5)
    demo.queue()
    demo.launch(share=True)
