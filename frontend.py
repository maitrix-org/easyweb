import argparse
import base64
import json
import os
import queue
import time
from datetime import datetime
from io import BytesIO

import gradio as gr
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
parser.add_argument('--ip', type=str, default=None, help='server name for public demo')
parser.add_argument(
    '--port', type=int, default=None, help='server port for public demo'
)
parser.add_argument(
    '--ssl-certfile', type=str, default=None, help='path to SSL certfile'
)
parser.add_argument('--ssl-keyfile', type=str, default=None, help='path to SSL keyfile')
parser.add_argument(
    '--ssl-verify',
    type=bool,
    default=False,
    help='whether to run certificate validation',
)
args = parser.parse_args()

backend_ports = [5000 + i for i in range(args.num_backends)]
default_api_key = 'sk-123'
global_sessions = dict()


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
        if port in self.available_ports.queue:
            print(f'{port} already available')
        else:
            self.available_ports.put(port, block=True)
            print(f'Released backend on port {port}')


backend_manager = BackendManager(backend_ports)


class EasyWebSession:
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

        self._reset()

    def initialize(self, as_generator=False):
        # create an output path that is global to all functions called within the
        # EasyWebSession class, so that it can be referred back to later
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

    def stop(self):
        # if self.agent_state != 'running':
        #     raise ValueError('Agent not running, nothing to stop')
        print('Stopping')

        payload = {'action': 'change_agent_state', 'args': {'agent_state': 'stopped'}}
        self.ws.send(json.dumps(payload))

        self.agent_state = 'stopped'
        self._reset

    def run(self, task, request: gr.Request):
        if self.agent_state not in ['init', 'running']:
            raise ValueError(
                'Agent not initialized. Please run the initialize() method first'
            )

        if task is not None:
            payload = {'action': 'message', 'args': {'content': task}}
            self.ws.send(json.dumps(payload))
        try:
            while self.agent_state not in ['finished', 'stopped']:
                message = self._get_message()
                self._read_message(message)

                print(self.agent_state)
                yield message
        finally:
            if request.session_hash in global_sessions.keys():
                backend_manager.release_backend(self.port)
                del global_sessions[request.session_hash]

    def _get_message(self):
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
        return message

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

    def _reset(self, agent_state=None):
        self.token, self.status = None, None
        self.ws, self.agent_state = None, agent_state
        self.raw_messages = []
        self.browser_history = []
        self.action_history = []
        self.last_active_strategy = ''
        self.action_messages = []

    def save_log(self):
        print(f'Closing connection {self.token}')
        if self.ws:
            self.ws.close()

        if self.output_path:
            print('Saving log to', self.output_path)
            json.dump(self.raw_messages, open(self.output_path, 'w'))

    def save_user_feedback(self, vote):
        path = self.output_path
        if vote:
            stars = 1
        else:
            stars = 0
        try:
            if os.path.exists(path):
                with open(path, 'r') as file:
                    f = json.load(file)
            else:
                f = self.raw_messages
            f.insert(0, {'user feedback: ': stars})
            json.dump(f, open(path, 'w'))
            print('User feedback saved!')
        except Exception:
            print("Couldn't find output log: " + str(path) + '.')


def get_status(agent_state):
    if agent_state == 'loading':
        status = 'Agent Status: 🟡 Loading'
    elif agent_state == 'init':
        status = 'Agent Status: 🟢 Initialized'
    elif agent_state == 'running':
        status = 'Agent Status: 🟢 Running'
    elif agent_state == 'finished':
        status = 'Agent Status: 🟢 Finished'
    elif agent_state == 'stopped':
        status = 'Agent Status: 🔴 Stopped'
    elif agent_state is None:
        status = 'Agent Status: 🔴 Inactive'
    else:
        status = f'Agent Status: 🔴 {agent_state}'

    return f'<font size="4"> {status} </font>'


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
    request: gr.Request,
):
    agent_selection = agent_display2class[agent_selection]
    model_selection = model_display2name[model_selection]
    model_key_filename = model_name2keypath.get(model_selection)
    if model_key_filename:
        model_key_filepath = os.path.join(os.getcwd(), model_key_filename)
        with open(model_key_filepath, 'r') as f:
            api_key = f.read().strip()

    print(api_key)

    user_message = None
    thinking_flag = False
    if len(chat_history) > 0:
        # check to see if user has sent a message previously
        if chat_history[-1]['role'] == 'user' and chat_history[-1]['content'] != '':
            user_message = chat_history[-1]['content']
            thinking_flag = True  # so i can know if i need to remove the last message

    # stop_flag = False
    browser_starting_flag = False
    # Initialize a new session if it doesn't exist
    if (
        session is None
        or session.agent_state is None
        or session.agent_state in ['finished', 'stopped']
    ):
        loading_message = gr.ChatMessage(
            role='assistant', content='⏳ Browser Starting...'
        ).__dict__
        chat_history.append(loading_message)
        browser_starting_flag = True

        if session is not None and session.agent_state is not None:
            stop_flag = session.agent_state == 'stopped'

        new_session = EasyWebSession(
            agent=agent_selection,
            port=backend_manager.acquire_backend(),
            model=model_selection,
            # api_key=api_key if model_requires_key[model_selection] else default_api_key,
            api_key=api_key,
        )
        session = new_session
        if request.session_hash not in global_sessions.keys():
            global_sessions[request.session_hash] = session
        if user_message is None:
            backend_manager.release_backend(session.port)
            del global_sessions[request.session_hash]
            session.agent_state = None
            chat_history = chat_history[:-1]

    if thinking_flag and not browser_starting_flag:
        loading_message = gr.ChatMessage(
            role='assistant', content='⏳ Thinking...'
        ).__dict__
        chat_history.append(loading_message)

    if (
        session.agent_state is None or session.agent_state in ['finished', 'stopped']
    ) and user_message is None:
        clear = gr.Button('🗑️ Clear', interactive=True)
        status = get_status(session.agent_state)

        screenshot, url = browser_history[-1]
        upvote = gr.Button('👍 Good Response', interactive=False)
        downvote = gr.Button('👎 Bad Response', interactive=False)
        submit = gr.Button(
            'Submit',
            variant='primary',
            scale=1,
            min_width=150,
            visible=session.agent_state != 'running',
        )
        stop = gr.Button(
            'Stop', scale=1, min_width=150, visible=session.agent_state == 'running'
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
            options_visible,
            upvote,
            downvote,
            submit,
            stop,
        )
    else:
        clear = gr.Button('🗑️ Clear', interactive=False)
        upvote = gr.Button('👍 Good Response', interactive=False)
        downvote = gr.Button('👎 Bad Response', interactive=False)
        if session.agent_state not in [
            'init',
            'running',
        ]:
            # if stop_flag:
            #     print(chat_history, 'chat history bruh')
            #     stop_flag = False
            #     finished = session.agent_state in ['finished', 'stopped']
            #     clear = gr.Button('🗑️ Clear', interactive=finished)
            #     screenshot, url = browser_history[-1]
            #     # find 2nd last index of last user message
            #     last_user_index = None
            #     user_message_count = 0
            #     print(chat_history, "bruh")
            #     for i in range(len(chat_history) - 1, -1, -1):
            #         msg = chat_history[i]
            #         if msg['role'] == 'user':
            #             user_message_count += 1
            #         if user_message_count == 2:
            #             last_user_index = i
            #             break

            #     # keep most recent message
            #     chat_history = chat_history[:last_user_index] + chat_history[-1:]
            #     print(chat_history, "oi oi oi ")
            #     session._reset()
            #     action_messages = []

            #     submit = gr.Button(
            #         'Submit',
            #         variant='primary',
            #         scale=1,
            #         min_width=150,
            #         visible=session.agent_state != 'running',
            #     )
            #     stop = gr.Button(
            #         'Stop',
            #         scale=1,
            #         min_width=150,
            #         visible=session.agent_state == 'running',
            #     )

            #     yield (
            #         chat_history,
            #         screenshot,
            #         url,
            #         [],
            #         browser_history,
            #         session,
            #         status,
            #         clear,
            #         options_visible,
            #         upvote,
            #         downvote,
            #         submit,
            #         stop,
            #     )

            session.agent = agent_selection
            session.model = model_selection
            session.api_key = api_key
            # if model_requires_key[model_selection]:
            #     session.api_key = api_key
            # elif model_port_config[model_selection].get('default_key', None):
            #     session.api_key = model_port_config[model_selection].get(
            #         'default_key', None
            #     )
            # else:
            #     session.api_key = ''

            print('API Key:', session.api_key)
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
                    visible=False,
                )
                stop = gr.Button('Stop', scale=1, min_width=150, visible=True)

                finished = session.agent_state in ['finished', 'stopped']
                clear = gr.Button('🗑️ Clear', interactive=finished)

                yield (
                    chat_history,
                    screenshot,
                    url,
                    action_messages,
                    browser_history,
                    session,
                    status,
                    clear,
                    options_visible,
                    upvote,
                    downvote,
                    submit,
                    stop,
                )

        website_counter = 0
        message_list = []
        if thinking_flag:
            if browser_starting_flag:
                chat_history = chat_history[:-1]
                browser_starting_flag = False
            loading_message = gr.ChatMessage(
                role='assistant', content='⏳ Thinking...'
            ).__dict__
            chat_history.append(loading_message)

        for message in session.run(user_message, request):
            message_list.append(message['message'])
            if website_counter == 1:
                options_visible = True
            finished = session.agent_state in ['finished', 'stopped']
            clear = gr.Button('🗑️ Clear', interactive=finished)
            upvote = gr.Button('👍 Good Response', interactive=finished)
            downvote = gr.Button('👎 Bad Response', interactive=finished)

            if message.get('action', '') in ['message', 'finish']:
                if thinking_flag:
                    chat_history = chat_history[:-1]
                    thinking_flag = False

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
                            options_visible,
                            upvote,
                            downvote,
                            submit,
                            stop,
                        )
            elif (
                session.agent.startswith('ReasonerAgent')
                and message.get('action', '') == 'browse_interactive'
                and message.get('args', {}).get('thought', '')
            ):
                if thinking_flag:
                    chat_history = chat_history[:-1]
                    thinking_flag = False
                full_output_dict = json.loads(message['args']['thought'])
                plan = full_output_dict.get('plan', message['message'])

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
                            options_visible,
                            upvote,
                            downvote,
                            submit,
                            stop,
                        )
            elif (
                session.agent == 'BrowsingAgent'
                and message.get('action', '') == 'browse_interactive'
                # and message.get('args', {}).get('thought', '')
            ):
                if thinking_flag:
                    chat_history = chat_history[:-1]
                    thinking_flag = False
                thought = message.get('args', {}).get('thought', '')
                if not thought:
                    thought = message['message']
                # chat_history.append(gr.ChatMessage(role='assistant', content=thought))
                print(thought)

                chat_history.append(gr.ChatMessage(role='assistant', content=''))
                assistant_message = thought
                assistant_message_chars = []
                for i, char in enumerate(assistant_message):
                    assistant_message_chars.append(char)
                    updated_message = ''.join(assistant_message_chars)
                    if (i + 1) % 10 == 0 or i == len(assistant_message) - 1:
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
                            options_visible,
                            upvote,
                            downvote,
                            submit,
                            stop,
                        )

            if session.agent_state == 'finished':
                session.save_log()
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

            submit = gr.Button(
                'Submit',
                variant='primary',
                scale=1,
                min_width=150,
                visible=session.agent_state != 'running',
            )
            stop = gr.Button(
                'Stop', scale=1, min_width=150, visible=session.agent_state == 'running'
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
                options_visible,
                upvote,
                downvote,
                submit,
                stop,
            )


def clear_page(browser_history, session):
    browser_history = browser_history[:1]
    current_screenshot, current_url = browser_history[-1]

    if session is not None:
        session._reset()
        status = get_status(session.agent_state)
    else:
        status = get_status(None)

    return (
        None,
        current_screenshot,
        current_url,
        [],
        browser_history,
        session,
        status,
    )


def check_supported_models(agent_selection, model_selection, api_key):
    supported_models = agent_supported_models.get(agent_selection, model_list)
    selected_model = (
        model_selection if model_selection in supported_models else supported_models[0]
    )
    model_selection = gr.Dropdown(
        supported_models,
        value=selected_model,
        interactive=True,
        label='Backend LLM',
        scale=1,
        # info='Choose the model you would like to use',
    )
    return model_selection, check_requires_key(selected_model, api_key)


def check_requires_key(model_selection, api_key):
    model_real_name = model_display2name[model_selection]
    requires_key = model_requires_key[model_real_name]
    api_key = gr.Textbox(
        api_key,
        label='API Key',
        placeholder='Your API Key',
        visible=requires_key,
        scale=1,
        max_lines=2,
    )
    return api_key


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

    if 'goto' in action_messages[-1]:
        history_title = 'Browsing ' + message + '...'

    last_non_assistant_message_idx = 0
    for i, chat_message in enumerate(history):
        if not isinstance(chat_message, dict) and chat_message.role != 'assistant':
            last_non_assistant_message_idx = i
        elif isinstance(chat_message, dict) and chat_message['role'] != 'assistant':
            last_non_assistant_message_idx = i

    links_string_idx = last_non_assistant_message_idx + 1
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

    return history


def process_user_message(user_message, history, session):
    if not user_message.strip():
        chat_message = gr.ChatMessage(role='user', content='')
        history.append(chat_message)
        return '', history

    chat_message = gr.ChatMessage(role='user', content=user_message)
    history.append(chat_message)

    return '', history


def stop_task(session):
    # if session.agent_state == 'running':
    session.stop()
    status = get_status(session.agent_state)
    clear = gr.Button('🗑️ Clear', interactive=True)
    return session, status, clear


# toggle hiding and showing the browser. IfClick is basically because I call
# this function sometimes without the user specifically clicking on the button.
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


def unload_fn(request: gr.Request):
    if request.session_hash in global_sessions.keys():
        global_sessions[request.session_hash].stop()
        backend_manager.release_backend(global_sessions[request.session_hash].port)
        del global_sessions[request.session_hash]


current_dir = os.path.dirname(__file__)
print(os.path.dirname(__file__))

global model_port_config
model_port_config = {}
with open(os.path.join(current_dir, 'model_port_config.json')) as f:
    model_port_config = json.load(f)

global model_display2name
model_display2name = {
    cfg.get('display_name', model): model for model, cfg in model_port_config.items()
}
model_list = list(model_display2name.keys())
global model_requires_key
model_requires_key = {
    model: cfg.get('requires_key', False) for model, cfg in model_port_config.items()
}

default_model = 'gpt-4o'
for model, cfg in model_port_config.items():
    if cfg.get('default', None):
        default_model = cfg.get('display_name', model)
        break

current_dir = os.path.dirname(__file__)
default_api_key = None

model_name2keypath = {'gpt-4o-mini': 'default_openai_api_key.txt'}


def vote(vote, session):
    if vote:
        print('Upvoted!')
    else:
        print('Downvoted.')
    session.save_user_feedback(vote)
    upvote_button = gr.Button('👍 Good Response', interactive=False)
    downvote_button = gr.Button('👎 Bad Response', interactive=False)
    return upvote_button, downvote_button


tos_popup_js = r"""
() => {
    if (window.alerted_before) return;

    const msg = "Users of this website are required to agree to the following terms:\n\n" +
           "This service is a research preview offering limited safety measures and may perform unsafe actions. " +
           "It must not be used for any illegal, harmful, violent, racist, or sexual purposes. " +
           "Please refrain from uploading any private or sensitive information. " +
           "By using this service, you acknowledge that we collect user requests and webpage data (screenshots and text content), " +
           "and reserve the right to distribute this data under Creative Commons Attribution (CC-BY) or a similar license.";


    alert(msg);
    window.alerted_before = true;
}
"""
image_css = """
.sponsor-image-about img {
    margin: 0 20px;
    margin-top: 20px;
    height: 40px;
    max-height: 100%;
    width: auto;
    float: left;
}
"""
google_analytics_tracking_id = None
try:
    with open('google_analytics_api_key.txt', 'r') as f:
        google_analytics_tracking_id = f.read().strip()
except FileNotFoundError:
    pass
ga_head_snippet = ''
if google_analytics_tracking_id:
    ga_head_snippet = f"""
    if (typeof window.dataLayer === "undefined") {{
        window.dataLayer = [];
    }}
    function gtag() {{ window.dataLayer.push(arguments); }}
    gtag('js', new Date());
    gtag('config', '{google_analytics_tracking_id}');
    """

combined_js = r"""
() => {
    // TOS Popup code
    if (!window.alerted_before) {
        const msg = "Users of this website are required to agree to the following terms:\n\n" +
           "This service is a research preview offering limited safety measures and may perform unsafe actions. " +
           "It must not be used for any illegal, harmful, violent, racist, or sexual purposes. " +
           "Please refrain from uploading any private or sensitive information. " +
           "By using this service, you acknowledge that we collect user requests and webpage data (screenshots and text content), " +
           "and reserve the right to distribute this data under Creative Commons Attribution (CC-BY) or a similar license.";
        alert(msg);
        window.alerted_before = true;
    }

    {ga_head_snippet}
}
"""

agent_descriptions = [
    'DummyWebAgent — Debugging only',
    'BrowsingAgent — 🏃‍♂️ Good for quick tasks, but limited depth.',
    'ReasonerAgent (Fast) — ⚖️ Mix of speed and intelligence.',
    'ReasonerAgent (Full) — 🧠 Most advanced reasoning, but slower.',
]

agent_display_ids = [1, 2, 3]
agent_display_names = [agent_descriptions[idx] for idx in agent_display_ids]

default_agent_id = 2
default_agent = agent_descriptions[default_agent_id]

agent_display2class = {
    agent_descriptions[0]: 'DummyWebAgent',
    agent_descriptions[1]: 'BrowsingAgent',
    agent_descriptions[2]: 'ReasonerAgentFast',
    agent_descriptions[3]: 'ReasonerAgentFull',
}

agent_supported_models = {
    agent_descriptions[3]: ['GPT-4o-mini (Free)', 'GPT-4o'],
    agent_descriptions[3]: ['GPT-4o-mini (Free)', 'GPT-4o'],
}

with gr.Blocks(
    theme=gr.themes.Default(text_size=gr.themes.sizes.text_lg),
    css=image_css,
    title='EasyWeb: AI-Powered Web Agents at Your Fingertips',
) as demo:  # css=css
    with gr.Tab('🌐 Demo'):
        action_messages = gr.State([])
        session = gr.State(None)
        title = gr.Markdown(
            """\
# 🌐 EasyWeb: AI-Powered Web Agents at Your Fingertips
<font size="4">

[X](https://x.com/MaitrixOrg) | [Discord](https://discord.gg/b5NEhRbvJg) | [GitHub](https://github.com/maitrix-org/easyweb)

</font>
"""
        )

        description = gr.Markdown(
            """\

<font size="4">

**Example Prompts:**
- "Use DuckDuckGo to search for the current president of USA."
- "I want to buy a black mattress. Find one black mattress option from Amazon and eBay?"
- "Go to the website of MinnPost, find an article about Trump's second inauguration, and summarize the main points for me."

**Note:** The agent currently **does not remember previous messages**, and defaults to **DuckDuckGo** for search engine due to restrictions. \
Include specific websites or detailed instructions in your prompt for more consistent behavior.

**⚠️ The interface is currently optimized for Chrome.** If you encounter any issues, please try using Chrome.

**❗ This is currently an early research preview, where agents may make mistakes or have challenges navigating certain websites. For research purposes, we log user prompts and feedback and may release to the public in the future. Please do not upload any confidential or personal information.**

</font>
"""
        )

        with gr.Group():
            with gr.Row():
                agent_selection = gr.Dropdown(
                    agent_display_names,
                    value=default_agent,
                    interactive=True,
                    label='Agent',
                    scale=2,
                    # info='Choose your own adventure partner!',
                )
                model_selection = gr.Dropdown(
                    model_list,
                    value=default_model,
                    interactive=True,
                    label='Backend LLM',
                    scale=1,
                    # info='Choose the model you would like to use',
                )
                api_key = check_requires_key(default_model, default_api_key)

        with gr.Row(equal_height=False):
            with gr.Column(scale=1):
                chatbot = gr.Chatbot(type='messages')

                with gr.Group():
                    with gr.Row():
                        msg = gr.Textbox(container=False, show_label=False, scale=7)

                        submit = gr.Button(
                            'Submit',
                            variant='primary',
                            scale=1,
                            min_width=150,
                        )
                        stop = gr.Button('Stop', scale=1, min_width=150, visible=False)
                        submit_triggers = [msg.submit, submit.click]
            with gr.Column(scale=2, visible=False) as visualization_column:
                with gr.Group():
                    start_url = 'about:blank'
                    url = gr.Textbox(
                        start_url, label='URL', interactive=False, max_lines=1
                    )
                    blank = Image.new('RGB', (1280, 720), (255, 255, 255))
                    screenshot = gr.Image(blank, interactive=False, label='Webpage')

        with gr.Row():
            toggle_button = gr.Button('🔍 Show Browser')
            upvote = gr.Button('👍 Good Response', interactive=False)
            downvote = gr.Button('👎 Bad Response', interactive=False)
            clear = gr.Button('🗑️ Clear')
        status = gr.Markdown('<font size="4"> Agent Status: 🔴 Inactive </font>')
        browser_history = gr.State([(blank, start_url)])
        options_visible = gr.State(False)
        upvote.click(vote, inputs=[gr.State(True), session], outputs=[upvote, downvote])
        downvote.click(
            vote, inputs=[gr.State(False), session], outputs=[upvote, downvote]
        )
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
        chat_msg = gr.events.on(
            submit_triggers,
            process_user_message,
            [msg, chatbot, session],
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
                options_visible,
                upvote,
                downvote,
                submit,
                stop,
            ],
            concurrency_limit=args.num_backends,
        )
        (
            stop.click(
                stop_task,
                [session],
                [session, status, clear],
                queue=False,
            )
        )
        (
            clear.click(
                clear_page,
                [browser_history, session],
                [
                    chatbot,
                    screenshot,
                    url,
                    action_messages,
                    browser_history,
                    session,
                    status,
                ],
                queue=False,
            ).then(fn=None)
        )
        agent_selection.select(
            check_supported_models,
            [agent_selection, model_selection, api_key],
            [model_selection, api_key],
            queue=False,
        )
        model_selection.select(
            check_requires_key, [model_selection, api_key], api_key, queue=False
        )
        tos = gr.Markdown(
            """\
<font size="4">

#### Terms of Service
Users of this website are required to agree to the following terms:\n\n
This service is a research preview offering limited safety measures and may perform unsafe actions.
It must not be used for any illegal, harmful, violent, racist, or sexual purposes.
Please refrain from uploading any private or sensitive information.
By using this service, you acknowledge that we collect user requests and webpage data (screenshots and text content),
and reserve the right to distribute this data under Creative Commons Attribution (CC-BY) or a similar license.

#### Please report any bug or issue to our [Discord](https://discord.gg/b5NEhRbvJg)

#### Acknowledgment

We thank the [OpenHands](https://github.com/All-Hands-AI/OpenHands) team for their technical support.
We also thank [MBZUAI](https://mbzuai.ac.ae/) and [Samsung](https://www.samsung.com/) for their generous sponsorship. Contact us to learn more about partnership.

</font>

<div class="sponsor-image-about">
    <img src="https://storage.googleapis.com/public-arena-asset/mbzuai.jpeg" alt="MBZUAI" style="width: 150px; height: auto;">
    <img src="https://upload.wikimedia.org/wikipedia/commons/f/f1/Samsung_logo_blue.png" alt="Samsung" style="width: 150px; height: auto;">
</div>

"""
        )
    with gr.Tab('📖 Instructions'):
        with gr.Row():
            instructions = gr.Markdown(
                """\
## 📖 How It Works
<font size="4">

1️⃣ **Choose an Agent & LLM:**
- Use GPT-4o-mini for free, or bring your own GPT-4o API key for better performance.
- We are working on enabling other LLMs soon!

2️⃣ **Ask the agent to perform web-related tasks like:**
- "Use DuckDuckGo to search for the current president of USA."
- "I want to buy a black mattress. Find one black mattress option from Amazon and eBay?"
- "Go to the website of MinnPost, find an article about Trump's second inauguration, and summarize the main points for me."

3️⃣ **Give us a 👍 or 👎 once the agent completes the task!**

**⚠️ Privacy and Data Usage:** Submitted data may be used for research, and user prompts and feedback are logged. Please avoid uploading confidential or personal information.
## 🔎 Browsing Tips:
- 🦆 Due to restrictions, the agents are at their best with DuckDuckGo as the search engine.
- ⚡ Speed may vary depending on backend API load.
- 🔄 The agent may repeat actions before trying alternative approaches.
- 🎯 Clearer prompts help — specific websites or detailed instructions improve performance.
- 🛡️ We honor site protections like CAPTCHAs and anti-bot measures to maintain user and website integrity.
- 💡 The agent currently only sees **up to the latest user message**. Stay tuned as we work on enabling multi-turn interactions.

</font>
"""
            )
    with gr.Tab('ℹ️ About Us'):
        with gr.Row():
            introductions = gr.Markdown(
                """\
## About Us

<font size="4">

EasyWeb is an open platform for building and serving AI web agents, hosted by students at CMU [Sailing Lab](https://sailing-lab.github.io/) and UCSD [Maitrix Team](https://maitrix.org/).
We open-source the [EasyWeb](https://github.com/maitrix-org/easyweb) project at Github, and always welcome contributions from the community.
If you are interested in collaboration, please contact us, we'd love to hear from you!

</font>

### Open-Source Contributors

<font size="4">

- Contributors: [Mingkai Deng](https://mingkaid.github.io/), [Jackie Wang](https://www.linkedin.com/in/yikunjwang/), [Mason Choey](https://www.linkedin.com/in/mason-choey-9a6657325/), [Ariel Wu](https://www.linkedin.com/in/ariel-wu-63624716b/), [Brandon Chiou](https://www.linkedin.com/in/brandon-chiou/), [Jinyu Hou](https://www.linkedin.com/in/jinyu-hou-uoft/)
- Advisors: [Zhiting Hu](https://zhiting.ucsd.edu/), [Hongxia Jin](https://www.linkedin.com/in/hongxiajin/), [Li Erran Li](https://www.cs.columbia.edu/~lierranli/), [Graham Neubig](https://www.phontron.com/), [Yilin Shen](https://www.linkedin.com/in/yilin-shen-65a56622/), [Eric P. Xing](https://www.cs.cmu.edu/~epxing/)

📩 **Correspondence to**: [Mingkai Deng](mailto:mingkaid34@gmail.com) and [Jackie Wang](mailto:yikunjwang@gmail.com)

</font>

### Contact Us

<font size="4">

- Follow our [X](https://x.com/MaitrixOrg) and [Discord](https://discord.gg/b5NEhRbvJg)
- File issues on [Github](https://github.com/maitrix-org/easyweb)

</font>

### Acknowledgment

<font size="4">

We thank the [OpenHands](https://github.com/All-Hands-AI/OpenHands) team for their technical support.
We also thank [MBZUAI](https://mbzuai.ac.ae/) and [Samsung](https://www.samsung.com/) for their generous sponsorship. Contact us to learn more about partnership.

</font>

<div class="sponsor-image-about">
    <img src="https://storage.googleapis.com/public-arena-asset/mbzuai.jpeg" alt="MBZUAI" style="width: 150px; height: auto;">
    <img src="https://upload.wikimedia.org/wikipedia/commons/f/f1/Samsung_logo_blue.png" alt="Samsung" style="width: 150px; height: auto;">
</div>
"""
            )
    # demo.load(None, None, None, js=tos_popup_js + ga_head_snippet)
    demo.load(None, None, None, js=combined_js)
    demo.unload(unload_fn)

if __name__ == '__main__':
    demo.queue()
    if args.ip and args.port:
        demo.launch(
            share=False,
            server_name=args.ip,
            server_port=args.port,
            favicon_path='./frontend-icon.png',
            ssl_certfile=args.ssl_certfile,
            ssl_keyfile=args.ssl_keyfile,
            ssl_verify=args.ssl_verify,
        )
    else:
        demo.launch(share=True, favicon_path='./frontend-icon.png')
