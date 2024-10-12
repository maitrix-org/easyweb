import asyncio
import base64
import io
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import gradio as gr
import gymnasium as gym
import numpy as np
from browsergym.utils.obs import flatten_axtree_to_str
from PIL import Image

executor = ThreadPoolExecutor()


async def run_sync_function_in_thread(sync_func, *args, **kwargs):
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(executor, sync_func, *args, **kwargs)
    return result


def get_scroll_position(page):
    return page.evaluate("""() => {
        const scrollTop = window.scrollY;
        const windowHeight = window.innerHeight;
        const documentHeight = document.documentElement.scrollHeight;
        const remainingPixels = documentHeight - (scrollTop + windowHeight);

        return {
            'scrollTop': scrollTop,
            'windowHeight': windowHeight,
            'documentHeight': documentHeight,
            'remainingPixels': remainingPixels
        };
    }""")


def image_to_png_base64_url(
    image: np.ndarray | Image.Image, add_data_prefix: bool = False
):
    """Convert a numpy array to a base64 encoded png image url."""

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if image.mode in ('RGBA', 'LA'):
        image = image.convert('RGB')

    # original_width, original_height = image.size
    # aspect_ratio = original_width / original_height
    # new_height = 720
    # new_width = int(new_height * aspect_ratio)

    # resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    buffered = io.BytesIO()
    image.save(buffered, format='PNG')
    # resized_image.save(buffered, format='PNG', optimize=True, compress_level=9)

    image_base64 = base64.b64encode(buffered.getvalue()).decode()
    return (
        f'data:image/png;base64,{image_base64}'
        if add_data_prefix
        else f'{image_base64}'
    )


class BrowserGymSession:
    __SLOW_MO = None
    __HEADLESS = True
    __TIMEOUT = 5000
    __VIEWPORT = {'width': 1280, 'height': 720}
    __WAIT_FOR_USER_MESSAGE = False

    def __init__(self):
        self._reset()

    def _reset(self):
        self.goal = None
        self.current_datetime = None
        self.env = None
        self.obs = None
        self.obs_save = None
        self.info = None
        self.is_complete = False
        self.history = []

    def start(self, goal):
        self.goal = goal
        print('Goal:', goal)
        # Get current datetime and format it
        self.current_datetime = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        print(self.current_datetime)

        self.env = gym.make(
            'browsergym/openended',
            task_kwargs={'start_url': 'about:blank', 'goal': goal},
            wait_for_user_message=self.__WAIT_FOR_USER_MESSAGE,
            headless=self.__HEADLESS,
            slow_mo=self.__SLOW_MO,
            viewport=self.__VIEWPORT,
            timeout=self.__TIMEOUT,
            # disable_env_checker=True,
        )

        # self.obs, self.info = await run_sync_function_in_thread(self.env.reset)

        self.obs, self.info = self.env.reset()

        self.history = []
        self.is_complete = False
        print('Session started.')

    def close(self):
        # self.env.close()
        del self.env
        self._reset()

    def mark_complete(self):
        self.is_complete = True

    def save(self):
        session_data = {
            'goal': self.goal,
            'history': self.history,
            'is_complete': self.is_complete,
        }
        with open('./data-collection/' + self.current_datetime + '.json', 'w') as f:
            json.dump(session_data, f)

    def get_obs(self):
        scroll_position = get_scroll_position(self.env.page)
        error_prefix = ''
        # print(self.obs.keys())
        # print(self.obs['last_action_error'])
        if self.obs['last_action_error']:
            # add error recovery prompt prefix
            error_prefix = f'IMPORTANT! Last action is incorrect:\n{self.obs["last_action"]}\n{self.obs["last_action_error"]}\nThink again with the current observation of the page.\n'

        cur_axtree_txt = flatten_axtree_to_str(
            self.obs['axtree_object'],
            extra_properties=self.obs['extra_element_properties'],
            with_clickable=True,
            filter_visible_only=True,
        )

        clean_axtree_lines = []
        num_static_text_lines = 0
        max_static_text_lines = 20
        last_bracket_line = 0
        max_after_last_bracket_lines = 10
        for i, line in enumerate(cur_axtree_txt.split('\n')):
            if line.strip().startswith('['):
                last_bracket_line = i

        for i, line in enumerate(cur_axtree_txt.split('\n')):
            if line.strip().startswith('StaticText') or line.strip().startswith(
                'ListMarker'
            ):
                num_static_text_lines += 1
            else:
                num_static_text_lines = 0

            if num_static_text_lines <= max_static_text_lines and i < (
                last_bracket_line + max_after_last_bracket_lines
            ):
                clean_axtree_lines.append(line)

        clean_axtree_txt = '\n'.join(clean_axtree_lines)

        scroll_progress = (
            1 - scroll_position['remainingPixels'] / scroll_position['documentHeight']
        )
        clean_axtree_txt = (
            f"URL {self.obs['url']}\n"
            f"Scroll Position: {scroll_position['scrollTop']}, "
            f"Window Height: {scroll_position['windowHeight']}, "
            f"Webpage Height: {scroll_position['documentHeight']}, "
            f"Remaining Pixels: {scroll_position['remainingPixels']}, "
            f"Scrolling Progress: {scroll_progress:.1%}\n"
        ) + clean_axtree_txt

        screenshot = image_to_png_base64_url(self.obs['screenshot'])

        self.obs_save = {
            'url': self.obs['url'],
            'scroll_position': scroll_position,
            'raw_axtree_txt': cur_axtree_txt,
            'clean_axtree_txt': clean_axtree_txt,
            'error_prefix': error_prefix,
            'axtree_object': self.obs['axtree_object'],
            'extra_element_properties': self.obs['extra_element_properties'],
            'last_action': self.obs['last_action'],
            'last_action_error': self.obs['last_action_error'],
            'screenshot': screenshot,
        }

        obs_prompt = clean_axtree_txt
        if len(error_prefix) > 0:
            obs_prompt = f'{error_prefix}\n' + obs_prompt

        # print(self.env.chat.messages)

        return obs_prompt, self.obs['url'], screenshot

    def record_step(self, state, instruction, action):
        if self.obs_save is None:
            self.get_obs()
        self.history.append((self.obs_save, state, instruction, action))

        self.obs, reward, terminated, truncated, self.info = self.env.step(action)
        self.obs_save = None

    def get_history_prompt(self):
        history_steps = []
        for i, (obs, state, instruction, action) in enumerate(self.history):
            step = f"""\
# Step {i + 1}:

## State:
{state}\

## Action:
{instruction}
{action}
"""
            history_steps.append(step)

        history_prompt = '\n\n'.join(history_steps)
        if len(history_prompt) > 0:
            return history_prompt
        else:
            return 'Beginning of task'


if __name__ == '__main__':
    with gr.Blocks() as demo:
        title = gr.Markdown("""\
# Data Collection Interface

## Action Space
16 different types of actions are available:

noop(wait_ms: float = 1000)
    Examples:
        noop()
        noop(500)

send_msg_to_user(text: str)
    Examples:
        send_msg_to_user('Based on the results of my search, the city was built in 1751.')

scroll(delta_x: float, delta_y: float)
    Examples:
        scroll(0, 200)
        scroll(-50.2, -100.5)

fill(bid: str, value: str)
    Examples:
        fill('237', 'example value')
        fill('45', 'multi-line\
example')
        fill('a12', 'example with "quotes"')

select_option(bid: str, options: str | list[str])
    Examples:
        select_option('a48', 'blue')
        select_option('c48', ['red', 'green', 'blue'])

click(bid: str, button: Literal['left', 'middle', 'right'] = 'left', modifiers: list[typing.Literal['Alt', 'Control', 'Meta', 'Shift']] = [])
    Examples:
        click('a51')
        click('b22', button='right')
        click('48', button='middle', modifiers=['Shift'])

dblclick(bid: str, button: Literal['left', 'middle', 'right'] = 'left', modifiers: list[typing.Literal['Alt', 'Control', 'Meta', 'Shift']] = [])
    Examples:
        dblclick('12')
        dblclick('ca42', button='right')
        dblclick('178', button='middle', modifiers=['Shift'])

hover(bid: str)
    Examples:
        hover('b8')

press(bid: str, key_comb: str)
    Examples:
        press('88', 'Backspace')
        press('a26', 'Control+a')
        press('a61', 'Meta+Shift+t')

focus(bid: str)
    Examples:
        focus('b455')

clear(bid: str)
    Examples:
        clear('996')

drag_and_drop(from_bid: str, to_bid: str)
    Examples:
        drag_and_drop('56', '498')

upload_file(bid: str, file: str | list[str])
    Examples:
        upload_file('572', 'my_receipt.pdf')
        upload_file('63', ['/home/bob/Documents/image.jpg', '/home/bob/Documents/file.zip'])

go_back()
    Examples:
        go_back()

go_forward()
    Examples:
        go_forward()

goto(url: str)
    Examples:
        goto('http://www.example.com')

Only a single action can be provided at once. Example:
    fill('a12', 'example with "quotes"')
""")
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                with gr.Group():
                    goal = gr.Textbox(label='Goal')
                    with gr.Row():
                        start_discard = gr.Button('Start')
                        save_close = gr.Button('Save & Close', interactive=False)
                    history = gr.Markdown('No History Yet')
                with gr.Group():
                    with gr.Row():
                        state = gr.Textbox(label='State')
                        termination = gr.Checkbox(label='Replan', value=True)
                    strategy = gr.Textbox(label='Strategy', interactive=True)
                    action = gr.Textbox(label='Action')
                    explanation = gr.Textbox(label='Explanation')

            with gr.Column(scale=2):
                with gr.Group():
                    start_url = 'about:blank'
                    url = gr.Textbox(
                        start_url, label='URL', interactive=False, max_lines=1
                    )
                    blank = Image.new('RGB', (1280, 720), (255, 255, 255))
                    screenshot = gr.Image(blank, interactive=False, label='Webpage')
                placeholder = 'Blank'
                webpage = gr.Textbox(
                    placeholder,
                    interactive=False,
                    lines=20,
                    max_lines=30,
                )

        session = gr.State(BrowserGymSession())

        def handle_start_discard(start_discard, session, goal):
            if session.env is not None:
                session.close()
                start_discard = 'Start'
                goal = ''
                save_close = gr.Button('Save & Close', interactive=False)
            else:
                session.start(goal)
                start_discard = 'Discard'
                goal = gr.Textbox(value=goal, interactive=False)
                save_close = gr.Button('Save & Close', interactive=True)
            return start_discard, session, goal, save_close

        start_discard_session = start_discard.click(
            handle_start_discard,
            [start_discard, session, goal],
            [start_discard, session, goal, save_close],
            queue=False,
        )

        def handle_save_close(session):
            session.save()
            session.close()
            start_discard = 'Start'
            goal = ''
            save_close = gr.Button('Save & Close', interactive=False)
            return start_discard, session, goal, save_close

        save_close_session = save_close.click(
            handle_save_close, [session], [start_discard, session, goal, save_close]
        )

    # demo.queue(default_concurrency_limit=5)
    # demo.queue()
    demo.launch(share=False)
