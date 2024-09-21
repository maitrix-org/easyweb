import base64
from io import BytesIO

import gradio as gr
import requests
from PIL import Image


def handle_start(goal):
    response = requests.post('http://localhost:5001/start', json={'goal': goal})
    payload = response.json()
    print(payload['status'])
    frozen_goal = gr.Textbox(value=goal, interactive=False)

    start = gr.Button(interactive=False)
    close = gr.Button(interactive=True)
    step = gr.Button(interactive=True)
    complete = gr.Button(interactive=True)
    return (
        frozen_goal,
        'A new browser session has started!',
        start,
        close,
        step,
        complete,
    )


def handle_close(goal):
    response = requests.post('http://localhost:5001/close')
    print(response.json())
    goal = gr.Textbox(value='', interactive=True)
    start = gr.Button(interactive=True)
    close = gr.Button(interactive=False)
    step = gr.Button(interactive=False)
    complete = gr.Button(interactive=False)
    save = gr.Button(interactive=False)
    return (
        goal,
        'The current browser session has been closed!',
        '',
        blank,
        '',
        start,
        close,
        step,
        complete,
        save,
    )


def handle_complete():
    response = requests.post('http://localhost:5001/complete')
    print(response.json())
    step = gr.Button(interactive=False)
    complete = gr.Button(interactive=False)
    save = gr.Button(interactive=True)
    update = "You've marked your current session as complete!"
    return update, step, complete, save


def handle_save():
    response = requests.post('http://localhost:5001/save')
    print(response.json())
    save = gr.Button(interactive=False)
    update = "You've saved your current session!"
    return update, save


def handle_step(state, instruction, action):
    response = requests.post(
        'http://localhost:5001/step',
        json={'state': state, 'instruction': instruction, 'action': action},
    )
    print(response.json())
    update = "You've taken a step in the browser!"

    # update, state, instruction, action
    return update, '', '', ''


def refresh_history():
    response = requests.get('http://localhost:5001/history')
    history = response.json()['history']
    print(history)
    return history


def refresh_observation():
    response = requests.get('http://localhost:5001/observation')
    observation = response.json()

    image_data = base64.b64decode(observation['screenshot'])
    screenshot = Image.open(BytesIO(image_data))
    return observation['url'], screenshot, observation['axtree']


with gr.Blocks() as demo:
    with gr.Row():
        title = gr.Markdown("""\
# Data Collection Interface
""")
        update = gr.Markdown('')
    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            with gr.Group():
                goal = gr.Textbox(label='Enter your goal')
                with gr.Row():
                    start = gr.Button('Start')
                    close = gr.Button('Close', interactive=False)
            history = gr.Textbox(
                label='History', interactive=False, lines=10, max_lines=10
            )
            with gr.Group():
                state = gr.Textbox(
                    label='State',
                    placeholder='(Enter anything you see in the observation that is relevant to your decision about next steps)',
                    interactive=True,
                    lines=5,
                    max_lines=5,
                )
                instruction = gr.Textbox(
                    label='High-Level Action / Instruction',
                    placeholder='(Enter the instruction someone can use to select the next action. Start with a verb)',
                    interactive=True,
                )
                action = gr.Textbox(
                    label='Low-Level Action',
                    placeholder='(Enter the API call for the action you are taking)',
                    interactive=True,
                )
                step = gr.Button('Step', interactive=False)
                with gr.Row():
                    complete = gr.Button('Complete', interactive=False)
                    save = gr.Button('Save', interactive=False)

        with gr.Column(scale=1):
            url = gr.Textbox(label='URL', interactive=False, max_lines=1)
            blank = Image.new('RGB', (1280, 720), (255, 255, 255))
            screenshot = gr.Image(blank, interactive=False, label='Webpage')
            webpage = gr.Textbox(
                interactive=False, lines=20, max_lines=30, label='Webpage AXTree'
            )

    start_session = (
        start.click(handle_start, [goal], [goal, update, start, close, step, complete])
        .then(refresh_history, [], [history])
        .then(refresh_observation, [], [url, screenshot, webpage])
    )
    close_session = close.click(
        handle_close,
        [goal],
        [goal, update, url, screenshot, webpage, start, close, step, complete, save],
    )

    toggle_complete = complete.click(
        handle_complete, [], [update, step, complete, save]
    )
    toggle_save = save.click(handle_save, [], [update, save])

    make_step = (
        step.click(
            handle_step,
            [state, instruction, action],
            [update, state, instruction, action],
        )
        .then(refresh_history, [], [history])
        .then(refresh_observation, [], [url, screenshot, webpage])
    )

    action_space = gr.Markdown("""\
## Action Space
16 different types of actions are available:
```
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
```
""")

if __name__ == '__main__':
    demo.queue()
    demo.launch(share=False)
