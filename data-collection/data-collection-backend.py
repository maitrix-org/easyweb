from datetime import datetime

from data_collection_interface import BrowserGymSession
from flask import Flask, jsonify, request
from gevent import monkey

monkey.patch_all()

app = Flask(__name__)

session = BrowserGymSession()

browser_obs = {'axtree': None, 'url': None, 'screenshot': None}


@app.route('/start', methods=['POST'])
def start():
    data = request.json
    goal = data['goal']
    session.start(goal)
    axtree, url, screenshot = session.get_obs()
    browser_obs.update({'axtree': axtree, 'url': url, 'screenshot': screenshot})
    return jsonify(
        {
            'status': 'running',
            'start_time': session.current_datetime,
            'timestamp': datetime.now().strftime('%Y-%m-%d-%H:%M:%S'),
        }
    )


@app.route('/close', methods=['POST'])
def close():
    # session.save()
    session.close()
    return jsonify(
        {'status': 'closed', 'timestamp': datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}
    )


@app.route('/complete', methods=['POST'])
def complete():
    session.mark_complete()
    return jsonify(
        {
            'status': 'completed',
            'timestamp': datetime.now().strftime('%Y-%m-%d-%H:%M:%S'),
        }
    )


@app.route('/save', methods=['POST'])
def save():
    session.save()
    return jsonify(
        {'status': 'saved', 'timestamp': datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}
    )


@app.route('/step', methods=['POST'])
def step():
    data = request.json
    session.record_step(**data)

    axtree, url, screenshot = session.get_obs()
    browser_obs.update({'axtree': axtree, 'url': url, 'screenshot': screenshot})

    return jsonify(
        {'status': 'success', 'timestamp': datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}
    )


@app.route('/observation', methods=['GET'])
def observation():
    axtree, url, screenshot = session.get_obs()
    return jsonify(
        {
            'timestamp': datetime.now().strftime('%Y-%m-%d-%H:%M:%S'),
            'axtree': axtree,
            'url': url,
            'screenshot': screenshot,
        }
    )

    # return jsonify({"timestamp": datetime.now().strftime("%Y-%m-%d-%H:%M:%S"),
    #                 "axtree": browser_obs['axtree'],
    #                 "url": browser_obs['url'],
    #                 "screenshot": image_to_png_base64_url(browser_obs['screenshot'])})


@app.route('/history', methods=['GET'])
def history():
    history_prompt = session.get_history_prompt()
    return jsonify({'history': history_prompt})


if __name__ == '__main__':
    app.run(port=5001, debug=True)  # Run on a different port if Gradio is on 5000
