import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from web_evaluation.inference import main

data = json.load(open('./task_data/fanout-dev-0-30.json'))
# data = json.load(open('/Users/mingkaid/Documents/Tech-Life/lwm/fanoutqa/fanoutqa/data/fanout-final-dev.json'))

# questions = ['Using Google, ' + row['question'] for row in data]
questions = [row['question'] for row in data]

if __name__ == '__main__':
    main(questions[:1])
    # main(questions[:])
