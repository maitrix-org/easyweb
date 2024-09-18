import json

from my_evaluator import main

data = json.load(open('./fanout-dev-0-30.json'))

questions = ['Using Google, ' + row['question'] for row in data]

if __name__ == '__main__':
    main(questions[:20])
