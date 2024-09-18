import json

from my_evaluator import main

data = json.load(open('./assistantbench-dev-0-33.json'))

questions = [row['task'] for row in data]

if __name__ == '__main__':
    main(questions[:])
