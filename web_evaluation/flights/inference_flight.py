import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from web_evaluation.inference import main

data = []
with open('task_data/flight_questions_train.jsonl') as f:
    for line in f:
        data.append(json.loads(line))

questions = [row['question'] for row in data if row['level'] < 3]
print(questions)

if __name__ == '__main__':
    # skip = [2, 4, 1, 6, 7, 9, 10, 11, 12, 16]
    main(questions[:5])
    # main([questions[i] for i in range(len(questions)) if i not in skip])
