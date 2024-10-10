import json

from my_evaluator import main

data = []
with open('./notebooks/questions_train.jsonl') as f:
    for line in f:
        data.append(json.loads(line))

questions = [row['question'] for row in data if row['level'] < 3]
print(questions)

if __name__ == '__main__':
    main(questions[:])
