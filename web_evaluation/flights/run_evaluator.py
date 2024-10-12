from glob import glob

from evaluator import FlightSearchEvaluator

if __name__ == '__main__':
    evaluator = FlightSearchEvaluator('./task_data/flight_questions_train.jsonl')

    my_evaluator_log_paths = glob('./my_evaluator_logs/*')

    n_correct = 0
    for path in my_evaluator_log_paths:
        n_correct += evaluator.evaluate(path)['correct']
    print(n_correct / len(my_evaluator_log_paths))
