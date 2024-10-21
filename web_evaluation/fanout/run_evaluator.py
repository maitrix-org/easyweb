from glob import glob

from evaluator import FanOutQAEvaluator

# '/Users/mingkaid/Documents/Tech-Life/lwm/web-agent-application/formal_evaluation_logs/onepass_8b_inst_fanout_logs/*.json'

if __name__ == '__main__':
    evaluator = FanOutQAEvaluator('./task_data/fanout-dev-0-30.json')

    my_evaluator_log_paths = glob('my_evaluator_logs/*.json')

    scores, records = evaluator.evaluate_batch(my_evaluator_log_paths)
    print(scores)
