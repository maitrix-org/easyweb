import json
import os
import sys
from glob import glob

from evaluator import FlightSearchEvaluator
from helpers import tqdm_joblib
from joblib import Parallel, delayed
from tqdm import tqdm

# '/Users/mingkaid/Documents/Tech-Life/lwm/web-agent-application/formal_evaluation_logs/onepass_8b_inst_flight_logs'
if __name__ == '__main__':
    evaluator = FlightSearchEvaluator('./task_data/flight_questions_train.jsonl')

    log_dir = sys.argv[1]
    log_paths = glob(os.path.join(log_dir, '*.json'))

    with tqdm_joblib(tqdm(desc='Evaluating', disable=False)) as progress_bar:
        results = Parallel(n_jobs=4, backend='threading')(
            delayed(evaluator.evaluate)(path) for path in log_paths
        )

    result_metrics = evaluator.get_result_metrics(results)
    result_metrics['log_dir'] = log_dir
    result_metrics['num_logs'] = len(log_paths)
    with open('./result_metrics.jsonl', 'a') as f:
        f.write(json.dumps(result_metrics) + '\n')
