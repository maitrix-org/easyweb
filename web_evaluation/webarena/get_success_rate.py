import argparse
import json
import os

import browsergym.webarena  # noqa F401 register webarena tasks as gym environments

parser = argparse.ArgumentParser(description='Calculate average reward.')
# parser.add_argument('output_path', type=str, help='path to output.jsonl')
parser.add_argument(
    'output_root', type=str, help='path to dictionary containing output.jsonl'
)
parser.add_argument('--index_success', '-v', action='store_true')

args = parser.parse_args()

if __name__ == '__main__':
    # env_ids = [
    #     id for id in gym.envs.registry.keys() if id.startswith('browsergym/webarena')
    # ]
    env_ids = [
        f'browsergym/{fname}'
        for fname in os.listdir(args.output_root)
        if fname.startswith('webarena.')
    ]
    total_num = len(env_ids)
    print('Total number of tasks: ', total_num)
    total_reward = 0
    total_cost = 0
    actual_num = 0
    out_path = os.path.join(args.output_root, 'output.jsonl')
    success_cases = []
    with open(out_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            actual_num += 1
            total_cost += data['metrics']['accumulated_cost']
            total_reward += data['test_result']
            if args.index_success and data['test_result'] > 0:
                success_cases.append(int(data['instance_id'].split('.')[-1]))
    if args.index_success:
        print(f'Successful instances: {str(sorted(success_cases))}')

    avg_reward = total_reward / total_num
    print('Success Rate: ', avg_reward)

    avg_cost = total_cost / actual_num
    print('Avg Cost: ', avg_cost)
    print('Actual number of tasks finished: ', actual_num)
