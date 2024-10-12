import multiprocessing.dummy as mp_dummy
import warnings
from collections import defaultdict

import numpy as np
from nltk.translate.bleu_score import sentence_bleu

# For deduping purpose
from openai import OpenAI

from . import prompt_factory
from .reasoners import SearchConfig, WorldModel

client_llama = OpenAI(
    base_url='http://localhost:8000/v1',
)


def apply_function(function, args, kwargs):
    return function(*args, **kwargs)


class WebWorldModel(WorldModel):
    def __init__(self, get_llm_output_fn, add_to_log_fn=None):
        super().__init__()
        self.get_llm_output_fn = get_llm_output_fn
        self.add_to_log_fn = add_to_log_fn

    def init_state(self):
        return {
            'history': self.history,
            'action_history': [],
            'current_state': self.current_state,
            'goal': self.goal,
        }

    def step(self, state, action):
        """World Model"""
        action, freq = action
        world_model_prompt = prompt_factory.get_world_model_prompt(
            state['current_state'], action, state['history'], state['goal']
        )
        answer_keys = ['next_state', 'reward', 'termination']
        answer_dict = self.get_llm_output_fn(world_model_prompt, answer_keys)

        for key in answer_keys:
            self.add_to_log_fn(key, answer_dict[key])

        next_state = {
            'history': state['history']
            + [(None, state['current_state'], action, None)],
            'action_history': state['action_history'] + [action],
            'current_state': answer_dict['next_state'],
            'reward': answer_dict['reward'],
            'termination': answer_dict['termination'],
            'goal': state['goal'],
        }

        status = {
            'reward': answer_dict['reward'],
            'termination': answer_dict['termination'],
        }

        return next_state, status

    def is_terminal(self, state):
        return state['termination'] == 'yes'

    def update_example(self, example, **kwargs):
        super().update_example(example, **kwargs)
        self.history = example['history']
        self.current_state = example['current_state']
        self.goal = example['goal']


def dedup_and_match_actions(action_count_dict):
    action_list = list(action_count_dict.keys())
    for _ in range(5):
        try:
            prompt = f"""\
Example 1:

Here is a list of actions:

['Fill the search box with the query "Josh Hamilton batting hand"', 'Click into the ESPN result for the 1998 MLB draft', 'Fill the search box with query "Josh Hamilton batting hand"']

Here are the semantically unique actions from the list above:

['Fill the search box with the query "Josh Hamilton batting hand"', 'Click into the ESPN result for the 1998 MLB draft']

Example 2:

Here is a list of actions:

{action_list}

Here are the semantically unique actions from the list above:
"""

            completion = client_llama.chat.completions.create(
                model='Meta-Llama-3.1-70B-Instruct',
                messages=[
                    {'role': 'system', 'content': 'You are a helpful assistant.'},
                    {'role': 'user', 'content': prompt},
                ],
                stop='\n',
            )
            response = completion.choices[0].message.content
            deduped_action_list = eval(response)

            # Find a mapping between each deduped action to the original actions
            similarities = []
            for deduped_action in deduped_action_list:
                new_to_old_sims = []
                for original_action in action_list:
                    sim = 0
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        sim = sentence_bleu(
                            [original_action.split()],
                            deduped_action.split(),
                            weights=(0.75, 0.25, 0, 0),
                        )
                    new_to_old_sims.append(sim)
                similarities.append(new_to_old_sims)

            deduped_action_count_dict = defaultdict(lambda: 0)
            assignment = np.argmax(similarities, axis=0)
            for assignment_id, old_action in zip(assignment, action_list):
                deduped_action = deduped_action_list[assignment_id]
                deduped_action_count_dict[deduped_action] += action_count_dict[
                    old_action
                ]

            return deduped_action_count_dict

        except Exception as e:
            print(e)

    return action_count_dict


class WebSearchConfig(SearchConfig):
    def __init__(
        self,
        get_llm_output_fn,
        num_actions,
        max_retries=3,
        reward_scale=0.7,
        add_to_log_fn=None,
    ):
        super().__init__()
        self.get_llm_output_fn, self.add_to_log_fn = get_llm_output_fn, add_to_log_fn
        self.num_actions, self.max_retries, self.reward_scale = (
            num_actions,
            max_retries,
            reward_scale,
        )

    def get_actions(self, state):
        policy_prompt = prompt_factory.get_policy_prompt(
            state['current_state'], state['history'], state['goal']
        )

        # num_retries = 0
        # actions = set()
        actions = defaultdict(lambda: 0)
        # while len(actions) < self.num_actions and num_retries < self.max_retries:
        with mp_dummy.Pool(processes=self.num_actions * self.max_retries) as pool:
            action_dicts = pool.starmap(
                apply_function,
                [
                    (
                        self.get_llm_output_fn,
                        (policy_prompt, ['instruction']),
                        {'temperature': 1.0},
                    )
                ]
                * (self.num_actions * self.max_retries),
            )
        for action_dict in action_dicts:
            # self.add_to_log_fn('think', action_dict['think'])
            # self.add_to_log_fn('instruction', action_dict['instruction'])
            # actions.add(action_dict['instruction'])
            actions[action_dict['instruction']] += 1
            # num_retries += 1

        actions = dedup_and_match_actions(actions)
        actions = [(action, count) for action, count in actions.items()]
        actions = sorted(actions, key=lambda x: -x[1])
        actions = actions[: self.num_actions]

        # actions = np.random.choice(list(actions), min(len(actions), self.num_actions), replace=False)
        # actions = np.random.choice(action_list, min(len(action_list), self.num_actions), replace=False)
        # actions = list(actions)
        self.add_to_log_fn('action_space', actions)

        return actions

    def fast_reward(self, state, action):
        if not isinstance(action, list):
            action = [action]

        total_action_counts = sum([count for _, count in action])
        action_freqs = [(act, count / total_action_counts) for act, count in action]
        fast_rewards = [freq for _, freq in action_freqs]
        reward_dicts = [{'intuition': freq} for _, freq in action_freqs]

        return fast_rewards, reward_dicts

    def reward(self, state, action, intuition, reward, termination, **kwargs):
        reward2score = {'closer-to-goal': 1, 'further-from-goal': -1, 'neutral': 0}
        reward_score = self.reward_scale * reward2score.get(reward, 0)
        termination_signal = termination == 'yes'
        reward = self.reward_scale * (
            intuition + reward_score + int(termination_signal)
        )

        return reward, {
            'reward_score': reward_score,
            'termination_signal': termination_signal,
        }

    def update_example(self, example, **kwargs):
        super().update_example(example, **kwargs)
        # self.state_history = example['state_history']
        self.history = example['history']
        self.current_state = example['current_state']
        self.goal = example['goal']
