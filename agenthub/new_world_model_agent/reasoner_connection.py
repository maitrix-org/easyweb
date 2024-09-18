import multiprocessing.dummy as mp_dummy

from . import prompt_factory
from .reasoners import SearchConfig, WorldModel


def apply_function(function, args, kwargs):
    return function(*args, **kwargs)


class WebWorldModel(WorldModel):
    def __init__(self, get_llm_output_fn, add_to_log_fn=None):
        super().__init__()
        self.get_llm_output_fn = get_llm_output_fn
        self.add_to_log_fn = add_to_log_fn

    def init_state(self):
        return {'partial_plan': [(None, self.current_state_dict)]}

    def step(self, state, action):
        """World Model"""
        next_state_prompt_answer_key_dict = prompt_factory.get_next_state_prompt(
            action['strategy'], state['partial_plan'], self.state_history
        )

        with mp_dummy.Pool(processes=len(next_state_prompt_answer_key_dict)) as pool:
            arguments = [
                (prompt, keys)
                for prompt, keys in next_state_prompt_answer_key_dict.items()
            ]
            answer_dicts = pool.starmap(
                apply_function, [(self.get_llm_output_fn, arg, {}) for arg in arguments]
            )

        next_state_dict = {}
        for ans_dict in answer_dicts:
            next_state_dict.update(ans_dict)

        if self.add_to_log_fn is not None:
            for key in [
                'summary',
                'content',
                'progress',
                'reflection',
                'think',
                'completion',
            ]:
                value = next_state_dict.get(key, 'None')
                if value != 'None':
                    self.add_to_log_fn(key, value)

        next_state = dict(state)
        next_state['partial_plan'].append((action['strategy'], next_state_dict))

        return next_state, {'completion': next_state_dict['completion']}

    def is_terminal(self, state):
        last_state = state['partial_plan'][-1][1]
        return last_state['completion'].strip('"') == 'finished'

    def update_example(self, example, **kwargs):
        super().update_example(example, **kwargs)
        self.current_state_dict = example['current_state_dict']
        self.state_history = example['state_history']


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
        strategy_prompt, answer_keys = prompt_factory.get_strategy_prompt(
            state['partial_plan'], self.state_history
        )
        seen_actions = set()
        all_action_dicts = []

        num_retries = 0
        while (
            len(all_action_dicts) < self.num_actions and num_retries < self.max_retries
        ):
            with mp_dummy.Pool(processes=self.num_actions) as pool:
                arguments = [(strategy_prompt, answer_keys)] * self.num_actions
                action_dicts = pool.starmap(
                    apply_function,
                    [
                        (self.get_llm_output_fn, arg, {'temperature': 1.0})
                        for arg in arguments
                    ],
                )
            for action_dict in action_dicts:
                if action_dict['strategy'] in seen_actions:
                    continue
                all_action_dicts.append(action_dict)
                seen_actions.add(action_dict['strategy'])
            num_retries += 1

        # if self.add_to_log_fn is not None:
        #     for action in all_actions:
        #         self.add_to_log_fn('action_candidate', action)

        return all_action_dicts

    def fast_reward(self, state, action):
        if isinstance(action, str) or isinstance(action, dict):
            action = [action]
        arguments = []
        for act in action:
            prompt, answer_keys = prompt_factory.get_action_reward_prompt(
                act['strategy'], self.state_history, state['partial_plan']
            )
            arguments.append((prompt, answer_keys))
        # action_reward_prompt, answer_keys = prompt_factory.get_action_reward_prompt(action, self.state_history, state['partial_plan'])
        with mp_dummy.Pool(processes=min(self.num_actions, len(action))) as pool:
            # arguments = [(action_reward_prompt, answer_keys)] * self.num_actions
            answer_dicts = pool.starmap(
                apply_function, [(self.get_llm_output_fn, arg, {}) for arg in arguments]
            )

        if self.add_to_log_fn is not None:
            for act_dict, ans_dict in zip(action, answer_dicts):
                self.add_to_log_fn('action_think', act_dict['think'])
                self.add_to_log_fn('action_candidate', act_dict['strategy'])
                self.add_to_log_fn('reward_think', ans_dict.get('think', 'None'))
                self.add_to_log_fn('reward', ans_dict.get('response', 'None'))

        response2reward = {
            'goal-achieved': 2,
            'towards-the-goal': 1,
            'not-sure': 0,
            'away-from-the-goal': -1,
        }

        rewards = [
            self.reward_scale * response2reward.get(ans_dict['response'].strip('"'), 0)
            for ans_dict in answer_dicts
        ]
        for ans_dict, reward in zip(answer_dicts, rewards):
            ans_dict['intuition'] = reward
        # if len(rewards) == 1:
        #     return rewards[0], answer_dicts[0]
        return rewards, answer_dicts

    def reward(self, state, action, completion, intuition, **kwargs):
        goal_reached = completion.strip('"') == 'finished'
        reward = self.reward_scale * (intuition + int(goal_reached))

        return reward, {'intuition': intuition, 'goal_reached': goal_reached}

    def update_example(self, example, **kwargs):
        super().update_example(example, **kwargs)
        self.state_history = example['state_history']
