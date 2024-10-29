import traceback
from abc import abstractmethod
from typing import Callable, Optional, Tuple

from opendevin.core.logger import opendevin_logger as logger
from opendevin.llm.llm import LLM


class BaseLLM:
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, *args, **kargs): ...


def IDENTITY(x):
    return x, True, None


class OpenDevinParserLLM(BaseLLM):
    def __init__(
        self,
        opendevin_llm: LLM,
        max_retries: int = 4,
        default_parser: Callable[[str], Tuple[str, bool, Optional[str]]] = IDENTITY,
    ):
        super().__init__()
        self.opendevin_llm = opendevin_llm
        self.default_parser = default_parser
        self.max_retries = max_retries
        self.cost_accumulator = 0

    def __call__(self, user_prompt, system_prompt=None, parser=None, **kwargs):
        if parser is None:
            parser = self.default_parser
        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        messages.append({'role': 'user', 'content': user_prompt})

        try:
            ans_dict = self._retry(
                messages, parser, n_retries=self.max_retries, **kwargs
            )
            ans_dict['n_retry'] = (len(messages) - 3) / 2
        except ValueError as e:
            # Likely due to maximum retry. We catch it here to be able to return
            # the list of messages for further analysis
            ans_dict = {}
            ans_dict['err_msg'] = str(e)
            ans_dict['stack_trace'] = traceback.format_exc()
            ans_dict['n_retries'] = self.max_retries

        ans_dict['messages'] = messages
        ans_dict['prompt'] = user_prompt

        return ans_dict

    def _retry(
        self,
        messages,
        parser,
        n_retries=4,
        min_retry_wait_time=60,
        rate_limit_max_wait_time=60 * 30,
        **kwargs,
    ):
        tries = 0
        rate_limit_total_delay = 0
        while tries < n_retries and rate_limit_total_delay < rate_limit_max_wait_time:
            response = self.opendevin_llm.completion(
                messages=messages,
                # messages=truncated_messages,  # added
                **kwargs,
            )
            answer = response['choices'][0]['message']['content'].strip()

            messages.append({'role': 'assistant', 'content': answer})

            value, valid, retry_message = parser(answer)
            if valid:
                self.log_cost(response)
                return value

            tries += 1
            msg = f'Query failed. Retrying {tries}/{n_retries}.\n[LLM]:\n{answer}\n[User]:\n{retry_message}'
            logger.info(msg)
            messages.append({'role': 'user', 'content': retry_message})

        raise ValueError(f'Could not parse a valid value after {n_retries} retries.')

    def log_cost(self, response):
        # TODO: refactor to unified cost tracking
        try:
            cur_cost = self.opendevin_llm.completion_cost(response)
        except Exception:
            cur_cost = 0
        self.cost_accumulator += cur_cost
        logger.info(
            'Cost: %.2f USD | Accumulated Cost: %.2f USD',
            cur_cost,
            self.cost_accumulator,
        )
