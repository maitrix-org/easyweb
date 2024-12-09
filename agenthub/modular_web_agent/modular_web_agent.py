import json
import os
from datetime import datetime
from functools import partial

from opendevin.controller.agent import Agent
from opendevin.events.action import Action
from opendevin.llm.llm import LLM
from opendevin.runtime.plugins import (
    PluginRequirement,
)
from opendevin.runtime.tools import RuntimeTool

from .agent_configs import default_config, webarena_config
from .agent_model.llms import OpenDevinParserLLM, OpenDevinParserMultiResponseLLM
from .agent_model.modules import (
    LLMReasonerPlanner,
    PolicyPlanner,
    PromptedActor,
    PromptedCritic,
    PromptedEncoder,
    PromptedPolicy,
    PromptedWorldModel,
    StateMemoryUpdateEncoder,
)
from .agent_model.variables import (
    AgentInstructionEnvironmentIdentity,
    OpenDevinBrowserActionSpace,
    OpenDevinBrowserObservationSpace,
    PromptedMemory,
    StepKeyValueMemory,
    StepPromptedMemory,
)
from .agent_prompts import (
    actor_concice_instruction,
    actor_prompt_template_dict,
    critic_prompt_template,
    encoder_memory_prompt_template,
    encoder_prompt_template_dict,
    memory_prompt_template,
    policy_inline_example_dict_with_nav,
    policy_prompt_template_dict,
    world_model_prompt_template_dict,
)
from .logger import AgentLogger
from .utils import ParseError, parse_html_tags_raise

AGENT_SELECTION = os.environ.get('AGENT_SELECTION', 'default')


def dict_update(d1, d2):
    d = d1.copy()
    d.update(d2)
    return d


AGENT_LIBRARY = {
    'default': default_config,
    'webarena': dict_update(default_config, webarena_config),
    'webarena_noplan': dict_update(
        dict_update(default_config, webarena_config),
        {'use_world_model_planning': False},
    ),
    'webarena_plan': dict_update(
        dict_update(default_config, webarena_config),
        {'use_world_model_planning': True},
    ),
}


def parser(text, keys, optional_keys=()):
    try:
        ans_dict = parse_html_tags_raise(text, keys, optional_keys)
    except ParseError as e:
        return None, False, str(e)
    return ans_dict, True, ''


class ModularWebAgent(Agent):
    VERSION = '0.1'
    """
    An agent that uses agent model abstractions to interact with the browser.
    """

    sandbox_plugins: list[PluginRequirement] = []
    runtime_tools: list[RuntimeTool] = [RuntimeTool.BROWSER]

    def __init__(
        self,
        llm: LLM,
    ) -> None:
        """
        Initializes a new instance of the AbstractBrowsingAgent class.

        Parameters:
        - llm (LLM): The llm to be used by this agent
        """
        super().__init__(llm)

        agent_config = AGENT_LIBRARY[AGENT_SELECTION]

        self.use_state_memory_encoder = agent_config['use_state_memory_encoder']
        self.memory_type = agent_config['memory_type']
        self.encoder_prompt_type = agent_config['encoder_prompt_type']
        self.policy_prompt_type = agent_config['policy_prompt_type']
        self.actor_prompt_type = agent_config['actor_prompt_type']
        self.world_model_prompt_type = agent_config['world_model_prompt_type']
        self.use_world_model_planning = agent_config['use_world_model_planning']

        self.action_space = OpenDevinBrowserActionSpace(
            action_subsets=['chat', 'bid'],
            use_nav=agent_config['use_nav'],
            strict=False,
            multiaction=False,
        )
        self.observation_space = OpenDevinBrowserObservationSpace(
            eval_mode=agent_config['eval_mode'],
            truncation=agent_config['truncate_axtree'],
        )

        # Agent identity
        self.identity = AgentInstructionEnvironmentIdentity(
            agent_name=agent_config['agent_name'],
            agent_description=agent_config['agent_description'],
            observation_space=self.observation_space,
            action_space=self.action_space,
        )

        # Encoder
        encoder_parser = partial(parser, keys=['state'])
        self.encoder_llm = OpenDevinParserLLM(llm, default_parser=encoder_parser)
        encoder_prompt_template = encoder_prompt_template_dict[self.encoder_prompt_type]

        if self.use_state_memory_encoder:
            memory_update_parser = partial(parser, keys=['memory_update'])
            self.memory_update_llm = OpenDevinParserLLM(
                llm, default_parser=memory_update_parser
            )
            self.encoder = StateMemoryUpdateEncoder(
                self.identity,
                self.encoder_llm,
                encoder_prompt_template,
                self.memory_update_llm,
                encoder_memory_prompt_template,
            )
        else:
            self.encoder = PromptedEncoder(
                self.identity, self.encoder_llm, prompt_template=encoder_prompt_template
            )

        # Memory
        if self.memory_type == 'prompted':
            memory_parser = partial(parser, keys=['updated_memory'])
            self.memory_llm = OpenDevinParserLLM(llm, default_parser=memory_parser)
            self.memory = PromptedMemory(
                self.identity, self.memory_llm, prompt_template=memory_prompt_template
            )
        elif self.memory_type == 'step_prompted':
            memory_update_parser = partial(parser, keys=['memory_update'])
            self.memory_update_llm = OpenDevinParserLLM(
                llm, default_parser=memory_update_parser
            )
            memory_update_prompt_template = encoder_memory_prompt_template
            self.memory = StepPromptedMemory(
                self.identity,
                self.memory_update_llm,
                prompt_template=memory_update_prompt_template,
                keys=['intent'],
            )
        elif self.memory_type == 'step_key_value':
            self.memory = StepKeyValueMemory(['state', 'intent'])
        elif self.memory_type == 'step_key_value_intent_only':
            self.memory = StepKeyValueMemory(['intent'])
        else:
            raise ValueError(f'Invalid memory type: {self.memory_type}')

        # Planner
        policy_prompt_template = policy_prompt_template_dict[self.policy_prompt_type]
        policy_prompt_template = policy_prompt_template.format(
            inline_example=policy_inline_example_dict_with_nav[agent_config['use_nav']]
        )
        if self.use_world_model_planning:
            policy_parser = partial(parser, keys=['intent'], optional_keys=['think'])
            self.policy_llm = OpenDevinParserMultiResponseLLM(
                llm, default_parser=policy_parser
            )
            self.policy = PromptedPolicy(
                self.identity, self.policy_llm, prompt_template=policy_prompt_template
            )

            world_model_parser = partial(parser, keys=['next_state'])
            self.world_model_llm = OpenDevinParserLLM(
                llm, default_parser=world_model_parser
            )
            world_model_prompt_template = world_model_prompt_template_dict[
                self.world_model_prompt_type
            ]
            self.world_model = PromptedWorldModel(
                self.identity,
                self.world_model_llm,
                prompt_template=world_model_prompt_template,
            )

            critic_parser = partial(
                parser, keys=['status', 'on_the_right_track'], optional_keys=['think']
            )
            self.critic_llm = OpenDevinParserMultiResponseLLM(
                llm, default_parser=critic_parser
            )
            self.critic = PromptedCritic(
                self.identity, self.critic_llm, prompt_template=critic_prompt_template
            )

            # self.planner = PolicyPlanner(self.policy)
            self.planner = LLMReasonerPlanner(
                self.policy, self.world_model, self.critic
            )

        else:
            policy_parser = partial(parser, keys=['intent'], optional_keys=['think'])
            self.policy_llm = OpenDevinParserLLM(llm, default_parser=policy_parser)
            self.policy = PromptedPolicy(
                self.identity, self.policy_llm, prompt_template=policy_prompt_template
            )

            self.planner = PolicyPlanner(self.policy)

        # Actor
        action_parser = partial(parser, keys=['action'])
        self.actor_llm = OpenDevinParserLLM(llm, default_parser=action_parser)
        actor_prompt_template = actor_prompt_template_dict[self.actor_prompt_type]
        if agent_config['use_actor_concice_instruction']:
            actor_prompt_template = actor_prompt_template.format(
                concise_instruction=actor_concice_instruction
            )
        else:
            actor_prompt_template = actor_prompt_template.format(concise_instruction='')
        self.actor = PromptedActor(
            self.identity, self.actor_llm, prompt_template=actor_prompt_template
        )

        self.reset()

    def reset(self):
        self.identity.reset()
        self.memory.reset()
        timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
        log_file = f'{timestamp}.log'
        self.logger = AgentLogger(log_file)
        self.planner.logger = self.logger

        self.last_action = ''
        self.num_repeats = 0

    def step(self, env_state) -> Action:
        observation, info = self.observation_space.parse_observation(env_state)
        if info.get('return_action') is not None:
            return info['return_action']
        self.identity.update(user_instruction=observation['goal'])

        obs_txt = observation['clean_axtree_txt']
        # logger.info(f'*Observation*: {obs_txt}')
        self.logger.info(f'*Observation*: {obs_txt}')

        kwargs = {}
        if self.use_state_memory_encoder:
            llm_output = self.encoder(obs_txt, self.memory)
            state, memory_update = llm_output['state'], llm_output['memory_update']
            self.logger.info(f'*State*: {state}')
            self.logger.info(f'*Memory update*: {memory_update}')
            kwargs['memory_update'] = memory_update
        else:
            state = self.encoder(obs_txt, self.memory)['state']
            self.logger.info(f'*State*: {state}')

        intent = self.planner(state, self.memory, **kwargs)['intent']
        self.logger.info(f'*Intent*: {intent}')

        action = self.actor(obs_txt, state, self.memory, intent, **kwargs)['action']
        self.logger.info(f'*Action*: {action}')

        if self.use_state_memory_encoder:
            step = {
                'observation': observation,
                'state': memory_update,
                'state_original': state,
                'intent': intent,
                'action': action,
            }
        else:
            step = {
                'observation': observation,
                'state': state,
                'intent': intent,
                'action': action,
            }
        self.memory.update(**step)
        step.update(self.memory.current_step)
        if self.memory_type == 'step_prompted':
            self.logger.info(
                f"*Memory update*: {self.memory.current_step['memory_update']}"
            )
        self.memory.step()

        if not action.startswith('scroll') and action == self.last_action:
            self.num_repeats += 1
        else:
            self.num_repeats = 0
            self.last_action = action

        if self.num_repeats >= 3:
            action = 'send_msg_to_user("Repetitive actions. Ending the task.")'
            step['action'] = action

        return self.action_space.parse_action(action, thought=json.dumps(step))

    def search_memory(self, query: str) -> list[str]:
        raise NotImplementedError('Implement this abstract method')
