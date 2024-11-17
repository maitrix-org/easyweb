default_config = {
    'agent_name': 'Web Browsing Agent',
    'agent_description': """An information and automation assistant who responds to \
user instructions by browsing the internet. The assistant strives to answer each question \
accurately, thoroughly, efficiently, and politely, and to be forthright when it is \
impossible to answer the question or carry out the instruction. The assistant will \
end the task once it sends a message to the user.""",
    'use_nav': True,
    'use_actor_concice_instruction': False,
    'use_state_memory_encoder': False,
    'memory_type': 'step_prompted',
    'encoder_prompt_type': 'no_memory',
    'policy_prompt_type': 'no_update',
    'actor_prompt_type': 'with_memory',
    'world_model_prompt_type': 'no_memory_with_update',
    'use_world_model_planning': True,
}

webarena_config = {
    'agent_description': """An information and automation assistant who responds to \
user instructions by browsing the internet. The response follows the following rules: \
1. When the intent is a question, and a complete answer to the question has been found, \
then send the answer; 2. the intent wants to locate specific information or navigate to \
a particular section of a site, and the current page satisfies, then send the url; \
3. the intent want to conduct an operation, and has been done, then send "Operation complete."
The assistatnt should try to acheive the goal in the current site without navigating to sites \
like Google. Be forthright when it is impossible to answer the question or carry out the task. \
The assistant will end the task once it sends a message to the user.""",
    'use_nav': False,
    'use_actor_concice_instruction': True,
}
