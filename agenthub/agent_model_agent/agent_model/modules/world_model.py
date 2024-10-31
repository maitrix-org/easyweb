from ..base import AgentModule


class BaseWorldModel(AgentModule):
    def __init__(self, identity):
        self.identity = identity

    def __call__(self, state, memory, intent, **kwargs):
        raise NotImplementedError


class PromptedWorldModel(BaseWorldModel):
    def __init__(self, identity, llm, prompt_template):
        super().__init__(identity)
        self.llm = llm
        self.prompt_template = prompt_template

    def __call__(self, state, memory, intent, **kwargs):
        user_prompt = self.prompt_template.format(
            state=state, memory=memory, intent=intent, **kwargs
        )
        llm_output = self.llm(
            system_prompt=str(self.identity), user_prompt=user_prompt, **kwargs
        )

        return llm_output
