from ..base import AgentModule


class BasePlanner(AgentModule): ...


class PolicyPlanner(AgentModule):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def __call__(self, state, memory, **kwargs):
        intent_response = self.policy(state=state, memory=memory, **kwargs)
        return intent_response


class WorldModelPlanner(AgentModule):
    def __init__(self, policy, world_model, critic):
        super().__init__()
        self.policy = policy
        self.world_model = world_model
        self.critic = critic
        self.reasoner = None

    def __call__(self, belief):
        intent = self.reasoner(belief)
        return intent
