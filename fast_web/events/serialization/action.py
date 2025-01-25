from fast_web.core.exceptions import AgentMalformedActionError
from fast_web.events.action.action import Action
from fast_web.events.action.agent import (
    AgentDelegateAction,
    AgentFinishAction,
    AgentRecallAction,
    AgentRejectAction,
    ChangeAgentStateAction,
)
from fast_web.events.action.browse import BrowseInteractiveAction, BrowseURLAction
from fast_web.events.action.commands import (
    CmdKillAction,
    CmdRunAction,
    IPythonRunCellAction,
)
from fast_web.events.action.empty import NullAction
from fast_web.events.action.files import FileReadAction, FileWriteAction
from fast_web.events.action.message import MessageAction
from fast_web.events.action.planning import FinishPlanningAction, StartPlanningAction
from fast_web.events.action.tasks import AddTaskAction, ModifyTaskAction

actions = (
    NullAction,
    CmdKillAction,
    CmdRunAction,
    IPythonRunCellAction,
    BrowseURLAction,
    BrowseInteractiveAction,
    FileReadAction,
    FileWriteAction,
    AgentRecallAction,
    AgentFinishAction,
    AgentRejectAction,
    AgentDelegateAction,
    AddTaskAction,
    ModifyTaskAction,
    ChangeAgentStateAction,
    MessageAction,
    StartPlanningAction,
    FinishPlanningAction,
)

ACTION_TYPE_TO_CLASS = {action_class.action: action_class for action_class in actions}  # type: ignore[attr-defined]


def action_from_dict(action: dict) -> Action:
    if not isinstance(action, dict):
        raise AgentMalformedActionError('action must be a dictionary')
    action = action.copy()
    if 'action' not in action:
        raise AgentMalformedActionError(f"'action' key is not found in {action=}")
    if not isinstance(action['action'], str):
        raise AgentMalformedActionError(
            f"'{action['action']=}' is not defined. Available actions: {ACTION_TYPE_TO_CLASS.keys()}"
        )
    action_class = ACTION_TYPE_TO_CLASS.get(action['action'])
    if action_class is None:
        raise AgentMalformedActionError(
            f"'{action['action']=}' is not defined. Available actions: {ACTION_TYPE_TO_CLASS.keys()}"
        )
    args = action.get('args', {})
    try:
        decoded_action = action_class(**args)
    except TypeError:
        raise AgentMalformedActionError(f'action={action} has the wrong arguments')
    return decoded_action
