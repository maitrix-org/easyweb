import json
import os
from abc import ABC, abstractmethod

from easyweb.core.config import config
from easyweb.core.schema import CancellableStream
from easyweb.runtime.docker.process import Process
from easyweb.runtime.plugins.mixin import PluginMixin


class Sandbox(ABC, PluginMixin):
    background_commands: dict[int, Process] = {}
    _env: dict[str, str] = {}
    is_initial_session: bool = True

    def __init__(self, **kwargs):
        for key in os.environ:
            if key.startswith('SANDBOX_ENV_'):
                sandbox_key = key.removeprefix('SANDBOX_ENV_')
                self.add_to_env(sandbox_key, os.environ[key])
        if config.enable_auto_lint:
            self.add_to_env('ENABLE_AUTO_LINT', 'true')
        self.initialize_plugins: bool = config.initialize_plugins

    def add_to_env(self, key: str, value: str):
        self._env[key] = value
        # Note: json.dumps gives us nice escaping for free
        self.execute(f'export {key}={json.dumps(value)}')

    @abstractmethod
    def execute(
        self, cmd: str, stream: bool = False, timeout: int | None = None
    ) -> tuple[int, str | CancellableStream]:
        pass

    @abstractmethod
    def execute_in_background(self, cmd: str) -> Process:
        pass

    @abstractmethod
    def kill_background(self, id: int) -> Process:
        pass

    @abstractmethod
    def read_logs(self, id: int) -> str:
        pass

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def copy_to(self, host_src: str, sandbox_dest: str, recursive: bool = False):
        pass

    @abstractmethod
    def get_working_directory(self):
        pass
