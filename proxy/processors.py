import importlib.util
import sys
from pathlib import Path

from tasks.base import Processor

_registry: dict[str, Processor] = {}
_default = Processor()


def register(task_name: str, processor: Processor) -> None:
    """
    Register a processor for the given task name.

    Parameters
    ----------
    task_name : str
        Task name that triggers this processor.
    processor : Processor
        Processor instance to use for the task.
    """
    _registry[task_name] = processor


def get_processor(task_name: str) -> Processor:
    """
    Return the processor for a task, or the no-op default.

    Parameters
    ----------
    task_name : str
        Task or model name.

    Returns
    -------
    Processor
        Registered processor or no-op passthrough.
    """
    return _registry.get(task_name, _default)


def _discover_task_processors() -> None:
    tasks_root = Path(__file__).parent.parent / "tasks"
    if not tasks_root.exists():
        tasks_root = Path("/app/tasks")
    if not tasks_root.exists():
        return

    for task_dir in sorted(tasks_root.iterdir()):
        processor_file = task_dir / "processor.py"
        if not processor_file.is_file():
            continue

        module_name = f"tasks.{task_dir.name}.processor"
        spec = importlib.util.spec_from_file_location(module_name, processor_file)
        if spec is None or spec.loader is None:
            continue

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)  # type: ignore[union-attr]

        if hasattr(module, "register_processor"):
            module.register_processor(register)


_discover_task_processors()

__all__ = ["Processor", "get_processor", "register"]
