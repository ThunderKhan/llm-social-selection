from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..tasks import Task
from ..seeding import derive_seed


TASK_SOURCE_VERSION = "fixed-task-source-v1"


class TaskSourceError(ValueError):
    """A deterministic task schedule cannot be constructed."""


class TaskSource(ABC):
    @abstractmethod
    def task_for(self, *, trial_seed: int, round_index: int) -> Task:
        """Return one task for an explicit trial round."""


@dataclass(frozen=True)
class FixedTaskSource(TaskSource):
    tasks: tuple[Task, ...]

    def __post_init__(self) -> None:
        tasks = tuple(self.tasks)
        if not tasks:
            raise TaskSourceError("fixed task source must contain at least one task")
        if len({task.task_id for task in tasks}) != len(tasks):
            raise TaskSourceError("fixed task source task IDs must be unique")
        object.__setattr__(self, "tasks", tasks)

    def task_for(self, *, trial_seed: int, round_index: int) -> Task:
        seed = derive_seed(
            trial_seed,
            round_index,
            "task_schedule",
            TASK_SOURCE_VERSION,
        )
        return self.tasks[seed % len(self.tasks)]
