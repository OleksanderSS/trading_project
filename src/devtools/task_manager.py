# src/devtools/task_manager.py

import re
import time
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any

from src.core.file_management.file_manager import FileManager
from src.core.logging.logger import Logger as ProjectLogger

# Initialize logger for the module
logger = ProjectLogger.get_logger("TaskManager")

# Pre-compiled regex for finding TODO comments
TODO_PATTERN = re.compile(r"#\s*(TODO|FIXME|XXX|HACK|BUG):?\s*(.+)")

class TaskStatus(Enum):
    """Enumeration for the status of a task."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"

class TaskPriority(Enum):
    """Enumeration for the priority of a task."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class Task:
    """Represents a single development task or a TODO item."""

    def __init__(self, id: str, title: str, **kwargs):
        self.id = id
        self.title = title
        self.description: str = kwargs.get('description', "")
        self.status: TaskStatus = kwargs.get('status', TaskStatus.PENDING)
        self.priority: TaskPriority = kwargs.get('priority', TaskPriority.MEDIUM)
        self.assigned_to: Optional[str] = kwargs.get('assigned_to')
        self.created_at: datetime = kwargs.get('created_at', datetime.now())
        self.due_date: Optional[datetime] = kwargs.get('due_date')
        self.tags: List[str] = kwargs.get('tags', [])
        self.dependencies: List[str] = kwargs.get('dependencies', [])
        self.updated_at: datetime = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the task object to a dictionary."""
        return {
            'id': self.id,
            'title': self.title,
            'description': self.description,
            'status': self.status.value,
            'priority': self.priority.value,
            'assigned_to': self.assigned_to,
            'created_at': self.created_at.isoformat(),
            'due_date': self.due_date.isoformat() if self.due_date else None,
            'tags': self.tags,
            'dependencies': self.dependencies,
            'updated_at': self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task":
        """Deserializes a dictionary into a Task object."""
        return cls(
            id=data['id'],
            title=data['title'],
            description=data.get('description', ""),
            status=TaskStatus(data.get('status', TaskStatus.PENDING.value)),
            priority=TaskPriority(data.get('priority', TaskPriority.MEDIUM.value)),
            assigned_to=data.get('assigned_to'),
            created_at=datetime.fromisoformat(data.get('created_at')) if data.get('created_at') else datetime.now(),
            due_date=datetime.fromisoformat(data['due_date']) if data.get('due_date') else None,
            tags=data.get('tags', []),
            dependencies=data.get('dependencies', []),
        )

class TaskManager:
    """A centralized system for managing development tasks."""

    def __init__(self, file_manager: FileManager, storage_path: str = "tasks.json"):
        self.fm = file_manager
        self.storage_path = Path("devtools") / storage_path # Store tasks in a dedicated folder
        self.tasks: Dict[str, Task] = self._load_tasks()

    def _load_tasks(self) -> Dict[str, Task]:
        """Loads tasks from the storage file using FileManager."""
        data = self.fm.load_json(self.storage_path)
        if not data:
            logger.info("No existing tasks file found. Starting fresh.")
            return {}
        
        tasks = {task_id: Task.from_dict(task_data) for task_id, task_data in data.items()}
        logger.info(f"Successfully loaded {len(tasks)} tasks from {self.storage_path}")
        return tasks

    def _save_tasks(self) -> None:
        """Saves all current tasks to the storage file using FileManager."""
        data_to_save = {task_id: task.to_dict() for task_id, task in self.tasks.items()}
        self.fm.save_json(data_to_save, self.storage_path)
        logger.debug(f"Saved {len(self.tasks)} tasks to {self.storage_path}")

    def create_task(self, title: str, **kwargs) -> Task:
        """Creates a new task, adds it to the manager, and saves."""
        task_id = f"task_{int(time.time() * 1000)}_{len(self.tasks) + 1}"
        task = Task(id=task_id, title=title, **kwargs)
        self.tasks[task.id] = task
        self._save_tasks()
        logger.info(f"Created task '{task.id}': {task.title}")
        return task

    def get_task(self, task_id: str) -> Optional[Task]:
        """Retrieves a task by its ID."""
        return self.tasks.get(task_id)

    def update_task(self, task_id: str, **updates: Any) -> Optional[Task]:
        """Updates attributes of an existing task."""
        task = self.get_task(task_id)
        if not task:
            logger.warning(f"Cannot update. Task with ID '{task_id}' not found.")
            return None
        
        for key, value in updates.items():
            if hasattr(task, key):
                # Handle enums
                if key == 'status' and isinstance(value, str):
                    value = TaskStatus(value)
                if key == 'priority' and isinstance(value, str):
                    value = TaskPriority(value)
                setattr(task, key, value)

        task.updated_at = datetime.now()
        self._save_tasks()
        logger.info(f"Updated task '{task_id}'.")
        return task

    def list_tasks(self, **filters) -> List[Task]:
        """Lists tasks, optionally filtering by status, priority, etc."""
        filtered_tasks = list(self.tasks.values())
        
        if not filters:
            return sorted(filtered_tasks, key=lambda t: t.created_at, reverse=True)

        for key, value in filters.items():
            if value is None:
                continue
            
            # Convert string to Enum if necessary
            if key == 'status' and isinstance(value, str):
                value_enum = TaskStatus(value.lower())
                filtered_tasks = [t for t in filtered_tasks if t.status == value_enum]
            elif key == 'priority' and isinstance(value, str):
                value_enum = TaskPriority(value.lower())
                filtered_tasks = [t for t in filtered_tasks if t.priority == value_enum]
            elif key == 'assigned_to':
                filtered_tasks = [t for t in filtered_tasks if t.assigned_to == value]
            elif key == 'tag' or key == 'tags':
                search_tags = [value] if isinstance(value, str) else value
                filtered_tasks = [t for t in filtered_tasks if any(tag in t.tags for tag in search_tags)]

        return sorted(filtered_tasks, key=lambda t: t.created_at, reverse=True)

    def consolidate_codebase_todos(self, project_root: str = ".") -> List[Task]:
        """Scans the codebase for TODO comments and creates tasks."""
        newly_created_tasks = []
        project_path = Path(project_root)

        for py_file in project_path.rglob("*.py"):
            try:
                with py_file.open('r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        match = TODO_PATTERN.search(line)
                        if match:
                            todo_type, todo_text = match.groups()
                            task_title = f"[{todo_type.upper()}] {todo_text[:60]}..."
                            
                            # Avoid creating duplicate tasks
                            if any(task.title.startswith(f"[{todo_type.upper()}]") and todo_text[:60] in task.title for task in self.tasks.values()):
                                continue

                            task = self.create_task(
                                title=task_title,
                                description=f"Found in `{py_file.relative_to(project_path)}` at line {line_num}.\n\nFull text: {todo_text}",
                                priority=TaskPriority.HIGH if todo_type.upper() == 'FIXME' else TaskPriority.MEDIUM,
                                tags=['autogenerated', 'code-scan', todo_type.lower()],
                            )
                            newly_created_tasks.append(task)
            except Exception as e:
                logger.warning(f"Could not scan file {py_file}: {e}")

        logger.info(f"Consolidation complete. Found and created {len(newly_created_tasks)} new tasks from codebase.")
        return newly_created_tasks