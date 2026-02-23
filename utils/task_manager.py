# utils/task_manager.py

"""
Task management system for TODO items and development tasks
"""

import json
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task status enumeration"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    """Task priority enumeration"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Task:
    """Individual task representation"""
    
    def __init__(
        self,
        id: str,
        title: str,
        description: str = "",
        status: TaskStatus = TaskStatus.PENDING,
        priority: TaskPriority = TaskPriority.MEDIUM,
        assigned_to: Optional[str] = None,
        created_at: Optional[datetime] = None,
        due_date: Optional[datetime] = None,
        tags: Optional[List[str]] = None,
        dependencies: Optional[List[str]] = None
    ):
        self.id = id
        self.title = title
        self.description = description
        self.status = status
        self.priority = priority
        self.assigned_to = assigned_to
        self.created_at = created_at or datetime.now()
        self.due_date = due_date
        self.tags = tags or []
        self.dependencies = dependencies or []
        self.updated_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary"""
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
            'updated_at': self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Task':
        """Create task from dictionary"""
        return cls(
            id=data['id'],
            title=data['title'],
            description=data.get('description', ''),
            status=TaskStatus(data.get('status', TaskStatus.PENDING.value)),
            priority=TaskPriority(data.get('priority', TaskPriority.MEDIUM.value)),
            assigned_to=data.get('assigned_to'),
            created_at=datetime.fromisoformat(data['created_at']) if data.get('created_at') else None,
            due_date=datetime.fromisoformat(data['due_date']) if data.get('due_date') else None,
            tags=data.get('tags', []),
            dependencies=data.get('dependencies', [])
        )


class TaskManager:
    """Centralized task management system"""
    
    def __init__(self, storage_path: str = "data/tasks.json"):
        self.storage_path = Path(storage_path)
        self.tasks: Dict[str, Task] = {}
        self.load_tasks()
    
    def create_task(
        self,
        title: str,
        description: str = "",
        priority: TaskPriority = TaskPriority.MEDIUM,
        assigned_to: Optional[str] = None,
        due_date: Optional[datetime] = None,
        tags: Optional[List[str]] = None,
        dependencies: Optional[List[str]] = None
    ) -> Task:
        """Create a new task"""
        task_id = f"task_{len(self.tasks) + 1:04d}"
        
        task = Task(
            id=task_id,
            title=title,
            description=description,
            priority=priority,
            assigned_to=assigned_to,
            due_date=due_date,
            tags=tags,
            dependencies=dependencies
        )
        
        self.tasks[task_id] = task
        self.save_tasks()
        
        logger.info(f"Created task: {task_id} - {title}")
        return task
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID"""
        return self.tasks.get(task_id)
    
    def update_task(
        self,
        task_id: str,
        title: Optional[str] = None,
        description: Optional[str] = None,
        status: Optional[TaskStatus] = None,
        priority: Optional[TaskPriority] = None,
        assigned_to: Optional[str] = None,
        due_date: Optional[datetime] = None,
        tags: Optional[List[str]] = None
    ) -> Optional[Task]:
        """Update task"""
        task = self.tasks.get(task_id)
        if not task:
            logger.warning(f"Task not found: {task_id}")
            return None
        
        if title is not None:
            task.title = title
        if description is not None:
            task.description = description
        if status is not None:
            task.status = status
        if priority is not None:
            task.priority = priority
        if assigned_to is not None:
            task.assigned_to = assigned_to
        if due_date is not None:
            task.due_date = due_date
        if tags is not None:
            task.tags = tags
        
        task.updated_at = datetime.now()
        self.save_tasks()
        
        logger.info(f"Updated task: {task_id}")
        return task
    
    def delete_task(self, task_id: str) -> bool:
        """Delete task"""
        if task_id in self.tasks:
            del self.tasks[task_id]
            self.save_tasks()
            logger.info(f"Deleted task: {task_id}")
            return True
        return False
    
    def list_tasks(
        self,
        status: Optional[TaskStatus] = None,
        priority: Optional[TaskPriority] = None,
        assigned_to: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[Task]:
        """List tasks with optional filters"""
        tasks = list(self.tasks.values())
        
        if status:
            tasks = [t for t in tasks if t.status == status]
        if priority:
            tasks = [t for t in tasks if t.priority == priority]
        if assigned_to:
            tasks = [t for t in tasks if t.assigned_to == assigned_to]
        if tags:
            tasks = [t for t in tasks if any(tag in t.tags for tag in tags)]
        
        return sorted(tasks, key=lambda t: (t.priority.value, t.created_at), reverse=True)
    
    def get_overdue_tasks(self) -> List[Task]:
        """Get overdue tasks"""
        now = datetime.now()
        return [
            task for task in self.tasks.values()
            if task.due_date and task.due_date < now and task.status != TaskStatus.COMPLETED
        ]
    
    def get_upcoming_tasks(self, days: int = 7) -> List[Task]:
        """Get tasks due in the next N days"""
        now = datetime.now()
        future = now + timedelta(days=days)
        
        return [
            task for task in self.tasks.values()
            if task.due_date and now <= task.due_date <= future and task.status != TaskStatus.COMPLETED
        ]
    
    def get_task_statistics(self) -> Dict[str, Any]:
        """Get task statistics"""
        total_tasks = len(self.tasks)
        status_counts = {}
        priority_counts = {}
        
        for task in self.tasks.values():
            status_counts[task.status.value] = status_counts.get(task.status.value, 0) + 1
            priority_counts[task.priority.value] = priority_counts.get(task.priority.value, 0) + 1
        
        return {
            'total_tasks': total_tasks,
            'status_counts': status_counts,
            'priority_counts': priority_counts,
            'overdue_count': len(self.get_overdue_tasks()),
            'upcoming_count': len(self.get_upcoming_tasks())
        }
    
    def load_tasks(self) -> None:
        """Load tasks from storage"""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                self.tasks = {
                    task_id: Task.from_dict(task_data)
                    for task_id, task_data in data.items()
                }
                
                logger.info(f"Loaded {len(self.tasks)} tasks")
                
            except Exception as e:
                logger.error(f"Failed to load tasks: {e}")
                self.tasks = {}
        else:
            logger.info("No existing tasks file found")
    
    def save_tasks(self) -> None:
        """Save tasks to storage"""
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                task_id: task.to_dict()
                for task_id, task in self.tasks.items()
            }
            
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Saved {len(self.tasks)} tasks")
            
        except Exception as e:
            logger.error(f"Failed to save tasks: {e}")
    
    def consolidate_todo_comments(self, project_path: str = ".") -> Dict[str, List[str]]:
        """Scan project for TODO comments and create tasks"""
        todo_pattern = r"#\s*(TODO|FIXME|XXX|HACK|BUG):?\s*(.+)"
        
        todos_found = {}
        
        for py_file in Path(project_path).rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                file_todos = []
                for line_num, line in enumerate(lines, 1):
                    import re
                    match = re.search(todo_pattern, line.strip())
                    if match:
                        todo_type = match.group(1)
                        todo_text = match.group(2)
                        
                        file_todos.append({
                            'line': line_num,
                            'type': todo_type,
                            'text': todo_text,
                            'file': str(py_file.relative_to(project_path))
                        })
                
                if file_todos:
                    todos_found[str(py_file.relative_to(project_path))] = file_todos
                    
                    # Create tasks for high-priority TODOs
                    for todo in file_todos:
                        if todo['type'] in ['TODO', 'FIXME']:
                            priority = TaskPriority.HIGH if todo['type'] == 'FIXME' else TaskPriority.MEDIUM
                            
                            self.create_task(
                                title=f"[{todo['type']}] {todo['text'][:50]}...",
                                description=f"Found in {todo['file']}:{todo['line']}\n\n{todo['text']}",
                                priority=priority,
                                tags=['code-review', todo['type'].lower()],
                                due_date=datetime.now() + timedelta(days=7)
                            )
                
            except Exception as e:
                logger.warning(f"Failed to scan {py_file}: {e}")
        
        logger.info(f"Found TODOs in {len(todos_found)} files")
        return todos_found


# Global instance
task_manager = TaskManager()


# Convenience functions
def create_task(title: str, **kwargs) -> Task:
    """Create a new task"""
    return task_manager.create_task(title, **kwargs)


def get_my_tasks(assigned_to: str = "me") -> List[Task]:
    """Get tasks assigned to user"""
    return task_manager.list_tasks(assigned_to=assigned_to)


def get_pending_tasks() -> List[Task]:
    """Get pending tasks"""
    return task_manager.list_tasks(status=TaskStatus.PENDING)


def consolidate_project_todos(project_path: str = ".") -> Dict[str, List[str]]:
    """Consolidate all TODO comments in project"""
    return task_manager.consolidate_todo_comments(project_path)
