"""
Backup tasks for WiFi-DensePose API.

Database layer removed. Tasks are no-ops until a persistence backend is
re-integrated.
"""

from typing import Dict, Any, Optional, List

from src.config.settings import Settings
from src.logger import get_logger

logger = get_logger(__name__)


class BackupManager:
    """Stub backup manager — database not configured."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.running = False

    async def run_all_tasks(self) -> Dict[str, Any]:
        logger.info("Database not configured — backup tasks skipped")
        return {"status": "skipped", "message": "Database not configured"}

    async def run_task(self, task_name: str) -> Dict[str, Any]:
        logger.info("Database not configured — backup task %s skipped", task_name)
        return {"status": "skipped", "task": task_name}

    def get_stats(self) -> Dict[str, Any]:
        return {"manager": {"running": False, "total_backup_size_mb": 0}, "tasks": []}

    def list_backups(self) -> Dict[str, List[Dict[str, Any]]]:
        return {}


_backup_manager: Optional[BackupManager] = None


def get_backup_manager(settings: Optional[Settings] = None) -> BackupManager:
    global _backup_manager
    if _backup_manager is None:
        _backup_manager = BackupManager(settings)
    return _backup_manager


async def run_periodic_backup(settings: Settings):
    logger.info("Database not configured — periodic backup disabled")
