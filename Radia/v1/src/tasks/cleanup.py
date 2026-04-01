"""
Periodic cleanup tasks for WiFi-DensePose API.

Database layer removed. Tasks are no-ops until a persistence backend is
re-integrated.
"""

from typing import Dict, Any, Optional

from src.config.settings import Settings
from src.logger import get_logger

logger = get_logger(__name__)


class CleanupManager:
    """Stub cleanup manager — database not configured."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.running = False

    async def run_all_tasks(self) -> Dict[str, Any]:
        logger.info("Database not configured — cleanup tasks skipped")
        return {"status": "skipped", "message": "Database not configured"}

    async def run_task(self, task_name: str) -> Dict[str, Any]:
        logger.info("Database not configured — cleanup task %s skipped", task_name)
        return {"status": "skipped", "task": task_name}

    def get_stats(self) -> Dict[str, Any]:
        return {"manager": {"running": False, "total_cleaned": 0}, "tasks": []}

    def enable_task(self, task_name: str) -> bool:
        return False

    def disable_task(self, task_name: str) -> bool:
        return False


_cleanup_manager: Optional[CleanupManager] = None


def get_cleanup_manager(settings: Optional[Settings] = None) -> CleanupManager:
    global _cleanup_manager
    if _cleanup_manager is None:
        _cleanup_manager = CleanupManager(settings)
    return _cleanup_manager


async def run_periodic_cleanup(settings: Settings):
    logger.info("Database not configured — periodic cleanup disabled")
