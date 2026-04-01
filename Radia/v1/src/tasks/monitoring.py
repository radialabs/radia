"""
Monitoring tasks for WiFi-DensePose API.

Database layer removed. Tasks are no-ops until a persistence backend is
re-integrated.
"""

from typing import Dict, Any, Optional

from src.config.settings import Settings
from src.logger import get_logger

logger = get_logger(__name__)


class MonitoringManager:
    """Stub monitoring manager — database not configured."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.running = False

    async def run_all_tasks(self) -> Dict[str, Any]:
        logger.info("Database not configured — monitoring tasks skipped")
        return {"status": "skipped", "message": "Database not configured"}

    async def run_task(self, task_name: str) -> Dict[str, Any]:
        logger.info("Database not configured — monitoring task %s skipped", task_name)
        return {"status": "skipped", "task": task_name}

    def get_stats(self) -> Dict[str, Any]:
        return {"manager": {"running": False}, "tasks": []}

    def get_performance_task(self):
        return None


_monitoring_manager: Optional[MonitoringManager] = None


def get_monitoring_manager(settings: Optional[Settings] = None) -> MonitoringManager:
    global _monitoring_manager
    if _monitoring_manager is None:
        _monitoring_manager = MonitoringManager(settings)
    return _monitoring_manager


async def run_periodic_monitoring(settings: Settings):
    logger.info("Database not configured — periodic monitoring disabled")
