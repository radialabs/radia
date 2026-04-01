"""
WebSocket connection manager for WiFi-DensePose API
"""

import asyncio
import json
import logging
import uuid
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
from collections import defaultdict

from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


class WebSocketConnection:
    """Represents a WebSocket connection with metadata."""

    def __init__(self, websocket: WebSocket, client_id: str, stream_type: str,
                 zone_ids: Optional[List[str]] = None, **config):
        self.websocket = websocket
        self.client_id = client_id
        self.stream_type = stream_type
        self.zone_ids = zone_ids or []
        self.config = config
        self.connected_at = datetime.utcnow()
        self.last_ping = datetime.utcnow()
        self.message_count = 0
        self.is_active = True

    async def send_json(self, data: Dict[str, Any]):
        try:
            await self.websocket.send_json(data)
            self.message_count += 1
        except Exception as e:
            logger.error(f"Error sending to client {self.client_id}: {e}")
            self.is_active = False
            raise

    async def send_text(self, message: str):
        try:
            await self.websocket.send_text(message)
            self.message_count += 1
        except Exception as e:
            logger.error(f"Error sending text to client {self.client_id}: {e}")
            self.is_active = False
            raise

    def update_config(self, config: Dict[str, Any]):
        self.config.update(config)
        if "zone_ids" in config:
            self.zone_ids = config["zone_ids"] or []

    def matches_filter(self, stream_type: Optional[str] = None,
                       zone_ids: Optional[List[str]] = None, **filters) -> bool:
        if stream_type and self.stream_type != stream_type:
            return False
        if zone_ids:
            if self.zone_ids and not any(z in self.zone_ids for z in zone_ids):
                return False
        for key, value in filters.items():
            if key in self.config and self.config[key] != value:
                return False
        return True

    def get_info(self) -> Dict[str, Any]:
        return {
            "client_id": self.client_id,
            "stream_type": self.stream_type,
            "zone_ids": self.zone_ids,
            "connected_at": self.connected_at.isoformat(),
            "last_ping": self.last_ping.isoformat(),
            "message_count": self.message_count,
            "is_active": self.is_active,
            "uptime_seconds": (datetime.utcnow() - self.connected_at).total_seconds(),
        }


class ConnectionManager:
    """Manages WebSocket connections for real-time streaming.

    Uses a single connections dict. Type/zone views are computed on demand
    since the number of concurrent WebSocket connections is small.
    """

    def __init__(self):
        self.connections: Dict[str, WebSocketConnection] = {}
        self._total_connections = 0
        self._messages_sent = 0
        self._errors = 0
        self._start_time = datetime.utcnow()
        self._cleanup_task = None
        self._started = False

    @property
    def stats(self) -> Dict[str, Any]:
        """Single source of truth for all connection statistics."""
        by_type: Dict[str, int] = defaultdict(int)
        by_zone: Dict[str, int] = defaultdict(int)
        active = 0
        for c in self.connections.values():
            by_type[c.stream_type] += 1
            for z in c.zone_ids:
                by_zone[z] += 1
            if c.is_active:
                active += 1
        uptime = (datetime.utcnow() - self._start_time).total_seconds()
        return {
            "total_connections": self._total_connections,
            "active_connections": active,
            "inactive_connections": len(self.connections) - active,
            "clients_by_type": dict(by_type),
            "clients_by_zone": dict(by_zone),
            "messages_sent": self._messages_sent,
            "errors": self._errors,
            "uptime_seconds": uptime,
            "messages_per_second": self._messages_sent / max(uptime, 1),
            "error_rate": self._errors / max(self._messages_sent, 1),
            "start_time": self._start_time,
            "total_clients": len(self.connections),
            "active_clients": active,
            "inactive_clients": len(self.connections) - active,
        }

    # Compatibility shims for callers that use the old method names
    async def get_connection_stats(self) -> Dict[str, Any]:
        return self.stats

    async def get_metrics(self) -> Dict[str, Any]:
        return self.stats

    async def get_connected_clients(self) -> List[Dict[str, Any]]:
        return [c.get_info() for c in self.connections.values()]

    async def get_client_status(self, client_id: str) -> Optional[Dict[str, Any]]:
        c = self.connections.get(client_id)
        return c.get_info() if c else None

    async def connect(self, websocket: WebSocket, stream_type: str,
                      zone_ids: Optional[List[str]] = None, **config) -> str:
        client_id = str(uuid.uuid4())
        connection = WebSocketConnection(
            websocket=websocket, client_id=client_id,
            stream_type=stream_type, zone_ids=zone_ids, **config,
        )
        self.connections[client_id] = connection
        self._total_connections += 1
        logger.info(f"WebSocket client {client_id} connected for {stream_type}")
        return client_id

    async def disconnect(self, client_id: str) -> bool:
        connection = self.connections.pop(client_id, None)
        if connection is None:
            return False
        if connection.is_active:
            try:
                await connection.websocket.close()
            except Exception:
                pass
        logger.info(f"WebSocket client {client_id} disconnected")
        return True

    async def disconnect_all(self):
        for cid in list(self.connections):
            await self.disconnect(cid)

    async def send_to_client(self, client_id: str, data: Dict[str, Any]) -> bool:
        connection = self.connections.get(client_id)
        if not connection:
            return False
        try:
            await connection.send_json(data)
            self._messages_sent += 1
            return True
        except Exception as e:
            logger.error(f"Error sending to client {client_id}: {e}")
            self._errors += 1
            connection.is_active = False
            return False

    async def broadcast(self, data: Dict[str, Any], stream_type: Optional[str] = None,
                        zone_ids: Optional[List[str]] = None, **filters) -> int:
        matching = [
            c for c in self.connections.values()
            if c.is_active and c.matches_filter(stream_type=stream_type, zone_ids=zone_ids, **filters)
        ]
        sent = 0
        failed = []
        for c in matching:
            if await self.send_to_client(c.client_id, data):
                sent += 1
            else:
                failed.append(c.client_id)
        for cid in failed:
            await self.disconnect(cid)
        return sent

    async def update_client_config(self, client_id: str, config: Dict[str, Any]) -> bool:
        connection = self.connections.get(client_id)
        if not connection:
            return False
        connection.update_config(config)
        return True

    async def ping_clients(self):
        ping_data = {"type": "ping", "timestamp": datetime.utcnow().isoformat()}
        failed = []
        for cid, c in self.connections.items():
            try:
                await c.send_json(ping_data)
                c.last_ping = datetime.utcnow()
            except Exception:
                failed.append(cid)
        for cid in failed:
            await self.disconnect(cid)

    async def cleanup_inactive_connections(self):
        now = datetime.utcnow()
        stale = [
            cid for cid, c in self.connections.items()
            if not c.is_active or (now - c.last_ping > timedelta(minutes=5))
        ]
        for cid in stale:
            await self.disconnect(cid)
        if stale:
            logger.info(f"Cleaned up {len(stale)} stale connections")

    async def start(self):
        if not self._started:
            self._start_cleanup_task()
            self._started = True
            logger.info("Connection manager started")

    def _start_cleanup_task(self):
        async def cleanup_loop():
            while True:
                try:
                    await asyncio.sleep(60)
                    await self.cleanup_inactive_connections()
                    if datetime.utcnow().minute % 2 == 0:
                        await self.ping_clients()
                except Exception as e:
                    logger.error(f"Error in cleanup task: {e}")
        try:
            self._cleanup_task = asyncio.create_task(cleanup_loop())
        except RuntimeError:
            logger.debug("No event loop running, cleanup task will start later")

    async def shutdown(self):
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        await self.disconnect_all()
        logger.info("Connection manager shutdown complete")


connection_manager = ConnectionManager()
