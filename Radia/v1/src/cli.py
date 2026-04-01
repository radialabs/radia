"""
Command-line interface for WiFi-DensePose API
"""

import asyncio
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Optional

import click
import psutil
import uvicorn

from src.config.settings import get_settings, load_settings_from_file
from src.logger import setup_logging, get_logger

settings = get_settings()
setup_logging(settings)
logger = get_logger(__name__)


def get_settings_with_config(config_file: Optional[str] = None):
    if config_file:
        return load_settings_from_file(config_file)
    return get_settings()


def _pid_file(s) -> Path:
    return Path(s.log_directory) / "wifi-densepose-api.pid"


def _get_server_status(s) -> dict:
    pf = _pid_file(s)
    status = {"running": False, "pid": None, "pid_file": str(pf), "pid_file_exists": pf.exists()}
    if pf.exists():
        try:
            pid = int(pf.read_text().strip())
            status["pid"] = pid
            os.kill(pid, 0)
            status["running"] = True
        except (OSError, ValueError):
            status["running"] = False
    return status


# ── CLI root ────────────────────────────────────────────────

@click.group()
@click.option("--config", "-c", type=click.Path(exists=True), help="Path to configuration file")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option("--debug", is_flag=True, help="Enable debug mode")
@click.pass_context
def cli(ctx, config: Optional[str], verbose: bool, debug: bool):
    """WiFi-DensePose API Command Line Interface."""
    ctx.ensure_object(dict)
    ctx.obj["config_file"] = config
    ctx.obj["debug"] = debug
    if debug:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    elif verbose:
        import logging
        logging.getLogger().setLevel(logging.INFO)


# ── start / stop / status ───────────────────────────────────

@cli.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=8000, type=int, help="Port to bind to")
@click.option("--workers", default=1, type=int, help="Number of worker processes")
@click.option("--reload", is_flag=True, help="Enable auto-reload for development")
@click.option(
    "--daemon",
    "-d",
    is_flag=True,
    help="Ignored legacy flag; use a process manager to run in the background.",
)
@click.pass_context
def start(ctx, host, port, workers, reload, daemon):
    """Start the WiFi-DensePose API server."""
    s = get_settings_with_config(ctx.obj.get("config_file"))
    if ctx.obj.get("debug"):
        s.debug = True
    if daemon:
        logger.warning("--daemon is ignored; the server runs in the foreground.")

    async def _start():
        logger.info("Starting WiFi-DensePose API server (env=%s)", s.environment)

        for _, d in [("Log", s.log_directory), ("Backup", s.backup_directory)]:
            Path(d).mkdir(parents=True, exist_ok=True)

        pf = _pid_file(s)
        track_pid = not reload
        if track_pid:
            pf.parent.mkdir(parents=True, exist_ok=True)
            if pf.exists():
                try:
                    os.kill(int(pf.read_text().strip()), 0)
                    logger.error("Another process may already be running (PID file present).")
                    sys.exit(1)
                except (OSError, ValueError):
                    pf.unlink(missing_ok=True)
            try:
                pf.write_text(str(os.getpid()))
            except OSError as e:
                logger.warning("Could not write PID file %s: %s", pf, e)

        cfg = uvicorn.Config(
            "src.api.main:app", host=host, port=port,
            workers=workers if not reload else 1,
            reload=reload,
            log_level="debug" if s.debug else "info",
        )
        try:
            await uvicorn.Server(cfg).serve()
        finally:
            if track_pid:
                try:
                    if pf.exists() and pf.read_text().strip() == str(os.getpid()):
                        pf.unlink()
                except OSError:
                    pass

    try:
        asyncio.run(_start())
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error("Failed to start server: %s", e); sys.exit(1)


@cli.command()
@click.option("--force", "-f", is_flag=True, help="Force stop without graceful shutdown")
@click.option("--timeout", default=30, type=int, help="Graceful shutdown timeout")
@click.pass_context
def stop(ctx, force, timeout):
    """Stop the WiFi-DensePose API server."""
    s = get_settings_with_config(ctx.obj.get("config_file"))
    st = _get_server_status(s)
    pf = _pid_file(s)

    if not st["running"]:
        if st["pid_file_exists"]:
            pf.unlink(missing_ok=True); click.echo("Cleaned up stale PID file.")
        else:
            click.echo("Server is not running.")
        return

    pid = st["pid"]
    sig = signal.SIGKILL if force else signal.SIGTERM
    os.kill(pid, sig)
    click.echo(f"Sent {'SIGKILL' if force else 'SIGTERM'} to PID {pid}")

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            os.kill(pid, 0); time.sleep(1)
        except OSError:
            click.echo("Server stopped."); pf.unlink(missing_ok=True); return

    if not force:
        os.kill(pid, signal.SIGKILL); click.echo("Timeout — sent SIGKILL.")
        time.sleep(2)
    pf.unlink(missing_ok=True)


@cli.command()
@click.option("--format", "fmt", type=click.Choice(["text", "json"]), default="text")
@click.option("--detailed", is_flag=True, help="Show detailed status")
@click.pass_context
def status(ctx, fmt, detailed):
    """Show the status of the WiFi-DensePose API server."""
    s = get_settings_with_config(ctx.obj.get("config_file"))
    st = _get_server_status(s)

    data = {"server": st, "system": {
        "hostname": psutil.os.uname().nodename,
        "platform": psutil.os.uname().sysname,
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    }, "config": {"environment": s.environment, "debug": s.debug, "version": s.version}}

    if st["running"] and st["pid"]:
        try:
            p = psutil.Process(st["pid"])
            data["server"]["uptime_s"] = time.time() - p.create_time()
            data["server"]["memory_mb"] = p.memory_info().rss / (1024 * 1024)
            data["server"]["threads"] = p.num_threads()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    if detailed:
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        data["resources"] = {
            "cpu_percent": psutil.cpu_percent(interval=0.5),
            "memory_percent": mem.percent,
            "disk_percent": round(disk.used / disk.total * 100, 1),
        }

    if fmt == "json":
        click.echo(json.dumps(data, indent=2, default=str))
    else:
        running = data["server"]["running"]
        click.echo(f"Server: {'RUNNING (PID ' + str(st['pid']) + ')' if running else 'NOT RUNNING'}")
        click.echo(f"Env: {s.environment}  Debug: {s.debug}  Version: {s.version}")
        if running and "uptime_s" in data["server"]:
            click.echo(f"Uptime: {int(data['server']['uptime_s'])}s  "
                        f"Memory: {data['server']['memory_mb']:.1f}MB")
        if detailed and "resources" in data:
            r = data["resources"]
            click.echo(f"CPU: {r['cpu_percent']}%  Memory: {r['memory_percent']}%  Disk: {r['disk_percent']}%")


# ── db (stubs — database layer removed) ────────────────────

@cli.group()
def db():
    """Database management commands (not configured)."""
    pass

@db.command()
@click.pass_context
def init(ctx, **_):
    """Initialize the database schema."""
    click.echo("Database layer removed. Re-integrate a persistence backend to use this command.")

@db.command()
@click.pass_context
def migrate(ctx, **_):
    """Run database migrations."""
    click.echo("Database layer removed.")

@db.command()
@click.pass_context
def rollback(ctx, **_):
    """Rollback database migrations."""
    click.echo("Database layer removed.")


# ── tasks ───────────────────────────────────────────────────

@cli.group()
def tasks():
    """Background task management."""
    pass

@tasks.command("run")
@click.option("--task", type=click.Choice(["cleanup", "monitoring", "backup"]))
@click.pass_context
def tasks_run(ctx, task):
    """Run background tasks."""
    from src.tasks.cleanup import get_cleanup_manager
    from src.tasks.monitoring import get_monitoring_manager
    from src.tasks.backup import get_backup_manager
    s = get_settings_with_config(ctx.obj.get("config_file"))
    async def _run():
        mgrs = {"cleanup": get_cleanup_manager, "monitoring": get_monitoring_manager, "backup": get_backup_manager}
        for name, factory in mgrs.items():
            if task is None or task == name:
                click.echo(json.dumps(await factory(s).run_all_tasks(), indent=2, default=str))
    asyncio.run(_run())

@tasks.command("status")
@click.pass_context
def tasks_status(ctx):
    """Show background task status."""
    from src.tasks.cleanup import get_cleanup_manager
    from src.tasks.monitoring import get_monitoring_manager
    from src.tasks.backup import get_backup_manager
    s = get_settings_with_config(ctx.obj.get("config_file"))
    click.echo(json.dumps({
        "cleanup": get_cleanup_manager(s).get_stats(),
        "monitoring": get_monitoring_manager(s).get_stats(),
        "backup": get_backup_manager(s).get_stats(),
    }, indent=2, default=str))


# ── config ──────────────────────────────────────────────────

@cli.group()
def config():
    """Configuration management."""
    pass

@config.command()
@click.pass_context
def show(ctx):
    """Show current configuration."""
    s = get_settings_with_config(ctx.obj.get("config_file"))
    click.echo(json.dumps({
        "app_name": s.app_name, "version": s.version, "environment": s.environment,
        "debug": s.debug, "host": s.host, "port": s.port,
        "wifi_interface": s.wifi_interface, "csi_buffer_size": s.csi_buffer_size,
        "pose_confidence_threshold": s.pose_confidence_threshold,
    }, indent=2))

@config.command()
@click.pass_context
def validate(ctx):
    """Validate configuration."""
    s = get_settings_with_config(ctx.obj.get("config_file"))
    click.echo("- Database: NOT CONFIGURED (removed)")
    for name, d in [("Data storage", s.data_storage_path), ("Model storage", s.model_storage_path)]:
        p = Path(d)
        p.mkdir(parents=True, exist_ok=True)
        click.echo(f"+ {name}: OK ({d})")
    click.echo("Configuration valid.")


# ── version ─────────────────────────────────────────────────

@cli.command()
def version():
    """Show version information."""
    s = get_settings()
    click.echo(f"WiFi-DensePose API v{s.version}")
    click.echo(f"Environment: {s.environment}")
    click.echo(f"Python: {sys.version}")


def create_cli(orchestrator=None):
    return cli


if __name__ == "__main__":
    cli()
