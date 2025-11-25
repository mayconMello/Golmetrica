import json
import logging
import os
import signal
import subprocess
import sys
from datetime import datetime, timedelta
from threading import Event

import pytz
import schedule
from decouple import config

LOG_LEVEL = config("LOG_LEVEL", default="INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger("scheduler")

TIMEZONE = config("DEFAULT_TIMEZONE", default="America/Sao_Paulo")
tz = pytz.timezone(TIMEZONE)

PROCESSOR_PATH = os.path.join(os.path.dirname(__file__), "football_processor.py")

shutdown_event = Event()


def signal_handler(sig, frame):
    logger.info(f"Received signal {sig}, shutting down gracefully...")
    shutdown_event.set()


def run_command(command_type: str) -> bool:
    """Executa um comando do football processor"""
    try:
        logger.info(f"Starting command: {command_type}")

        result = subprocess.run(
            [sys.executable, PROCESSOR_PATH, command_type, "--force"],
            capture_output=True,
            text=True,
            timeout=3600,
            env=os.environ.copy()  # garante as env vars
        )

        if result.returncode == 0:
            logger.info(f"Command '{command_type}' completed successfully")
            if result.stdout:
                logger.info(f"STDOUT: {result.stdout.strip()[:500]}")
            return True
        else:
            logger.error(f"Command '{command_type}' failed with code {result.returncode}")
            if result.stderr:
                logger.error(f"STDERR: {result.stderr.strip()[:500]}")
            return False

    except subprocess.TimeoutExpired:
        logger.error(f"Command '{command_type}' timed out after 1 hour")
        return False
    except Exception as e:
        logger.error(f"Command '{command_type}' failed with exception: {e}")
        return False



def check_system_health() -> bool:
    """Verifica se o sistema está funcionando"""
    try:
        result = subprocess.run(
            [sys.executable, PROCESSOR_PATH, "--status"],
            capture_output=True,
            text=True,
            timeout=30
        )
        return result.returncode == 0
    except:
        return False


def job_full_load():
    logger.info("🔄 Executing FULL LOAD job")
    success = run_command("full")
    logger.info("✅ FULL LOAD" if success else "❌ FULL LOAD failed")


def job_incremental():
    logger.info("⚡ Executing INCREMENTAL job")
    success = run_command("incremental")
    logger.info("✅ INCREMENTAL" if success else "❌ INCREMENTAL failed")


def job_leagues():
    logger.info("🏆 Executing LEAGUES job")
    success = run_command("leagues")
    logger.info("✅ LEAGUES" if success else "❌ LEAGUES failed")


def job_odds():
    logger.info("🎰 Executing ODDS job")
    success = run_command("odds")
    logger.info("✅ ODDS" if success else "❌ ODDS failed")


def job_auto():
    logger.info("🤖 Executing AUTO job")
    success = run_command("auto")
    logger.info("✅ AUTO" if success else "❌ AUTO failed")


def job_health_check():
    logger.info("🩺 Executing health check")
    if check_system_health():
        logger.info("✅ System health OK")
    else:
        logger.warning("⚠️ System health check failed")


def job_schedule_fixtures():
    """Agenda jobs incrementais dinâmicos próximos dos jogos"""
    logger.info("📅 Checking upcoming fixtures for dynamic scheduling...")

    try:

        result = subprocess.run(
            [sys.executable, PROCESSOR_PATH, "incremental", "--status"],
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode != 0 or not result.stdout:
            logger.warning("⚠️ Could not fetch fixtures for scheduling")
            return

        status = json.loads(result.stdout)

        fixtures = status.get("upcoming_fixtures", [])

        if not fixtures:
            logger.info("Nenhum fixture encontrado para agendamento")
            return

        kickoff_times = {}
        for f in fixtures:
            start_str = f.get("start_utc")
            if not start_str:
                continue
            try:
                dt = datetime.fromisoformat(start_str.replace("Z", "+00:00")).astimezone(tz)
                kickoff_times.setdefault(dt, []).append(f["id"])
            except Exception:
                continue

        for kickoff_dt, games in kickoff_times.items():

            for offset in [0, 5]:
                sched_time = (kickoff_dt + timedelta(minutes=offset)).strftime("%H:%M")
                schedule.every().day.at(sched_time, tz).do(job_incremental)
                logger.info(f"⚡ Scheduled incremental at {sched_time} for {len(games)} games")

    except Exception as e:
        logger.error(f"Failed to dynamically schedule fixtures: {e}")


def setup_schedules():
    """Configura agendamentos"""
    logger.info("📅 Setting up schedule strategy (HYBRID)")

    schedule.every().day.at("03:00", tz).do(job_full_load)

    for hour in range(12, 24):
        for minute in [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]:
            schedule.every().day.at(f"{hour:02d}:{minute:02d}", tz).do(job_incremental)

    for hour in list(range(0, 12)) + [23]:
        for minute in [0, 30]:
            schedule.every().day.at(f"{hour:02d}:{minute:02d}", tz).do(job_incremental)

    schedule.every().day.at("10:00", tz).do(job_odds)
    schedule.every().day.at("16:00", tz).do(job_odds)
    schedule.every().day.at("20:00", tz).do(job_odds)

    schedule.every().day.at("05:00", tz).do(job_leagues)

    schedule.every(1).hours.do(job_health_check)

    schedule.every().hour.at(":00").do(job_schedule_fixtures)

    logger.info("📋 Scheduled jobs:")
    for job in schedule.get_jobs():
        logger.info(f"  - {job}")


def main():
    logger.info("🚀 Football Data Processor Scheduler starting...")
    logger.info(f"🌍 Using timezone: {TIMEZONE}")

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    setup_schedules()

    logger.info("🔍 Initial health check...")
    job_health_check()

    try:
        while not shutdown_event.is_set():
            schedule.run_pending()
            if shutdown_event.wait(60):
                break
    except KeyboardInterrupt:
        logger.info("🛑 Keyboard interrupt received")

    logger.info("🏁 Scheduler stopped gracefully")


if __name__ == "__main__":
    main()
