import logging
import os
import signal
import subprocess
import sys
import time
from threading import Event

import pytz
import schedule
from decouple import config

# Configuração de Logs
LOG_LEVEL = config("LOG_LEVEL", default="INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger("scheduler")

# Configuração de Timezone e Caminhos
TIMEZONE = config("DEFAULT_TIMEZONE", default="America/Sao_Paulo")
try:
    tz = pytz.timezone(TIMEZONE)
except pytz.UnknownTimeZoneError:
    logger.warning(f"Timezone {TIMEZONE} not found, defaulting to UTC")
    tz = pytz.UTC

PROCESSOR_PATH = os.path.join(os.path.dirname(__file__), "football_processor.py")

shutdown_event = Event()


def signal_handler(sig, frame):
    logger.info(f"Received signal {sig}, shutting down gracefully...")
    shutdown_event.set()


def run_command(command_type: str) -> bool:
    """Executa um comando do football processor via subprocesso"""
    try:
        logger.info(f"Starting command: {command_type}")

        # Removido '--force' pois o novo processador não usa mais
        result = subprocess.run(
            [sys.executable, PROCESSOR_PATH, command_type],
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hora de timeout
            env=os.environ.copy()
        )

        if result.returncode == 0:
            logger.info(f"Command '{command_type}' completed successfully")
            # Loga apenas se houver algo relevante no stdout (opcional)
            # if result.stdout:
            #     logger.debug(f"STDOUT: {result.stdout.strip()[:200]}...")
            return True
        else:
            logger.error(f"Command '{command_type}' failed with code {result.returncode}")
            if result.stderr:
                logger.error(f"STDERR: {result.stderr.strip()}")
            return False

    except subprocess.TimeoutExpired:
        logger.error(f"Command '{command_type}' timed out after 1 hour")
        return False
    except Exception as e:
        logger.error(f"Command '{command_type}' failed with exception: {e}")
        return False


def check_system_health() -> bool:
    """Verifica se o script do processador está acessível"""
    return os.path.exists(PROCESSOR_PATH)


# --- Definição dos Jobs ---

def job_full_load():
    logger.info("🔄 Executing FULL LOAD job (Leagues, Standings, Fixtures)")
    run_command("full")


def job_incremental():
    logger.info("⚡ Executing INCREMENTAL job (Live scores, Predictions)")
    run_command("incremental")


def job_leagues():
    logger.info("🏆 Executing LEAGUES job")
    run_command("leagues")


def job_odds():
    logger.info("🎰 Executing ODDS job")
    run_command("odds")


def job_squads():
    logger.info("busts Executing SQUADS job (Players & Photos)")
    run_command("squads")


def job_standings():
    logger.info("📊 Executing STANDINGS job")
    run_command("standings")


def job_health_check():
    logger.info("🩺 Executing health check")
    if check_system_health():
        logger.info("✅ System processor file found")
    else:
        logger.error(f"❌ Processor file not found at {PROCESSOR_PATH}")


# --- Configuração do Cronograma ---

def setup_schedules():
    """Configura a estratégia de agendamento"""
    logger.info("📅 Setting up schedule strategy")

    # 1. Carga Pesada (Madrugada)
    schedule.every().day.at("03:00", tz).do(job_full_load)

    # 2. Atualização de Elencos (Uma vez por dia, cedo)
    schedule.every().day.at("04:00", tz).do(job_squads)

    # 3. Ligas e Tabelas (Manhã)
    schedule.every().day.at("05:00", tz).do(job_leagues)
    schedule.every().day.at("06:00", tz).do(job_standings)

    # 4. Odds (Vários momentos do dia para pegar movimentos de mercado)
    schedule.every().day.at("10:00", tz).do(job_odds)
    schedule.every().day.at("16:00", tz).do(job_odds)
    schedule.every().day.at("20:00", tz).do(job_odds)

    # 5. Incremental (Jogos ao Vivo e Atualizações Rápidas)
    # Horário Nobre (12:00 - 23:59): A cada 5 minutos
    for hour in range(12, 24):
        for minute in range(0, 60, 5):  # 0, 5, 10, ... 55
            schedule.every().day.at(f"{hour:02d}:{minute:02d}", tz).do(job_incremental)

    # Horário de Baixa (00:00 - 11:59): A cada 30 minutos (para jogos noturnos/outros fusos)
    for hour in range(0, 12):
        for minute in [0, 30]:
            schedule.every().day.at(f"{hour:02d}:{minute:02d}", tz).do(job_incremental)

    # 6. Health Check
    schedule.every(1).hours.do(job_health_check)

    logger.info(f"📋 Scheduled {len(schedule.get_jobs())} jobs.")


def main():
    logger.info("🚀 Football Data Processor Scheduler starting...")
    logger.info(f"🌍 Using timezone: {TIMEZONE}")

    # Configura handlers para encerramento gracioso (CTRL+C, Docker stop)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    setup_schedules()

    logger.info("🔍 Initial health check...")
    job_health_check()

    # Loop principal
    try:
        while not shutdown_event.is_set():
            schedule.run_pending()
            # Verifica shutdown a cada segundo para resposta rápida
            if shutdown_event.wait(1):
                break
    except KeyboardInterrupt:
        logger.info("🛑 Keyboard interrupt received")
    except Exception as e:
        logger.critical(f"🔥 Scheduler crashed: {e}")
    finally:
        logger.info("🏁 Scheduler stopped gracefully")


if __name__ == "__main__":
    main()