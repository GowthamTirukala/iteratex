"""Background scheduler that periodically triggers retraining job.

Runs inside a long-lived container. Uses APScheduler to execute `scripts/retrain.py` every N minutes.
"""

import os
import subprocess
from pathlib import Path

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "training" / "training_data.parquet"


def retrain_job():
    cmd = [
        "python",
        str(PROJECT_ROOT / "scripts" / "retrain.py"),
        "--trainer",
        os.getenv("TRAINER", "dummy"),
        "--data",
        str(DATA_PATH),
    ]
    logger.info("Running retrain command: %s", " ".join(cmd))
    subprocess.run(cmd, check=False)


def main():
    sched = BlockingScheduler(timezone="UTC")
    # Cron expression via env, default every hour
    cron_expr = os.getenv("RETRAIN_CRON", "0 * * * *")
    trigger = CronTrigger.from_crontab(cron_expr)
    sched.add_job(retrain_job, trigger, name="model_retrain")
    logger.info("Scheduler started with cron '%s'", cron_expr)
    try:
        sched.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped")


if __name__ == "__main__":
    main()
