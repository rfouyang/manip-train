import logging
import sys

from loguru import logger


class InterceptHandler(logging.Handler):
    """把 lerobot / accelerate / clearml 的标准库日志转给 loguru。"""

    def emit(self, record):
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        logger.opt(depth=6, exception=record.exc_info).log(level, record.getMessage())


def setup_logger(log_dir, level="INFO"):
    logger.remove()
    logger.add(sys.stderr, level=level)
    logger.add(log_dir / "manip_train_{time:YYYY-MM-DD}.log", level=level, rotation="00:00", retention="30 days",
               encoding="utf-8")
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)


def main():
    from config.settings import Settings

    settings = Settings()
    setup_logger(settings.log_dir, settings.log_level)
    logging.getLogger("demo").info("standard logging goes to loguru")


if __name__ == "__main__":
    main()
