import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger


class Settings:
    def __init__(self, **kwargs):
        self.root = Path(__file__).resolve().parents[1]
        load_dotenv(self.root / ".env", encoding="utf-8")
        self.log_level = kwargs.get("log_level", os.getenv("MANIP_TRAIN_LOG_LEVEL", "INFO"))
        self.output_dir = self._path("output_dir", "output", kwargs)
        self.data_dir = self._path("data_dir", "data", kwargs)
        self.log_dir = self._path("log_dir", "log", kwargs)
        self.config_dir = self.root / "config"
        # 没有凭证时不连 ClearML，训练照常进行，报告只写本地日志。
        has_credentials = bool(os.getenv("CLEARML_API_ACCESS_KEY")) or (Path.home() / "clearml.conf").exists()
        enabled = kwargs.get("clearml", os.getenv("MANIP_TRAIN_CLEARML", "true"))
        self.clearml = str(enabled).lower() == "true" and has_credentials

    def _path(self, name, default, options):
        value = options.get(name, os.getenv(f"MANIP_TRAIN_{name.upper()}", default))
        path = (self.root / Path(value)).resolve()
        return path


def demo_settings():
    settings = Settings()
    logger.info("Output: {}; data: {}", settings.output_dir, settings.data_dir)
    logger.info("ClearML enabled: {}", settings.clearml)


def main():
    demo_settings()


if __name__ == "__main__":
    main()
