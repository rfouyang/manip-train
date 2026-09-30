from loguru import logger

from component.training.train_runner import TrainRunner
from config.settings import Settings


class AppContext:
    """顶层对象：持有配置与组件。命令行（以后的任务 API）只通过它调用组件。"""

    def __init__(self, **kwargs):
        self.settings = Settings(**kwargs)
        self.runner = TrainRunner(self.settings)


def main():
    context = AppContext()
    logger.info("Output: {}; ClearML: {}", context.settings.output_dir, context.settings.clearml)


if __name__ == "__main__":
    main()
