import os
import sys
import logging

from bentoml import bundler
from bentoml.cli import create_bento_service_cli
from bentoml.utils.log import configure_logging

# By default, ignore warnings when loading BentoService installed as PyPI distribution
# CLI will change back to default log level in config(info), and by adding --quiet or
# --verbose CLI option, user can change the CLI output behavior
configure_logging(logging.ERROR)

__VERSION__ = "20200206150926_DCA9FA"

__module_path = os.path.abspath(os.path.dirname(__file__))

WordEmbeddingModel = bundler.load_bento_service_class(__module_path)

cli=create_bento_service_cli(__module_path)


def load():
    return bundler.load(__module_path)


__all__ = ['__version__', 'WordEmbeddingModel', 'load']
