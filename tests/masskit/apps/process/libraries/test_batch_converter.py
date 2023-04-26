import pytest
import os
from masskit.apps.process.libraries.batch_converter import batch_converter_app

def test_catch_converter(config_batch_converter):
    batch_converter_app(config_batch_converter)
