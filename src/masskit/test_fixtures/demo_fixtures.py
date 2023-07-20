import tempfile
from hydra import compose, initialize
from pathlib import Path
from masskit.apps.process.libraries.batch_converter import batch_converter_app

def cho_uniq_short_parquet():
    data_dir = Path("../../../tests/data")
    out = Path(tempfile.gettempdir()) / 'cho_uniq_short.parquet'
    with initialize(version_base=None, config_path="../apps/process/libraries/conf"):
        cfg = compose(config_name="config_batch_converter",
                      overrides=[f"input.file.names={data_dir / 'cho_uniq_short.msp'}",
                                 f"output.file.name={out}",
                                ])
        batch_converter_app(cfg)
        return out
    raise ValueError()
