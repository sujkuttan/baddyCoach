import pytest
from pathlib import Path
from tempfile import TemporaryDirectory


@pytest.fixture
def tmp_job_dir():
    with TemporaryDirectory() as d:
        yield Path(d)
