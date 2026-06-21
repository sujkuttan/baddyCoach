import pytest
from pathlib import Path
from tempfile import TemporaryDirectory
from tests.hardware import get_ram_gb, get_vram_gb, has_gpu, get_machine_tier


@pytest.fixture
def tmp_job_dir():
    with TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def machine_tier():
    return get_machine_tier()


@pytest.fixture
def skip_if_low_memory():
    if get_ram_gb() < 4:
        pytest.skip("Insufficient RAM (<4GB)")


@pytest.fixture
def skip_if_no_gpu():
    if not has_gpu():
        pytest.skip("No CUDA GPU available")


def pytest_collection_modifyitems(config, items):
    """Auto-skip tests based on hardware."""
    ram = get_ram_gb()
    vram = get_vram_gb()

    skip_low_ram = pytest.mark.skip(reason=f"Low RAM ({ram:.1f}GB)")
    skip_no_gpu = pytest.mark.skip(reason="No CUDA GPU")
    skip_no_model = pytest.mark.skip(reason="Model checkpoint not found")

    for item in items:
        if "memory_intensive" in item.keywords and ram < 4:
            item.add_marker(skip_low_ram)

        if "gpu" in item.keywords and not vram:
            item.add_marker(skip_no_gpu)

        if "model" in item.keywords:
            ckpt_paths = [
                Path("ckpts/TrackNet_best.pt"),
                Path("ckpts/rtmpose/rtmpose-m_8xb64-270e_coco-256x192.onnx"),
                Path("BST/weight/bst_CG_JnB_bone_merged.pt"),
            ]
            if not any(p.exists() for p in ckpt_paths):
                item.add_marker(skip_no_model)
