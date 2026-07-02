from app.report.generator import ReportGenerator
from app.storage.artifacts import ArtifactStore


def test_report_includes_physics_summary(tmp_path):
    store = ArtifactStore(tmp_path)
    summary = {
        "total": 3,
        "bst": 1,
        "bst_no_physics": 1,
        "physics_fallback": 0,
        "agree": 0,
        "physics_override": 1,
        "bst_gate_distrusted": 0,
        "usable": 2,
        "skipped": 1,
        "distrusted": 0,
        "overrides": 1,
    }
    store.set("physics_summary", summary)

    report = ReportGenerator().generate(tmp_path)

    assert report["physics_summary"] == summary
