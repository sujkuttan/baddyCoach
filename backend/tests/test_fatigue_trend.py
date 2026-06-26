import numpy as np
import pytest
from app.pipeline.analytics.fitness import FitnessAnalyticsStage


def test_fatigue_trend_declining():
    """High intensity early, low intensity late -> declining."""
    intensities = [3.0, 3.2, 2.8, 2.5, 2.0, 1.8, 1.5, 1.2, 1.0, 0.8]
    result = FitnessAnalyticsStage._compute_fatigue_trend(intensities)
    assert result == "declining"


def test_fatigue_trend_stable():
    """Consistent intensity throughout -> stable."""
    intensities = [2.0, 2.1, 1.9, 2.0, 2.1, 1.9, 2.0, 2.1, 1.9, 2.0]
    result = FitnessAnalyticsStage._compute_fatigue_trend(intensities)
    assert result == "stable"


def test_fatigue_trend_improving():
    """Low intensity early, high intensity late -> improving."""
    intensities = [1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 2.8, 3.0, 3.2, 3.5]
    result = FitnessAnalyticsStage._compute_fatigue_trend(intensities)
    assert result == "improving"


def test_fatigue_trend_insufficient_data():
    """Less than 5 rallies -> insufficient_data."""
    intensities = [2.0, 2.1, 1.9]
    result = FitnessAnalyticsStage._compute_fatigue_trend(intensities)
    assert result == "insufficient_data"


def test_fatigue_trend_empty():
    """Empty list -> insufficient_data."""
    result = FitnessAnalyticsStage._compute_fatigue_trend([])
    assert result == "insufficient_data"
