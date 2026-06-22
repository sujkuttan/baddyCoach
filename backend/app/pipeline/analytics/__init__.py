"""Analytics pipeline stages."""

try:
    from .court_position import CourtPositionAnalyticsStage
    from .fitness import FitnessAnalyticsStage
    from .footwork import FootworkAnalyticsStage
    from .tactical import TacticalAnalyticsStage
    from .technical import TechnicalAnalyticsStage
except (ImportError, Exception):
    CourtPositionAnalyticsStage = None
    FitnessAnalyticsStage = None
    FootworkAnalyticsStage = None
    TacticalAnalyticsStage = None
    TechnicalAnalyticsStage = None

__all__ = [
    "CourtPositionAnalyticsStage",
    "FitnessAnalyticsStage",
    "FootworkAnalyticsStage",
    "TacticalAnalyticsStage",
    "TechnicalAnalyticsStage",
]
