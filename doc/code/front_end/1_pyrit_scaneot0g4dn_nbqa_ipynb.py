# %%NBQA-CELL-SEP52c935
# my_custom_scenarios.py
from pyrit.common.apply_defaults import apply_defaults
from pyrit.scenario import Scenario
from pyrit.scenario.scenario_strategy import ScenarioStrategy


class MyCustomStrategy(ScenarioStrategy):
    """Strategies for my custom scenario."""
    ALL = ("all", {"all"})
    Strategy1 = ("strategy1", set[str]())
    Strategy2 = ("strategy2", set[str]())


@apply_defaults
class MyCustomScenario(Scenario):
    """My custom scenario that does XYZ."""

    @classmethod
    def get_strategy_class(cls):
        return MyCustomStrategy

    @classmethod
    def get_default_strategy(cls):
        return MyCustomStrategy.ALL

    def __init__(self, *, scenario_result_id=None, **kwargs):
        # Scenario-specific configuration only - no runtime parameters
        super().__init__(
            name="My Custom Scenario",
            version=1,
            strategy_class=MyCustomStrategy,
            default_aggregate=MyCustomStrategy.ALL,
            scenario_result_id=scenario_result_id,
        )
        # ... your scenario-specific initialization code

    async def _get_atomic_attacks_async(self):
        # Build and return your atomic attacks
        return []
