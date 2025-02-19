
from pyrit.orchestrator import Orchestrator


def test_get_identifier():
    orchestrator = Orchestrator()
    orchestrator_identifier = orchestrator.get_identifier()

    assert orchestrator_identifier["id"] == str(orchestrator._id)
    assert orchestrator_identifier["__type__"] == orchestrator.__class__.__name__
    assert orchestrator_identifier["__module__"] == orchestrator.__class__.__module__