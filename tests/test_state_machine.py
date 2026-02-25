from src.workflow_state import ReleaseContext, ReleaseStep, WorkflowState


def test_state_step_idle_without_active_release() -> None:
    state = WorkflowState()
    assert state.step == ReleaseStep.IDLE


def test_release_context_step_transition_updates_timestamp() -> None:
    ctx = ReleaseContext(release_version="5.101.0", step=ReleaseStep.WAIT_START_APPROVAL)
    old_updated = ctx.updated_at
    ctx.set_step(ReleaseStep.RELEASE_MANAGER_CREATED)
    assert ctx.step == ReleaseStep.RELEASE_MANAGER_CREATED
    assert ctx.updated_at >= old_updated


def test_processed_event_ids_are_deduplicated() -> None:
    state = WorkflowState()
    state.mark_event_processed("ev-1")
    state.mark_event_processed("ev-1")
    assert state.processed_event_ids == ["ev-1"]
