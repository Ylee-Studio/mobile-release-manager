from src.workflow_state import ReleaseContext, ReleaseStep, WorkflowState


def test_state_step_idle_without_active_release() -> None:
    state = WorkflowState()
    assert state.step == ReleaseStep.IDLE


def test_release_context_step_transition_updates_timestamp() -> None:
    ctx = ReleaseContext(release_version="5.101.0", step=ReleaseStep.WAIT_START_APPROVAL)
    old_updated = ctx.updated_at
    ctx.set_step(ReleaseStep.WAIT_MANUAL_RELEASE_CONFIRMATION)
    assert ctx.step == ReleaseStep.WAIT_MANUAL_RELEASE_CONFIRMATION
    assert ctx.updated_at >= old_updated


def test_state_roundtrip_keeps_active_release() -> None:
    state = WorkflowState(
        active_release=ReleaseContext(
            release_version="5.101.0",
            step=ReleaseStep.WAIT_START_APPROVAL,
            readiness_map={"Core": False},
        ),
    )
    restored = WorkflowState.from_dict(state.to_dict())
    assert restored.active_release is not None
    assert restored.active_release.release_version == "5.101.0"
