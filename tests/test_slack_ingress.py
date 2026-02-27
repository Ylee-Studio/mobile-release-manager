from src.slack_ingress import _extract_bot_user_id, _should_ignore_message_event


def test_extract_bot_user_id_from_authorizations() -> None:
    raw_event = {
        "authorizations": [
            {"enterprise_id": None, "team_id": "T1", "user_id": "U_BOT", "is_bot": True}
        ]
    }

    assert _extract_bot_user_id(raw_event) == "U_BOT"


def test_ignore_message_from_same_bot_user() -> None:
    raw_event = {
        "authorizations": [
            {"enterprise_id": None, "team_id": "T1", "user_id": "U_BOT", "is_bot": True}
        ]
    }
    message_event = {"type": "message", "user": "U_BOT", "text": "self echo"}

    assert _should_ignore_message_event(raw_event, message_event) is True


def test_allow_message_from_human_user() -> None:
    raw_event = {
        "authorizations": [
            {"enterprise_id": None, "team_id": "T1", "user_id": "U_BOT", "is_bot": True}
        ]
    }
    message_event = {"type": "message", "user": "U_HUMAN", "text": "please <@U_BOT> release"}

    assert _should_ignore_message_event(raw_event, message_event) is False


def test_ignore_human_message_without_bot_mention() -> None:
    raw_event = {
        "authorizations": [
            {"enterprise_id": None, "team_id": "T1", "user_id": "U_BOT", "is_bot": True}
        ]
    }
    message_event = {"type": "message", "user": "U_HUMAN", "text": "release please"}

    assert _should_ignore_message_event(raw_event, message_event) is True


def test_allow_human_message_with_mention_alias_form() -> None:
    raw_event = {
        "authorizations": [
            {"enterprise_id": None, "team_id": "T1", "user_id": "U_BOT", "is_bot": True}
        ]
    }
    message_event = {"type": "message", "user": "U_HUMAN", "text": "please <@U_BOT|bot> check"}

    assert _should_ignore_message_event(raw_event, message_event) is False
