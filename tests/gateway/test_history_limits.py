from gateway.config import Platform
from gateway.history_limits import truncate_api_server_agent_history


def test_api_server_history_truncation_keeps_recent_turns():
    history = [{"role": "user", "content": f"turn {i}"} for i in range(30)]

    truncated = truncate_api_server_agent_history(history, platform=Platform.API_SERVER)

    assert len(truncated) == 24
    assert truncated[0]["content"] == "turn 6"
    assert truncated[-1]["content"] == "turn 29"


def test_api_server_history_truncation_leaves_other_platforms_alone():
    history = [{"role": "user", "content": f"turn {i}"} for i in range(30)]

    assert truncate_api_server_agent_history(history, platform=Platform.DISCORD) is history
