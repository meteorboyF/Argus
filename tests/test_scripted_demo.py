from argus.scripted_demo import seat_reply


def test_scripted_seat_is_explicitly_opt_in_and_limited_to_seat_requests():
    question = 'Is there anyplace I can sit?'
    assert seat_reply(question) is None
    assert seat_reply(question, True) == (
        "Yes, there's a chair straight ahead, "
        "it's empty, and you can sit on it.")
    assert seat_reply('Argus shut down', True) is None
    assert seat_reply('What is in front of me?', True) is None
