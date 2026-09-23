from quire.prefix import common_prefix_len, share_question_prefix


def test_common_prefix_leaves_one_token_per_sequence():
    assert common_prefix_len([[1, 2, 3], [1, 2, 3]]) == 2
    assert common_prefix_len([[1, 2, 3, 4], [1, 2, 9, 4]]) == 2
    assert common_prefix_len([[5], [5]]) == 0


def test_share_moves_only_a_lone_questions_common_tokens():
    plans = [("q", ["A", "B"], [(["t", "f"], [7, 8, 1, 2]), (["f", "t"], [7, 8, 2, 1])])]
    prefix, out = share_question_prefix([0, 0], plans)
    assert prefix == [0, 0, 7, 8]
    assert out[0][2] == [(["t", "f"], [1, 2]), (["f", "t"], [2, 1])]
    # every sequence the model reads is unchanged
    for (o, before), (_, after) in zip(plans[0][2], out[0][2]):
        assert [0, 0] + before == prefix + after


def test_share_leaves_multi_question_and_single_ordering_requests_alone():
    two = [("q1", [], [([], [1, 2]), ([], [1, 3])]), ("q2", [], [([], [1, 2]), ([], [1, 3])])]
    assert share_question_prefix([0], two) == ([0], two)
    one = [("q", [], [([], [1, 2])])]
    assert share_question_prefix([0], one) == ([0], one)
