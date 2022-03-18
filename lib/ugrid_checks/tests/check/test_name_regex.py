from ...check import _VALID_NAME_REGEX as var_re


def test_valid_varnames():
    # Just check a few things that should be accepted, and not
    assert var_re.match("abc")
    assert var_re.match("x")  # single char

    # various unusual chars, cannot appear at start
    nonstart_chars = r"#$£+-*^%?!.:;,\()[]{}"  # almost anything !!
    for nonstart_char in nonstart_chars:
        assert not var_re.match(nonstart_char)
    # But these are all OK *after* the start position
    assert var_re.match("x" + nonstart_chars)

    # Examples of characters which *are* allowed at start
    start_chars = "_10"  # NB includes digits.
    for start_char in start_chars:
        assert var_re.match(start_char)

    # not empty
    assert not var_re.match("")
    # no spaces
    assert not var_re.match("space in name")
    # no forward-slash
    assert not var_re.match("forward/slash")
