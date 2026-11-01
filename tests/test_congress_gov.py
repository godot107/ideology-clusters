from ideology import congress_gov


def test_parse_bill():
    assert congress_gov.parse_bill("HR7148") == ("hr", 7148)
    assert congress_gov.parse_bill("HJRES42") == ("hjres", 42)
    assert congress_gov.parse_bill("hres1438") == ("hres", 1438)
    assert congress_gov.parse_bill("S1582") == ("s", 1582)
    assert congress_gov.parse_bill("SCONRES3") == ("sconres", 3)
    assert congress_gov.parse_bill("PN123") is None
    assert congress_gov.parse_bill(None) is None
