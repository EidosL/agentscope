from app.services.relation_factory import has_pred_kw


def test_has_pred_kw_supports_hits_keyword():
    quote = "这段材料直接支持该观点，提供了明确的支持证据。"
    assert has_pred_kw("supports", [quote]) is True


def test_has_pred_kw_supports_negative():
    quote = "这里描述的只是背景信息，没有提及相互印证。"
    assert has_pred_kw("supports", [quote]) is False


def test_has_pred_kw_unknown_predicate_falls_back():
    quote = "两者之间存在紧密联系，形成清晰的关联。"
    assert has_pred_kw("custom_relation", [quote]) is True


def test_has_pred_kw_handles_empty_snippets():
    assert has_pred_kw("supports", [""]) is False
    assert has_pred_kw("", ["任何内容"]) is True
