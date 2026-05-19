from datetime import date

from app import (
    ChatState,
    detect_intent,
    normalize_text,
    response_for_intent,
    update_knowledge_base,
)

CASES = [
    ("cách nấu mì cay", "out_of_scope"),
    ("em bị đau khi quan hệ", "sexual_function"),
    ("quan hệ lần đầu nên dùng bao cao su thế nào", "safe_sex"),
    ("trễ kinh 7 ngày có thai không", "contraception_pregnancy"),
    ("hiv có lây qua hôn không", "sti_std"),
]


def test_intents_and_structure():
    for text, expected in CASES:
        state = ChatState()
        norm = normalize_text(text)
        got = detect_intent(norm)
        assert got == expected, f"{text}: expected {expected}, got {got}"
        rsp = response_for_intent(got, norm, state)
        if got not in {"greeting", "ending", "out_of_scope", "missing_info"}:
            for section in ["Nhận định:", "Mức độ cần lưu ý:", "Hướng xử trí/lời khuyên:", "Khi nào cần đi khám:", "Lưu ý an toàn:"]:
                assert section in rsp or "Mình chưa đủ dữ liệu" in rsp or "Mình chưa đủ dữ kiện" in rsp


def test_out_of_scope_question():
    state = ChatState()
    text = normalize_text("hôm nay giá bitcoin bao nhiêu")
    reply = response_for_intent(detect_intent(text), text, state)
    assert "Mình chỉ hỗ trợ" in reply


def test_missing_info_question():
    state = ChatState()
    text = normalize_text("em lo quá có thai không")
    reply = response_for_intent(detect_intent(text), text, state)
    assert "Mình chưa đủ dữ kiện" in reply


def test_ambiguous_question_and_similarity_guard():
    state = ChatState()
    text = normalize_text("em bị sao")
    reply = response_for_intent(detect_intent(text), text, state)
    assert "Mình chưa đủ dữ kiện" in reply


def test_update_knowledge_base_reject_unapproved_source():
    ok, msg = update_knowledge_base(
        {
            "topic": "test topic unapproved",
            "keywords": ["abc"],
            "answer": "xyz",
            "source": "random blog",
            "last_updated": str(date.today()),
        }
    )
    assert not ok
    assert "Nguồn chưa được duyệt" in msg


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main(["-q", __file__]))
