from datetime import date

from app import ChatState, detect_intent, normalize_text, response_for_intent, update_knowledge_base


def test_main_intents_with_typos_and_short_forms():
    cases = [
        ("chao bot", "greeting"),
        ("qhe ko bao co thai ko", "contraception_pregnancy"),
        ("bi mun sinh duc", "sti_std"),
        ("em bi roi loan cuong", "sexual_function"),
    ]
    for text, expected in cases:
        assert detect_intent(normalize_text(text)) == expected


def test_missing_info_only_asks_unfilled_slots():
    state = ChatState(memory={"preg_sex_contact": "đã quan hệ thâm nhập"})
    text = normalize_text("em lo có thai")
    reply = response_for_intent("contraception_pregnancy", text, state)
    assert "quan hệ thâm nhập" not in reply
    assert "xuất tinh" in reply and "kỳ kinh" in reply


def test_enough_info_returns_structured_answer():
    state = ChatState(memory={
        "preg_sex_contact": "quan hệ thâm nhập",
        "preg_ejaculation": "có xuất tinh",
        "preg_timing": "2 ngày trước",
        "preg_lmp": "đầu tháng",
    })
    text = normalize_text("trễ kinh có thai không")
    reply = response_for_intent("contraception_pregnancy", text, state)
    for section in [
        "1. Nhận định tình trạng",
        "2. Chẩn đoán phù hợp nhất",
        "3. Mức độ nguy cơ",
        "4. Hướng xử trí cụ thể",
        "5. Khi nào cần đi khám",
        "6. Lưu ý an toàn",
    ]:
        assert section in reply


def test_out_of_scope_and_emergency_and_ending():
    oos = response_for_intent(detect_intent(normalize_text("giá vàng hôm nay")), normalize_text("giá vàng hôm nay"), ChatState())
    assert "Mình chỉ hỗ trợ" in oos

    emergency = response_for_intent(detect_intent(normalize_text("em đau bụng dữ dội và ngất")), normalize_text("em đau bụng dữ dội và ngất"), ChatState())
    assert "Đi khám ngay" in emergency or "cấp cứu" in emergency

    st = ChatState(memory={"preg_lmp": "..."}, asked_slots={"preg_lmp"})
    end = response_for_intent("ending", normalize_text("bye"), st)
    assert "kết thúc" in end
    assert st.memory == {} and st.asked_slots == set()


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
