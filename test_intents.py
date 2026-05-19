from app import ChatState, detect_intent, normalize_text, response_for_intent

CASES = [
    ("cách nấu mì cay", "out_of_scope"),
    ("em bị đau khi quan hệ", "sexual_function"),
    ("quan hệ lần đầu nên dùng bao cao su thế nào", "safe_sex"),
    ("trễ kinh 7 ngày có thai không", "contraception_pregnancy"),
    ("hiv có lây qua hôn không", "sti_std"),
    ("em bị xuất tinh sớm", "sexual_function"),
    ("khí hư vàng có mùi hôi", "hygiene"),
    ("em là gay và muốn come out", "lgbtq"),
    ("em lo lắng sau quan hệ", "psychology_consent"),
]


def run():
    for text, expected in CASES:
        state = ChatState()
        norm = normalize_text(text)
        got = detect_intent(norm)
        assert got == expected, f"{text}: expected {expected}, got {got}"
        rsp = response_for_intent(got, norm, state)
        if got not in {"greeting", "ending", "out_of_scope", "missing_info"}:
            for section in ["Nhận định:", "Mức độ cần lưu ý:", "Hướng xử trí/lời khuyên:", "Khi nào cần đi khám:", "Lưu ý an toàn:"]:
                assert section in rsp or "Mình chưa đủ dữ kiện" in rsp, f"Missing section {section} for {text}"
    print("All intent and response-structure tests passed")


if __name__ == "__main__":
    run()
