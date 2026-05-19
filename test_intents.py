from app import detect_intent, normalize_text, response_for_intent

CASES = [
    ("cách nấu mì cay", "out_of_scope"),
    ("em bị đau khi quan hệ", "missing_info"),
    ("quan hệ lần đầu nên dùng bao cao su thế nào", "safe_sex"),
    ("trễ kinh 7 ngày có thai không", "pregnancy_contraception"),
    ("hiv có lây qua hôn không", "sti_std"),
    ("em bị xuất tinh sớm", "missing_info"),
    ("khí hư vàng có mùi hôi", "female_health"),
    ("em là gay và muốn come out", "lgbtq"),
    ("em lo lắng sau quan hệ", "psychology_emotion"),
    ("quan hệ xong bị chảy máu nhiều và chóng mặt", "emergency"),
]


def run():
    for text, expected in CASES:
        got = detect_intent(normalize_text(text))
        assert got == expected, f"{text}: expected {expected}, got {got}"
        rsp = response_for_intent(got, normalize_text(text))
        if got not in {"greeting", "ending", "out_of_scope", "missing_info"}:
            for section in ["Nhận định:", "Mức độ nguy cơ:", "Hướng xử trí / lời khuyên:", "Khi nào cần đi khám:", "Lưu ý an toàn:"]:
                assert section in rsp, f"Missing section {section} for {text}"
    print("All intent and response-structure tests passed")


if __name__ == "__main__":
    run()
