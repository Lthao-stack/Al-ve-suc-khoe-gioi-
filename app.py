import re
import unicodedata
from dataclasses import dataclass, field

import gradio as gr
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

APP_NAME = "TRỢ LÝ ẢO HỖ TRỢ TRẢ LỜI CÂU HỎI THƯỜNG GẶP VỀ SỨC KHỎE GIỚI TÍNH"

knowledge_base = [
    {"topic": "dậy thì bình thường tuổi dậy thì nam nữ", "answer": "Dậy thì là giai đoạn cơ thể phát triển từ trẻ em sang người trưởng thành. Ở nữ thường bắt đầu khoảng 8 đến 13 tuổi, ở nam thường khoảng 9 đến 14 tuổi."},
    {"topic": "thủ dâm có hại không", "answer": "Thủ dâm là hiện tượng sinh lý có thể gặp ở tuổi dậy thì và không phải lúc nào cũng có hại. Nếu lạm dụng đến mức ảnh hưởng học tập, giấc ngủ hoặc tâm lý thì nên điều chỉnh."},
    {"topic": "quan hệ tình dục an toàn bao cao su", "answer": "Quan hệ tình dục an toàn là bảo vệ sức khỏe, tôn trọng sự đồng thuận và giảm nguy cơ mang thai ngoài ý muốn hoặc bệnh lây truyền qua đường tình dục."},
    {"topic": "thuốc tránh thai khẩn cấp", "answer": "Thuốc tránh thai khẩn cấp là biện pháp tạm thời sau tình huống có nguy cơ, không nên lạm dụng. Nếu cần dùng, nên đọc hướng dẫn thuốc hoặc hỏi dược sĩ."},
    {"topic": "dấu hiệu bệnh lây truyền qua đường tình dục sti", "answer": "Dấu hiệu có thể gồm ngứa rát vùng kín, đau khi đi tiểu, dịch tiết bất thường hoặc vết loét lạ. Một số bệnh có thể không có triệu chứng rõ ràng nên cần đi khám khi có nguy cơ."},
    {"topic": "mang thai ngoài ý muốn dấu hiệu có thai que thử thai", "answer": "Dấu hiệu có thai có thể gồm chậm kinh, căng ngực, buồn nôn, mệt mỏi. Có thể dùng que thử thai sau quan hệ 7–14 ngày hoặc sau khi trễ kinh để kết quả tin cậy hơn."},
    {"topic": "xuất tinh sớm", "answer": "Xuất tinh sớm là tình trạng khá phổ biến và có thể liên quan đến căng thẳng hoặc áp lực tâm lý. Có thể cải thiện bằng thay đổi lối sống và hỗ trợ y khoa phù hợp."},
    {"topic": "rối loạn cương dương", "answer": "Rối loạn cương dương có thể liên quan đến stress, thiếu ngủ, chất kích thích hoặc bệnh lý nền. Nếu kéo dài và ảnh hưởng cuộc sống, nên đi khám."},
    {"topic": "đồng thuận trong tình dục", "answer": "Đồng thuận là khi cả hai đều tự nguyện, tỉnh táo và đồng ý rõ ràng. Đồng thuận có thể được rút lại bất cứ lúc nào."},
    {"topic": "bị lạm dụng tình dục tổn thương tâm lý", "answer": "Người từng bị lạm dụng có thể gặp lo âu, sợ hãi hoặc ám ảnh kéo dài. Đây không phải lỗi của nạn nhân, nên tìm hỗ trợ từ chuyên gia hoặc người đáng tin cậy."},
]

OUT_OF_SCOPE_KEYWORDS = {
    "game", "gaming", "liên quân", "free fire", "pubg", "thời tiết", "nấu ăn", "công thức", "điện thoại", "iphone",
    "android", "laptop", "macbook", "phim", "anime", "bóng đá", "thể thao", "chứng khoán", "tiền ảo", "bitcoin",
    "lập trình", "python", "javascript", "học toán", "bài tập toán",
}

PSYCHOLOGICAL_SIGNALS = {
    "lo lắng có thai", "lo dính bầu", "sợ có thai", "áp lực quan hệ", "bị thúc ép", "tự ti cơ thể", "mặc cảm cơ thể",
    "bị ép quan hệ", "sợ bệnh xã hội", "mất cảm xúc", "lạnh nhạt chuyện ấy", "stress sau quan hệ", "lo vô sinh",
    "sợ bị đánh giá giới tính", "sợ lộ xu hướng", "ám ảnh", "hoảng loạn", "overthinking", "toang", "khủng hoảng",
}

DANGER_RULES = [
    ("🔴 nguy hiểm", ["ngất", "chóng mặt", "sốt cao", "xâm hại", "bị hiếp", "chảy máu ồ ạt", "đau dữ dội"]),
    ("🟠 cần khám sớm", ["chảy máu sau quan hệ", "rách bao", "nghi hiv", "nghi std", "mưng mủ", "sưng đỏ", "tiểu buốt nặng", "phát ban"]),
    ("🟡 trung bình", ["uống quá liều thuốc tránh thai", "đau bụng dưới", "dịch lạ", "lo mắc bệnh xã hội"]),
]

NEGATION_PATTERNS = ["không đau", "không bị ngứa", "không ngứa", "chưa quan hệ", "không quan hệ"]


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower().strip()
    return re.sub(r"\s+", " ", text)


def has_keyword(text: str, keyword: str) -> bool:
    return keyword in text


def is_negated(text: str, symptom: str) -> bool:
    return any(f"{neg} {symptom}" in text or neg in text for neg in NEGATION_PATTERNS)


df = pd.DataFrame(knowledge_base)
documents = (df["topic"] + " " + df["answer"]).tolist()
vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2))
tfidf_matrix = vectorizer.fit_transform(documents)

GREETING_PATTERNS = [r"^(xin chào|chào|hello|hi|hey|bot ơi)[\s!,.]*$"]
END_KEYWORDS = ["tạm biệt", "bye", "kết thúc", "dừng", "hết rồi", "không hỏi nữa", "xong rồi", "cảm ơn"]


@dataclass
class AgentContext:
    text: str
    topic: str = ""
    answer: str | None = None
    warning: str = ""
    risk_level: str = "🟢 nhẹ"
    out_of_scope: bool = False
    psychological: bool = False


def scope_guard_agent(text: str) -> bool:
    return any(k in text for k in OUT_OF_SCOPE_KEYWORDS)


def planner_agent(text: str) -> str:
    if any(k in text for k in ["gay", "lesbian", "bisexual", "lgbt", "come out", "đồng tính", "song tính", "chuyển giới"]):
        return "LGBTQ+ và bản dạng giới"
    if any(k in text for k in ["tâm lý", "lo", "sợ", "áp lực", "mất cảm xúc", "stress", "tự ti"]):
        return "Tâm lý và cảm xúc tình dục"
    if any(k in text for k in ["bao cao su", "tránh thai", "thuốc tránh thai", "có thai", "que thử thai"]):
        return "Tình dục an toàn và tránh thai"
    if any(k in text for k in ["sti", "std", "hiv", "bệnh lây", "ngứa", "rát", "dịch", "mùi hôi", "viêm"]):
        return "Sức khỏe sinh sản và bệnh lý"
    return "Sức khỏe giới tính chung"


def detect_psychological_agent(text: str) -> bool:
    return any(k in text for k in PSYCHOLOGICAL_SIGNALS)


def retrieve_answer_agent(question: str):
    user_vec = vectorizer.transform([question])
    scores = cosine_similarity(user_vec, tfidf_matrix).flatten()
    idx = int(scores.argmax())
    if scores[idx] < 0.12:
        return None
    return df.iloc[idx]["answer"]


def safety_agent(text: str) -> tuple[str, str]:
    for risk, signals in DANGER_RULES:
        if any(s in text and not is_negated(text, s) for s in signals):
            if risk == "🔴 nguy hiểm":
                msg = "\n\n**Cảnh báo khẩn:** Bạn có dấu hiệu nguy hiểm cao. Hãy đến cơ sở y tế/cấp cứu ngay, ưu tiên an toàn cá nhân."
            elif risk == "🟠 cần khám sớm":
                msg = "\n\n**Cảnh báo:** Đây là tình huống cần khám sớm trong ngày để được xử trí đúng (xét nghiệm, dự phòng phơi nhiễm, xử lý tổn thương)."
            else:
                msg = "\n\n**Lưu ý an toàn:** Triệu chứng mức trung bình, cần theo dõi sát và đi khám nếu kéo dài hoặc nặng hơn."
            return risk, msg
    return "🟢 nhẹ", ""


def psychological_support_agent(text: str) -> str:
    if not detect_psychological_agent(text):
        return ""

    guidance = [
        "- Cảm xúc lo lắng/xấu hổ lúc này là điều dễ hiểu, bạn không cô đơn.",
        "- Mình khuyên bạn tập trung vào dấu hiệu thực tế (thời điểm quan hệ, biện pháp bảo vệ, triệu chứng thật sự).",
        "- Nếu cảm giác hoảng loạn kéo dài, hãy nói chuyện với người tin cậy hoặc chuyên gia tâm lý.",
    ]

    if any(k in text for k in ["bị ép", "xâm hại", "cưỡng ép"]):
        guidance.insert(0, "- Bạn cần ưu tiên an toàn cá nhân ngay: rời khỏi môi trường nguy hiểm và liên hệ người hỗ trợ.")

    return "\n\n**Hỗ trợ tâm lý:**\n" + "\n".join(guidance)


def ask_followup_agent(text: str) -> str:
    missing = []
    if not any(k in text for k in ["bao cao su", "không bao", "rách bao", "xuất tinh", "chưa quan hệ"]):
        missing.append("bạn có dùng biện pháp bảo vệ nào không")
    if not any(k in text for k in ["bao lâu", "mấy ngày", "hôm qua", "hôm kia", "tuần trước"]):
        missing.append("sự việc xảy ra cách đây bao lâu")
    if not any(k in text for k in ["đau", "chảy máu", "sốt", "dịch", "ngứa", "không đau", "không bị ngứa"]):
        missing.append("hiện tại có triệu chứng cụ thể nào không")

    if not missing:
        return ""
    return "\n\n**Mình cần thêm thông tin để tư vấn chính xác hơn:** " + "; ".join(missing) + "."


def out_of_scope_response() -> str:
    return (
        "Xin lỗi, tôi chỉ hỗ trợ các vấn đề liên quan sức khỏe giới tính và sinh sản. "
        "Bạn có thể đặt câu hỏi về tránh thai, STI/STD, an toàn khi quan hệ, hoặc sức khỏe tâm lý liên quan tình dục nhé."
    )


def responder_agent(ctx: AgentContext) -> str:
    if ctx.out_of_scope:
        return out_of_scope_response()

    base = ctx.answer or "Mình chưa thấy đủ dữ liệu để kết luận ngay."
    response = f"### Nhóm nội dung: {ctx.topic}\n\n**Mức nguy cơ hiện tại: {ctx.risk_level}**\n\n{base}"
    response += psychological_support_agent(ctx.text)
    response += ctx.warning
    response += ask_followup_agent(ctx.text)
    response += "\n\n> Lưu ý: Thông tin chỉ mang tính tham khảo, không thay thế tư vấn y tế chuyên môn."
    response += "\n\n<sub>Agentic flow: Scope Guard → Planner → Retriever → Safety → Psychological Support → Responder</sub>"
    return response


@dataclass
class ChatState:
    asked_questions: list[str] = field(default_factory=list)

    def is_repeat(self, message: str) -> bool:
        if not self.asked_questions:
            return False
        texts = self.asked_questions + [message]
        temp = vectorizer.transform(texts)
        scores = cosine_similarity(temp[-1], temp[:-1]).flatten()
        return bool(scores.max() > 0.92)


def chatbot(message: str, history: list, state: ChatState):
    text = normalize_text(message or "")
    if not text:
        return "Bạn hãy nhập câu hỏi để mình hỗ trợ nhé.", state

    if any(re.match(p, text) for p in GREETING_PATTERNS):
        return "Xin chào 👋 Mình có thể hỗ trợ các câu hỏi về sức khỏe giới tính, sinh sản và an toàn tâm lý khi quan hệ.", state

    if any(k in text for k in END_KEYWORDS):
        state.asked_questions.clear()
        return "Mình đã kết thúc cuộc trò chuyện và làm mới ngữ cảnh. Cảm ơn bạn!", state

    if state.is_repeat(text):
        return "Bạn đang hỏi nội dung gần giống trước đó. Nếu được, hãy thêm mốc thời gian, triệu chứng và mức độ để mình tư vấn đúng hơn.", state

    state.asked_questions.append(text)

    ctx = AgentContext(text=text)
    ctx.out_of_scope = scope_guard_agent(text)
    if not ctx.out_of_scope:
        ctx.topic = planner_agent(text)
        ctx.answer = retrieve_answer_agent(text)
        ctx.risk_level, ctx.warning = safety_agent(text)
        ctx.psychological = detect_psychological_agent(text)

    return responder_agent(ctx), state


custom_css = """
:root {
    --bg: #0b1220;
    --surface: #111827;
    --text: #e5e7eb;
    --accent: #22c55e;
}
.gradio-container {background: linear-gradient(135deg, #0b1220 0%, #111827 100%) !important;}
footer {visibility: hidden}
#title-banner {padding: 14px; border-radius: 14px; background: #0f172a; color: var(--text); border: 1px solid #1f2937;}
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Soft(primary_hue="green")) as demo:
    gr.Markdown(f"## {APP_NAME}", elem_id="title-banner")
    gr.Markdown("Đặt câu hỏi ngắn gọn, rõ ý để nhận câu trả lời phù hợp hơn.")
    state = gr.State(ChatState())
    gr.ChatInterface(
        fn=chatbot,
        additional_inputs=[state],
        additional_outputs=[state],
        textbox=gr.Textbox(placeholder="Ví dụ: Trễ kinh 10 ngày thì nên làm gì?", lines=2),
        examples=[
            "Em lo có thai dù dùng bao cao su thì nên làm gì?",
            "Sau quan hệ bị chảy máu và đau dữ dội có nguy hiểm không?",
            "Mình bị ép quan hệ và rất hoảng loạn, giờ nên làm gì?",
        ],
    )

if __name__ == "__main__":
    demo.launch()
