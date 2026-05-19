import re
import unicodedata
from dataclasses import dataclass, field

try:
    import gradio as gr
except ModuleNotFoundError:  # for test environments
    gr = None

APP_NAME = "TRỢ LÝ ẢO HỖ TRỢ TRẢ LỜI CÂU HỎI THƯỜNG GẶP VỀ SỨC KHỎE GIỚI TÍNH"

INTENT_PRIORITY = [
    "emergency",
    "out_of_scope",
    "greeting",
    "ending",
    "missing_info",
    "safe_sex",
    "pregnancy_contraception",
    "sti_std",
    "male_health",
    "female_health",
    "lgbtq",
    "psychology_emotion",
]

INTENT_KEYWORDS = {
    "out_of_scope": [
        "nấu mì", "mì cay", "thời tiết", "mua laptop", "liên quân", "giá vàng", "game", "nấu ăn", "laptop", "điện thoại", "học bài", "tiền bạc", "thể thao",
    ],
    "safe_sex": ["quan hệ lần đầu", "bao cao su", "không dùng bao", "thuốc tránh thai khẩn cấp", "quan hệ ngoài", "xuất tinh ngoài", "ngày an toàn", "loại bao cao su"],
    "pregnancy_contraception": ["trễ kinh", "chậm kinh", "có thai", "mang thai", "thử thai", "que thử thai", "cho con bú", "uống thuốc tránh thai"],
    "sti_std": ["đau rát khi đi tiểu", "mùi hôi", "mụn sinh dục", "hiv", "quan hệ bằng miệng", "giang mai", "sùi mào gà", "lậu", "std", "sti", "loét", "mủ"],
    "male_health": ["xuất tinh sớm", "khó cương", "dương vật cong", "thủ dâm nhiều", "tinh dịch màu vàng", "đau tinh hoàn", "mất ham muốn"],
    "female_health": ["đau bụng khi quan hệ", "khí hư vàng", "ngứa rát vùng kín", "kinh nguyệt không đều", "khô rát khi quan hệ", "đau bụng dưới trước kỳ kinh", "khí hư hôi"],
    "lgbtq": ["bisexual", "gay", "lesbian", "thích người cùng giới", "come out", "chuyển giới", "hormone", "bối rối giới tính", "đồng tính", "song tính"],
    "psychology_emotion": ["sợ mang thai", "không muốn dùng bao", "áp lực khi quan hệ", "tự ti cơ thể", "lo vô sinh", "ép quan hệ", "sợ bệnh xã hội", "lo lắng sau quan hệ"],
    "emergency": ["chảy máu nhiều", "đau bụng dữ dội", "sưng đỏ nghiêm trọng", "xâm hại tình dục", "rách bao cao su", "quá liều thuốc tránh thai khẩn cấp", "ngất", "chóng mặt nhiều", "sốt cao", "mủ", "loét", "sưng đau nặng"],
}

MISSING_INFO_PATTERNS = {
    "đau khi quan hệ": ["Bạn cho mình biết bạn là nam hay nữ, đau ở vị trí nào, kéo dài bao lâu, có chảy máu/khí hư/sốt không?"],
    "ngứa vùng kín": ["Bạn có kèm khí hư bất thường, mùi hôi, đau tiểu hoặc nổi mụn không?"],
    "chậm kinh": ["Bạn trễ kinh bao nhiêu ngày, có quan hệ không bảo vệ không, ngày quan hệ gần nhất và đã thử thai chưa?"],
    "lo mình có thai": ["Bạn trễ kinh bao nhiêu ngày, có quan hệ không bảo vệ không, ngày quan hệ gần nhất và đã thử thai chưa?"],
    "nổi mụn ở vùng kín": ["Mụn có đau/rát/chảy dịch không, xuất hiện bao lâu rồi, và có quan hệ nguy cơ gần đây không?"],
    "đau bụng dưới": ["Mức độ đau hiện tại thế nào, có sốt/chảy máu/khí hư bất thường/chậm kinh không?"],
    "xuất tinh sớm": ["Tình trạng này kéo dài bao lâu, xảy ra thường xuyên không, và bạn có đang căng thẳng nhiều không?"],
    "không có cảm giác khi quan hệ": ["Tình trạng bắt đầu từ khi nào, có áp lực tâm lý/đau/khô rát/mất ham muốn không?"],
}

NEGATIONS = ["không đau", "không ngứa", "không quan hệ", "chưa quan hệ", "không chảy máu"]
GREETING_PATTERNS = [r"^(xin chào|chào|hello|hi|hey|bot ơi)[\s!,.]*$"]
ENDING_PATTERNS = ["tạm biệt", "bye", "kết thúc", "dừng", "xong rồi", "không hỏi nữa", "cảm ơn"]


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower().strip()
    return re.sub(r"\s+", " ", text)


def contains_keyword(text: str, kw: str) -> bool:
    return kw in text


def has_negation_for(text: str, symptom: str) -> bool:
    return any(neg in text and symptom in neg for neg in NEGATIONS)


def detect_intent(text: str) -> str:
    if any(re.match(p, text) for p in GREETING_PATTERNS):
        return "greeting"
    if any(k in text for k in ENDING_PATTERNS):
        return "ending"

    for kw in INTENT_KEYWORDS["emergency"]:
        if contains_keyword(text, kw):
            return "emergency"
    for kw in INTENT_KEYWORDS["out_of_scope"]:
        if contains_keyword(text, kw):
            return "out_of_scope"

    for base in MISSING_INFO_PATTERNS:
        if base in text:
            return "missing_info"

    ordered = ["safe_sex", "pregnancy_contraception", "female_health", "male_health", "sti_std", "lgbtq", "psychology_emotion"]
    best_intent = None
    best_score = 0
    for intent in ordered:
        score = sum(1 for kw in INTENT_KEYWORDS[intent] if kw in text)
        if score > best_score:
            best_score = score
            best_intent = intent
    if best_intent:
        return best_intent
    return "missing_info"


def response_for_intent(intent: str, text: str) -> str:
    if intent == "greeting":
        return "Xin chào 👋 Mình hỗ trợ chuyên sâu về sức khỏe giới tính, sinh sản, STI/STD, tránh thai và tâm lý liên quan."
    if intent == "ending":
        return "Mình đã kết thúc cuộc trò chuyện. Nếu cần, bạn quay lại bất cứ lúc nào nhé."
    if intent == "out_of_scope":
        return "Xin lỗi, tôi chỉ hỗ trợ các vấn đề liên quan đến sức khỏe giới tính, sức khỏe sinh sản, quan hệ an toàn, tránh thai, bệnh lây truyền qua đường tình dục và tâm lý liên quan. Bạn có thể hỏi lại đúng lĩnh vực này nhé."
    if intent == "missing_info":
        for pattern, asks in MISSING_INFO_PATTERNS.items():
            if pattern in text:
                return "Mình chưa đủ dữ kiện để kết luận. " + " ".join(asks)
        return "Mình chưa đủ dữ kiện để kết luận. Bạn mô tả thêm thời gian xảy ra, triệu chứng chính và mức độ hiện tại nhé."

    risk = "🟢"
    if intent == "emergency":
        risk = "🔴"
    elif "rách bao" in text or "không dùng bao" in text:
        risk = "🟠"

    bodies = {
        "safe_sex": "Bao cao su giúp giảm nguy cơ mang thai và STI. Xuất tinh ngoài và ngày an toàn không đảm bảo tuyệt đối. Thuốc tránh thai khẩn cấp hiệu quả nhất khi dùng càng sớm càng tốt và không nên dùng thường xuyên.",
        "pregnancy_contraception": "Trễ kinh có thể do mang thai hoặc stress/rối loạn nội tiết. Nên thử thai khi trễ kinh hoặc sau quan hệ nguy cơ khoảng 10-14 ngày. Đang cho con bú vẫn có thể mang thai. Không lạm dụng thuốc tránh thai khẩn cấp.",
        "sti_std": "HIV không lây qua hôn thông thường nếu không có máu/vết thương hở. Quan hệ bằng miệng vẫn có nguy cơ lây một số STI. Có đau tiểu, mủ, khí hư hôi, loét hoặc mụn sinh dục thì cần đi khám và xét nghiệm STI, không tự mua kháng sinh.",
        "male_health": "Xuất tinh sớm/khó cương có thể liên quan tâm lý, thói quen, nội tiết hoặc bệnh lý. Dương vật cong nhẹ có thể bình thường, nhưng cong đau hoặc khó quan hệ thì nên khám nam khoa. Đau tinh hoàn kèm sưng đỏ/sốt cần khám sớm.",
        "female_health": "Khí hư vàng/hôi/ngứa rát có thể gợi ý viêm nhiễm phụ khoa. Đau khi quan hệ có thể do khô rát, viêm nhiễm, tâm lý hoặc bệnh lý phụ khoa. Kinh nguyệt không đều có thể liên quan stress, nội tiết hoặc bệnh lý.",
        "lgbtq": "Bối rối xu hướng tính dục/bản dạng giới là điều có thể gặp và không có gì đáng xấu hổ. Come out nên ưu tiên an toàn và người tin cậy, không cần vội. Hormone chuyển giới cần bác sĩ theo dõi, không tự mua dùng.",
        "psychology_emotion": "Mình hiểu bạn đang lo lắng. Bạn có quyền từ chối khi chưa an toàn hoặc chưa sẵn sàng. Nếu bị ép quan hệ, hãy ưu tiên an toàn cá nhân và tìm người tin cậy hỗ trợ. Đánh giá nguy cơ thực tế rồi mới kết luận để tránh hoảng sợ quá mức.",
        "emergency": "Đây là tình huống nguy cơ cao cần xử trí sớm. Nếu chảy máu nhiều, đau dữ dội, ngất/chóng mặt nhiều, nghi xâm hại hoặc sưng đau nghiêm trọng: đi cấp cứu ngay. Nếu rách bao: cần đánh giá thời điểm, nguy cơ mang thai và STI để xử trí kịp thời.",
    }
    body = bodies.get(intent, "")
    return (
        f"Nhận định: {body}\n"
        f"Mức độ nguy cơ: {risk}\n"
        "Hướng xử trí / lời khuyên: Ưu tiên theo dõi triệu chứng thật sự, không tự dùng thuốc kê đơn, và áp dụng biện pháp an toàn phù hợp.\n"
        "Khi nào cần đi khám: Khi triệu chứng kéo dài, nặng dần, hoặc có dấu hiệu cảnh báo như sốt cao/chảy máu nhiều/đau dữ dội.\n"
        "Lưu ý an toàn: Thông tin này để tham khảo và không thay thế khám trực tiếp."
    )


@dataclass
class ChatState:
    asked_questions: list[str] = field(default_factory=list)


def chatbot(message: str, history: list, state: ChatState):
    text = normalize_text(message or "")
    if not text:
        return "Bạn hãy nhập câu hỏi để mình hỗ trợ nhé.", state

    intent = detect_intent(text)
    if intent == "ending":
        state.asked_questions.clear()
    else:
        state.asked_questions.append(text)
    return response_for_intent(intent, text), state


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

if gr is not None:
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

if __name__ == "__main__" and gr is not None:
    demo.launch()
