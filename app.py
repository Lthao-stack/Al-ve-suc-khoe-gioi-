import re
import unicodedata
from dataclasses import dataclass, field

try:
    import gradio as gr
except ModuleNotFoundError:  # for test environments
    gr = None

APP_NAME = "TRỢ LÝ ẢO HỖ TRỢ TRẢ LỜI CÂU HỎI THƯỜNG GẶP VỀ SỨC KHỎE GIỚI TÍNH"

SIMILARITY_THRESHOLD = 0.22
MIN_LEN_FOR_CONFIDENT = 8

OUT_OF_SCOPE_REPLY = (
    "Mình chỉ hỗ trợ các câu hỏi về sức khỏe giới tính, sức khỏe sinh sản, dậy thì, "
    "tránh thai, bệnh lây truyền qua đường tình dục và các vấn đề liên quan."
)

INTENT_KEYWORDS = {
    "greeting": ["xin chào", "chào", "hello", "hi", "hey", "bot ơi"],
    "ending": ["tạm biệt", "bye", "kết thúc", "dừng", "xong rồi", "không hỏi nữa", "cảm ơn"],
    "out_of_scope": [
        "thời tiết", "mua laptop", "giá vàng", "điểm thi", "chứng khoán", "bóng đá", "nấu ăn", "game", "điện thoại"
    ],
    "safe_sex": [
        "quan hệ an toàn", "đeo bao", "bao cao su", "quan hệ lần đầu", "cọ xát ngoài", "xuất tinh ngoài", "xuất vào miệng", "quan hệ bằng miệng",
        "ngày an toàn", "quan hệ không dùng bao", "rách bao"
    ],
    "contraception_pregnancy": [
        "trễ kinh", "chậm kinh", "mang thai", "có thai", "que thử thai", "thử thai", "thuốc tránh thai", "khẩn cấp", "uống thuốc ngừa thai", "cho con bú có thai"
    ],
    "sti_std": [
        "hiv", "hpv", "lậu", "giang mai", "herpes", "mụn sinh dục", "loét", "tiết dịch", "mủ", "đau rát khi tiểu", "bệnh xã hội", "std", "sti"
    ],
    "puberty": [
        "dậy thì", "vỡ giọng", "mọc lông", "ngực phát triển", "mộng tinh", "kinh nguyệt lần đầu", "cao lên", "tuổi dậy thì"
    ],
    "hygiene": [
        "vệ sinh vùng kín", "rửa vùng kín", "dung dịch vệ sinh", "khí hư", "viêm âm đạo", "viêm đường tiểu", "ngứa vùng kín", "mùi hôi"
    ],
    "lgbtq": ["gay", "lesbian", "đồng tính", "song tính", "chuyển giới", "come out", "bản dạng giới", "lgbt"],
    "psychology_consent": [
        "bị ép", "không đồng thuận", "sợ mang thai", "lo quá", "hoảng loạn", "ám ảnh", "đồng thuận", "tâm lý", "xâm hại"
    ],
    "sexual_function": [
        "xuất tinh sớm", "rối loạn cương", "khó cương", "đau khi quan hệ", "giảm ham muốn", "thủ dâm", "không lên đỉnh", "đau dương vật"
    ],
}

MISSING_SHORT_PHRASES = ["bị sao", "có sao không", "lo quá", "sao đây", "giờ sao", "mình lo"]

KNOWLEDGE_BASE = [
    {"topic": "dậy thì nam", "keywords": ["dậy thì nam", "vỡ giọng", "mọc ria", "mộng tinh", "tăng chiều cao"], "answer": "Dậy thì nam thường bắt đầu khoảng 9-14 tuổi với các dấu hiệu: tinh hoàn lớn dần, vỡ giọng, mọc lông, có thể mộng tinh. Đây thường là thay đổi sinh lý bình thường."},
    {"topic": "dậy thì nữ", "keywords": ["dậy thì nữ", "ngực phát triển", "kinh nguyệt đầu", "mọc lông mu", "cao nhanh"], "answer": "Dậy thì nữ thường bắt đầu khoảng 8-13 tuổi. Dấu hiệu thường gặp gồm ngực phát triển, cao nhanh, có kinh nguyệt lần đầu và thay đổi cảm xúc."},
    {"topic": "kinh nguyệt trễ", "keywords": ["trễ kinh", "chậm kinh", "mất kinh", "kinh không đều"], "answer": "Trễ kinh có thể do mang thai, căng thẳng, thay đổi cân nặng, rối loạn nội tiết hoặc bệnh lý phụ khoa. Nên thử thai đúng thời điểm nếu có nguy cơ."},
    {"topic": "đau bụng kinh", "keywords": ["đau bụng kinh", "đau bụng ngày đèn đỏ", "đau trước kỳ kinh"], "answer": "Đau bụng kinh mức nhẹ-vừa khá thường gặp. Nếu đau dữ dội, kéo dài hoặc kèm ngất/sốt/chảy máu bất thường thì cần khám phụ khoa."},
    {"topic": "khí hư", "keywords": ["khí hư", "dịch âm đạo", "huyết trắng", "khí hư hôi"], "answer": "Khí hư sinh lý thường trong hoặc trắng sữa, không mùi hôi nặng, không ngứa rát. Khí hư vàng/xanh, hôi, ngứa hoặc đau rát gợi ý viêm nhiễm cần khám."},
    {"topic": "viêm âm đạo", "keywords": ["viêm âm đạo", "ngứa âm đạo", "viêm phụ khoa", "mùi tanh"], "answer": "Viêm âm đạo có thể do nấm, vi khuẩn hoặc ký sinh trùng. Cần khám để xác định nguyên nhân, tránh tự mua thuốc đặt/kháng sinh."},
    {"topic": "viêm đường tiểu", "keywords": ["viêm đường tiểu", "tiểu buốt", "tiểu rát", "tiểu nhiều lần"], "answer": "Tiểu buốt/rát, tiểu lắt nhắt có thể do viêm đường tiểu. Uống đủ nước, đi khám sớm nếu sốt, đau hông lưng hoặc tiểu máu."},
    {"topic": "hpv", "keywords": ["hpv", "sùi mào gà", "mụn cóc sinh dục", "vắc xin hpv"], "answer": "HPV lây chủ yếu qua tiếp xúc tình dục. Tiêm vắc xin HPV và dùng bao cao su giúp giảm nguy cơ, dù không bảo vệ tuyệt đối."},
    {"topic": "hiv", "keywords": ["hiv", "phơi nhiễm hiv", "pep", "prep"], "answer": "HIV lây qua máu, tình dục không bảo vệ, và từ mẹ sang con. Nếu có phơi nhiễm nguy cơ cao, cần tư vấn PEP càng sớm càng tốt (tốt nhất trong 72 giờ)."},
    {"topic": "lậu", "keywords": ["lậu", "chảy mủ", "tiểu buốt", "gonorrhea"], "answer": "Lậu có thể gây tiểu buốt, chảy mủ niệu đạo/âm đạo, đau vùng chậu. Cần xét nghiệm và điều trị đúng phác đồ, điều trị cả bạn tình."},
    {"topic": "giang mai", "keywords": ["giang mai", "săng", "phát ban lòng bàn tay", "syphilis"], "answer": "Giang mai có nhiều giai đoạn, có thể khởi đầu bằng vết loét không đau. Cần xét nghiệm sớm vì bệnh có thể gây biến chứng nếu bỏ qua."},
    {"topic": "herpes", "keywords": ["herpes", "mụn nước sinh dục", "rát sinh dục", "hsv"], "answer": "Herpes sinh dục có thể gây mụn nước đau rát tái phát. Điều trị giúp giảm triệu chứng và giảm lây truyền, nhưng virus có thể tồn tại lâu dài."},
    {"topic": "bao cao su", "keywords": ["bao cao su", "đeo bao", "rách bao", "tuột bao"], "answer": "Bao cao su giúp giảm nguy cơ mang thai và STI khi dùng đúng cách từ đầu đến cuối cuộc quan hệ."},
    {"topic": "thuốc tránh thai hằng ngày", "keywords": ["thuốc tránh thai hằng ngày", "quên thuốc", "uống viên tránh thai"], "answer": "Thuốc tránh thai hằng ngày cần uống đều mỗi ngày đúng giờ để duy trì hiệu quả. Quên thuốc có thể làm tăng nguy cơ mang thai."},
    {"topic": "thuốc tránh thai khẩn cấp", "keywords": ["thuốc tránh thai khẩn cấp", "uống thuốc khẩn cấp", "72 giờ", "120 giờ"], "answer": "Thuốc tránh thai khẩn cấp hiệu quả cao hơn khi dùng càng sớm càng tốt sau quan hệ nguy cơ. Không nên lạm dụng như biện pháp thường xuyên."},
    {"topic": "que thử thai", "keywords": ["que thử thai", "thử thai", "2 vạch", "1 vạch"], "answer": "Nên thử thai sau quan hệ nguy cơ khoảng 10-14 ngày hoặc khi trễ kinh để tăng độ chính xác. Có thể thử lại sau 48 giờ nếu chưa rõ."},
    {"topic": "cọ xát ngoài", "keywords": ["cọ xát ngoài", "không đưa vào", "chạm bên ngoài"], "answer": "Cọ xát ngoài vẫn có nguy cơ mang thai thấp nếu tinh dịch tiếp xúc gần cửa âm đạo, và vẫn có nguy cơ STI qua tiếp xúc da-niêm mạc."},
    {"topic": "xuất tinh ngoài", "keywords": ["xuất tinh ngoài", "rút ra ngoài", "không xuất trong"], "answer": "Xuất tinh ngoài có tỷ lệ thất bại cao hơn bao cao su/thuốc tránh thai do có thể có tinh trùng trong dịch trước xuất tinh."},
    {"topic": "vệ sinh vùng kín", "keywords": ["vệ sinh vùng kín", "rửa vùng kín", "dung dịch vệ sinh"], "answer": "Nên vệ sinh nhẹ nhàng bằng nước sạch hoặc sản phẩm phù hợp, tránh thụt rửa sâu vì dễ làm mất cân bằng hệ vi sinh."},
    {"topic": "thủ dâm", "keywords": ["thủ dâm", "tự kích thích", "quay tay"], "answer": "Thủ dâm nhìn chung là hành vi tình dục bình thường nếu không gây đau, ám ảnh, ảnh hưởng học tập/công việc và không kèm hành vi nguy hiểm."},
    {"topic": "mộng tinh", "keywords": ["mộng tinh", "xuất tinh lúc ngủ"], "answer": "Mộng tinh là hiện tượng sinh lý thường gặp ở tuổi dậy thì và người trẻ, thường không nguy hiểm nếu không kèm đau hoặc triệu chứng bất thường."},
    {"topic": "xuất tinh sớm", "keywords": ["xuất tinh sớm", "ra nhanh", "chưa vào đã ra"], "answer": "Xuất tinh sớm có thể liên quan tâm lý, kỹ thuật quan hệ, hoặc yếu tố sinh học. Có thể cải thiện bằng tư vấn và điều trị phù hợp."},
    {"topic": "rối loạn cương", "keywords": ["rối loạn cương", "khó cương", "không cương"], "answer": "Rối loạn cương có thể liên quan stress, giấc ngủ, bệnh nền mạch máu/nội tiết hoặc thuốc. Nếu kéo dài, nên khám nam khoa."},
    {"topic": "đau khi quan hệ", "keywords": ["đau khi quan hệ", "quan hệ bị đau", "đau lúc thâm nhập"], "answer": "Đau khi quan hệ có thể do khô rát, viêm nhiễm, căng cơ sàn chậu, tâm lý hoặc bệnh lý cơ quan sinh dục. Cần đánh giá nguyên nhân cụ thể."},
    {"topic": "đồng thuận", "keywords": ["đồng thuận", "bị ép", "cưỡng ép", "không muốn"], "answer": "Quan hệ tình dục cần sự đồng thuận tự nguyện từ tất cả các bên. Nếu bị ép buộc, ưu tiên an toàn, tìm hỗ trợ tin cậy và chăm sóc y tế/pháp lý."},
    {"topic": "lgbtq+", "keywords": ["lgbt", "đồng tính", "song tính", "chuyển giới", "come out"], "answer": "Xu hướng tính dục và bản dạng giới đa dạng là bình thường. Bạn có thể tìm cộng đồng hỗ trợ an toàn và chuyên gia tâm lý thân thiện LGBTQ+."},
]


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower().strip()
    return re.sub(r"\s+", " ", text)


def tokenize(text: str) -> set[str]:
    return {t for t in re.findall(r"[\wàáạảãăắằặẳẵâấầậẩẫèéẹẻẽêếềệểễìíịỉĩòóọỏõôốồộổỗơớờợởỡùúụủũưứừựửữỳýỵỷỹđ]+", text) if t}


def jaccard_similarity(a: str, b: str) -> float:
    sa, sb = tokenize(a), tokenize(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def detect_intent(text: str) -> str:
    if any(re.fullmatch(r"(xin chào|chào|hello|hi|hey|bot ơi)[\s!,.]*", text) for _ in [0]):
        return "greeting"
    if any(k in text for k in INTENT_KEYWORDS["ending"]):
        return "ending"

    if any(k in text for k in ["lo lắng", "hoảng", "bị ép", "không đồng thuận"]) and any(x in text for x in ["quan hệ", "tình dục", "mang thai"]):
        return "psychology_consent"

    best_intent = "out_of_scope"
    best_score = 0.0
    for intent, kws in INTENT_KEYWORDS.items():
        if intent in {"greeting", "ending"}:
            continue
        if any(kw in text for kw in kws):
            return intent
        score = max((jaccard_similarity(text, kw) for kw in kws), default=0.0)
        if score > best_score:
            best_score, best_intent = score, intent

    if best_score < SIMILARITY_THRESHOLD:
        return "out_of_scope"

    if len(text) < MIN_LEN_FOR_CONFIDENT or any(p in text for p in MISSING_SHORT_PHRASES):
        return "missing_info"

    return best_intent


def infer_missing_fields(text: str, asked: set[str]) -> list[str]:
    needed = []
    if any(k in text for k in ["có thai", "mang thai", "trễ kinh", "chậm kinh", "thuốc tránh thai", "cọ xát", "xuất tinh ngoài"]):
        if "preg_sex_contact" not in asked:
            needed.append("Bạn có quan hệ thâm nhập hay chỉ cọ xát bên ngoài?")
        if "preg_ejaculation" not in asked:
            needed.append("Có xuất tinh trong hoặc gần âm đạo không?")
        if "preg_timing" not in asked:
            needed.append("Sự việc xảy ra khi nào (ngày/tháng gần đúng)?")
        if "preg_lmp" not in asked:
            needed.append("Ngày đầu kỳ kinh gần nhất của bạn là khi nào?")

    if any(k in text for k in ["sti", "std", "hiv", "hpv", "lậu", "giang mai", "herpes", "viêm", "tiểu buốt", "mụn", "loét"]):
        if "sti_symptoms" not in asked:
            needed.append("Bạn đang có triệu chứng gì cụ thể (đau, rát, mụn, loét, dịch bất thường...)?")
        if "sti_duration" not in asked:
            needed.append("Triệu chứng xuất hiện từ khi nào?")
        if "sti_risk" not in asked:
            needed.append("Gần đây có quan hệ nguy cơ không dùng bảo vệ không?")

    if any(k in text for k in ["dậy thì", "mộng tinh", "vỡ giọng", "ngực phát triển", "kinh nguyệt đầu"]):
        if "puberty_age" not in asked:
            needed.append("Bạn bao nhiêu tuổi?")
        if "puberty_sex" not in asked:
            needed.append("Giới tính sinh học của bạn là nam hay nữ?")
        if "puberty_signs" not in asked:
            needed.append("Biểu hiện cụ thể bạn đang gặp là gì?")

    return needed


def find_kb_answer(text: str) -> tuple[str, float]:
    best = ""
    best_score = 0.0
    for item in KNOWLEDGE_BASE:
        score = max((jaccard_similarity(text, kw) for kw in item["keywords"]), default=0.0)
        if score > best_score:
            best_score = score
            best = item["answer"]
    return best, best_score


def response_for_intent(intent: str, text: str, state) -> str:
    if intent == "greeting":
        return "Xin chào 👋 Mình hỗ trợ các câu hỏi về sức khỏe giới tính, sinh sản, dậy thì và STI/STD."
    if intent == "ending":
        return "Mình đã kết thúc cuộc trò chuyện. Khi cần bạn quay lại nhé."
    if intent == "out_of_scope":
        return OUT_OF_SCOPE_REPLY

    missing = infer_missing_fields(text, state.asked_slots)
    if intent == "missing_info" or missing:
        if missing:
            for q in missing:
                if "quan hệ thâm nhập" in q:
                    state.asked_slots.add("preg_sex_contact")
                elif "xuất tinh" in q:
                    state.asked_slots.add("preg_ejaculation")
                elif "xảy ra khi nào" in q:
                    state.asked_slots.add("preg_timing")
                elif "kỳ kinh" in q:
                    state.asked_slots.add("preg_lmp")
                elif "triệu chứng gì" in q:
                    state.asked_slots.add("sti_symptoms")
                elif "xuất hiện từ khi nào" in q:
                    state.asked_slots.add("sti_duration")
                elif "quan hệ nguy cơ" in q:
                    state.asked_slots.add("sti_risk")
                elif "bao nhiêu tuổi" in q:
                    state.asked_slots.add("puberty_age")
                elif "Giới tính sinh học" in q:
                    state.asked_slots.add("puberty_sex")
                elif "Biểu hiện cụ thể" in q:
                    state.asked_slots.add("puberty_signs")
            return "Mình chưa đủ dữ kiện để nhận định. " + " ".join(missing)
        return "Mình chưa đủ dữ kiện để nhận định. Bạn mô tả rõ bối cảnh, thời điểm và triệu chứng cụ thể nhé."

    answer, score = find_kb_answer(text)
    if score < SIMILARITY_THRESHOLD:
        return "Mình chưa đủ dữ kiện để nhận định. Bạn mô tả rõ hơn về triệu chứng, thời điểm và hành vi nguy cơ nhé."

    return (
        f"Nhận định:\n- {answer}\n"
        "Mức độ cần lưu ý:\n- Tạm thời chưa thể thay thế chẩn đoán trực tiếp; cần theo dõi dấu hiệu bất thường.\n"
        "Hướng xử trí/lời khuyên:\n- Theo dõi triệu chứng, tránh tự dùng thuốc kê đơn, ưu tiên biện pháp bảo vệ khi quan hệ.\n"
        "Khi nào cần đi khám:\n- Nếu triệu chứng kéo dài, nặng dần, tái phát hoặc có dấu hiệu cảnh báo như sốt cao, đau dữ dội, chảy máu bất thường.\n"
        "Lưu ý an toàn:\n- Thông tin mang tính tham khảo, không thay thế tư vấn/chẩn đoán của bác sĩ."
    )


@dataclass
class ChatState:
    asked_questions: list[str] = field(default_factory=list)
    asked_slots: set[str] = field(default_factory=set)


def chatbot(message: str, history: list, state: ChatState):
    text = normalize_text(message or "")
    if not text:
        return "Bạn hãy nhập câu hỏi để mình hỗ trợ nhé.", state

    intent = detect_intent(text)
    if intent == "ending":
        state.asked_questions.clear()
        state.asked_slots.clear()
    else:
        state.asked_questions.append(text)
    return response_for_intent(intent, text, state), state


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
            textbox=gr.Textbox(placeholder="Nhập câu hỏi của bạn...", lines=2),
        )

if __name__ == "__main__" and gr is not None:
    demo.launch()
