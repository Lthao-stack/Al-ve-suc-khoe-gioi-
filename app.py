import json
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

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

KNOWLEDGE_BASE_PATH = Path(__file__).with_name("knowledge_base.json")
ALLOWED_SOURCES = ("who", "cdc", "bộ y tế", "bo y te", "bệnh viện", "benh vien")


def load_knowledge_base() -> list[dict]:
    if not KNOWLEDGE_BASE_PATH.exists():
        return []
    with KNOWLEDGE_BASE_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def _is_approved_source(source: str) -> bool:
    src = normalize_text(source)
    return any(token in src for token in ALLOWED_SOURCES)


def update_knowledge_base(entry: dict) -> tuple[bool, str]:
    required = {"topic", "keywords", "answer", "source", "last_updated"}
    if not required.issubset(entry.keys()):
        return False, "Thiếu trường bắt buộc: topic, keywords, answer, source, last_updated"
    if not _is_approved_source(entry["source"]):
        return False, "Nguồn chưa được duyệt. Chỉ chấp nhận WHO, CDC, Bộ Y tế, bệnh viện uy tín."
    if not isinstance(entry["keywords"], list) or not entry["keywords"]:
        return False, "keywords phải là danh sách không rỗng"

    kb = load_knowledge_base()
    kb = [item for item in kb if item.get("topic") != entry["topic"]]
    kb.append(entry)
    with KNOWLEDGE_BASE_PATH.open("w", encoding="utf-8") as f:
        json.dump(kb, f, ensure_ascii=False, indent=2)
    return True, "Đã cập nhật knowledge base"


KNOWLEDGE_BASE = load_knowledge_base()


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


def find_kb_answer(text: str) -> tuple[dict | None, float]:
    best_item = None
    best_score = 0.0
    for item in KNOWLEDGE_BASE:
        score = max((jaccard_similarity(text, kw) for kw in item.get("keywords", [])), default=0.0)
        if score > best_score:
            best_score = score
            best_item = item
    return best_item, best_score


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

    kb_item, score = find_kb_answer(text)
    if kb_item is None or score < SIMILARITY_THRESHOLD:
        return "Mình chưa đủ dữ liệu phù hợp trong knowledge base đã duyệt để trả lời an toàn. Bạn vui lòng hỏi rõ hơn hoặc nhờ quản trị viên cập nhật tri thức."

    return (
        f"Nhận định:\n- {kb_item["answer"]}\n"
        f"Nguồn tham chiếu: {kb_item.get("source", "N/A")} (cập nhật: {kb_item.get("last_updated", "N/A")})\n"
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
