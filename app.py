import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

try:
    import gradio as gr
except ModuleNotFoundError:  # for test environments
    gr = None

APP_NAME = "TRỢ LÝ ẢO HỖ TRỢ TRẢ LỜI CÂU HỎI THƯỜNG GẶP VỀ SỨC KHỎE GIỚI TÍNH"
SIMILARITY_THRESHOLD = 0.2
MIN_LEN_FOR_CONFIDENT = 8

OUT_OF_SCOPE_REPLY = (
    "Mình chỉ hỗ trợ các câu hỏi về sức khỏe giới tính, sức khỏe sinh sản, dậy thì, "
    "tránh thai, bệnh lây truyền qua đường tình dục và các vấn đề liên quan. "
    "Bạn hãy đặt lại câu hỏi đúng lĩnh vực để mình hỗ trợ an toàn hơn nhé."
)

INTENT_KEYWORDS = {
    "greeting": ["xin chào", "chao", "chào", "hello", "hi", "hey", "bot oi", "alo", "ad ơi"],
    "ending": ["tạm biệt", "bye", "kết thúc", "dừng", "xong rồi", "không hỏi nữa", "cảm ơn", "ổn rồi", "ok rồi"],
    "out_of_scope": ["thời tiết", "mua laptop", "giá vàng", "điểm thi", "chứng khoán", "bóng đá", "nấu ăn", "game"],
    "safe_sex": ["quan hệ an toàn", "đeo bao", "bao cao su", "qhtd", "qhe", "ko bao", "khong bao", "rách bao", "tuột bao", "cọ xát"],
    "contraception_pregnancy": ["trễ kinh", "chậm kinh", "mang thai", "có thai", "que thử thai", "thuốc tránh thai", "khẩn cấp", "quan hệ ngoài", "xuất tinh ngoài"],
    "sti_std": ["hiv", "hpv", "lậu", "giang mai", "herpes", "mụn sinh dục", "loét", "tiết dịch", "mủ", "đau rát khi tiểu", "std", "sti", "mun sinh duc", "bi mun"],
    "puberty": ["dậy thì", "vỡ giọng", "mọc lông", "ngực phát triển", "mộng tinh", "kinh nguyệt lần đầu", "cao lên"],
    "hygiene": ["vệ sinh vùng kín", "rửa vùng kín", "dung dịch vệ sinh", "khí hư", "viêm âm đạo", "ngứa vùng kín", "mùi hôi"],
    "lgbtq": ["gay", "lesbian", "đồng tính", "song tính", "chuyển giới", "come out", "bản dạng giới", "lgbt"],
    "psychology_consent": ["bị ép", "không đồng thuận", "sợ mang thai", "lo quá", "hoảng loạn", "xâm hại", "áp lực"],
    "sexual_function": ["xuất tinh sớm", "rối loạn cương", "khó cương", "đau khi quan hệ", "giảm ham muốn", "không lên đỉnh", "roi loan cuong"],
}

MISSING_SHORT_PHRASES = ["bị sao", "có sao không", "lo quá", "sao đây", "giờ sao", "mình lo", "em lo"]
EMERGENCY_SIGNS = ["ra máu ồ ạt", "đau bụng dữ dội", "ngất", "sốt cao", "khó thở", "bị cưỡng bức", "tự tử"]

KNOWLEDGE_BASE_PATH = Path(__file__).with_name("knowledge_base.json")
ALLOWED_SOURCES = ("who", "cdc", "bộ y tế", "bo y te", "bệnh viện", "benh vien")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", (text or "")).lower().strip()
    text = text.replace("k ", "không ").replace("ko ", "không ")
    return re.sub(r"\s+", " ", text)


def tokenize(text: str) -> set[str]:
    return {t for t in re.findall(r"[\wàáạảãăắằặẳẵâấầậẩẫèéẹẻẽêếềệểễìíịỉĩòóọỏõôốồộổỗơớờợởỡùúụủũưứừựửữỳýỵỷỹđ]+", text) if t}


def jaccard_similarity(a: str, b: str) -> float:
    sa, sb = tokenize(a), tokenize(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


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


@dataclass
class ChatState:
    asked_slots: set[str] = field(default_factory=set)
    memory: dict = field(default_factory=dict)
    last_risk: str = "Chưa đánh giá"


class IntentAgent:
    def detect(self, text: str) -> str:
        if any(re.fullmatch(r"(xin chào|chào|chao bot|hello|hi|hey|alo|bot ơi|ad ơi)[\s!,.]*", text) for _ in [0]):
            return "greeting"
        if any(k in text for k in INTENT_KEYWORDS["ending"]):
            return "ending"
        if any(k in text for k in EMERGENCY_SIGNS):
            return "emergency"
        if any(k in text for k in ["có thai", "mang thai", "trễ kinh", "chậm kinh", "co thai"]):
            return "contraception_pregnancy"
        best_intent, best_score = "out_of_scope", 0.0
        for intent, kws in INTENT_KEYWORDS.items():
            if intent in {"greeting", "ending"}:
                continue
            if any(kw in text for kw in kws):
                return intent
            score = max((jaccard_similarity(text, kw) for kw in kws), default=0.0)
            if score > best_score:
                best_intent, best_score = intent, score
        if best_score < SIMILARITY_THRESHOLD:
            return "out_of_scope"
        if len(text) < MIN_LEN_FOR_CONFIDENT or any(p in text for p in MISSING_SHORT_PHRASES):
            return "missing_info"
        return best_intent


class MemoryAgent:
    SLOT_PATTERNS = {
        "preg_timing": ["hôm qua", "hôm kia", "tuần trước", "ngày"],
        "preg_lmp": ["kỳ kinh", "kinh cuối", "lmp"],
        "preg_ejaculation": ["xuất tinh", "ra ngoài", "trong âm đạo"],
        "preg_sex_contact": ["thâm nhập", "cọ xát", "quan hệ"],
        "sti_symptoms": ["mụn", "loét", "đau", "rát", "tiết dịch"],
    }

    def update(self, text: str, state: ChatState) -> None:
        for slot, pats in self.SLOT_PATTERNS.items():
            if slot not in state.memory and any(p in text for p in pats):
                state.memory[slot] = text


class SafetyAgent:
    def assess(self, text: str, intent: str) -> tuple[str, bool]:
        if intent == "emergency" or any(k in text for k in EMERGENCY_SIGNS):
            return "🔴 Cao", True
        if intent in {"sti_std", "contraception_pregnancy", "sexual_function"}:
            return "🟠 Trung bình", False
        return "🟢 Thấp", False


class MissingInfoAgent:
    SLOT_QUESTIONS = {
        "preg_sex_contact": "Bạn có quan hệ thâm nhập hay chỉ cọ xát bên ngoài?",
        "preg_ejaculation": "Có xuất tinh trong hoặc gần âm đạo không?",
        "preg_timing": "Sự việc xảy ra khi nào (ngày/tháng gần đúng)?",
        "preg_lmp": "Ngày đầu kỳ kinh gần nhất của bạn là khi nào?",
        "sti_symptoms": "Bạn có triệu chứng gì cụ thể (mụn, loét, đau rát, dịch bất thường...)?",
        "sti_duration": "Triệu chứng xuất hiện từ khi nào?",
        "sti_risk": "Gần đây có quan hệ nguy cơ không dùng bảo vệ không?",
    }

    def required_slots(self, intent: str) -> list[str]:
        if intent == "contraception_pregnancy":
            return ["preg_sex_contact", "preg_ejaculation", "preg_timing", "preg_lmp"]
        if intent == "sti_std":
            return ["sti_symptoms", "sti_duration", "sti_risk"]
        return []

    def ask_missing(self, intent: str, state: ChatState) -> list[str]:
        need = []
        for slot in self.required_slots(intent):
            if slot not in state.memory and slot not in state.asked_slots:
                state.asked_slots.add(slot)
                need.append(self.SLOT_QUESTIONS[slot])
        return need


class RetrievalAgent:
    def __init__(self):
        self.kb = load_knowledge_base()

    def find(self, text: str, intent: str) -> dict | None:
        pool = [x for x in self.kb if x.get("intent") in (None, intent)]
        best, best_score = None, 0.0
        for item in pool:
            score = max((jaccard_similarity(text, kw) for kw in item.get("keywords", [])), default=0.0)
            if score > best_score:
                best, best_score = item, score
        return best if best_score >= SIMILARITY_THRESHOLD else None


class DiagnosisAgent:
    def build(self, intent: str, kb_item: dict | None, emergency: bool) -> dict:
        if emergency:
            return {
                "nhan_dinh": "Bạn có dấu hiệu nguy cơ cao/cấp cứu cần xử trí sớm.",
                "chan_doan": "Chưa thể chẩn đoán online, ưu tiên loại trừ tình huống cấp cứu.",
                "xu_tri": "Đến cơ sở y tế gần nhất hoặc gọi cấp cứu ngay. Nếu có nguy cơ xâm hại tình dục, cần hỗ trợ y tế và pháp lý sớm.",
                "khi_nao_kham": "Đi khám ngay bây giờ.",
                "luu_y": "Không tự dùng thuốc kháng sinh/hormone hoặc trì hoãn thăm khám.",
            }
        base = kb_item["answer"] if kb_item else "Chưa đủ tri thức khớp hoàn toàn; cần thêm dữ kiện lâm sàng."
        return {
            "nhan_dinh": base,
            "chan_doan": "Chẩn đoán phù hợp nhất hiện tại dựa trên thông tin bạn đã cung cấp.",
            "xu_tri": "Theo dõi triệu chứng, quan hệ an toàn, không tự dùng thuốc kê đơn/hormone/kháng sinh.",
            "khi_nao_kham": "Đi khám nếu triệu chứng kéo dài, nặng dần, tái phát hoặc gây lo lắng nhiều.",
            "luu_y": "Thông tin tham khảo, không thay thế khám trực tiếp.",
        }


class ResponseAgent:
    def render(self, diagnosis: dict, risk: str, source: str | None = None) -> str:
        src = f"\nNguồn tham chiếu: {source}" if source else ""
        return (
            f"1. Nhận định tình trạng\n- {diagnosis['nhan_dinh']}"
            f"\n2. Chẩn đoán phù hợp nhất\n- {diagnosis['chan_doan']}"
            f"\n3. Mức độ nguy cơ\n- {risk}"
            f"\n4. Hướng xử trí cụ thể\n- {diagnosis['xu_tri']}"
            f"\n5. Khi nào cần đi khám\n- {diagnosis['khi_nao_kham']}"
            f"\n6. Lưu ý an toàn\n- {diagnosis['luu_y']}{src}"
        )


class AgenticOrchestrator:
    def __init__(self):
        self.intent = IntentAgent()
        self.memory = MemoryAgent()
        self.safety = SafetyAgent()
        self.missing = MissingInfoAgent()
        self.retrieval = RetrievalAgent()
        self.diagnosis = DiagnosisAgent()
        self.response = ResponseAgent()

    def reply(self, message: str, state: ChatState) -> str:
        text = normalize_text(message)
        if not text:
            return "Bạn hãy nhập câu hỏi để mình hỗ trợ nhé."
        intent = self.intent.detect(text)
        if intent == "greeting":
            return "Xin chào 👋 Mình hỗ trợ tư vấn sức khỏe giới tính, bạn cứ hỏi tự nhiên nhé."
        if intent == "ending":
            state.asked_slots.clear()
            state.memory.clear()
            state.last_risk = "Chưa đánh giá"
            return "Mình đã kết thúc ca tư vấn hiện tại. Khi cần bạn quay lại nhé."
        if intent == "out_of_scope":
            return OUT_OF_SCOPE_REPLY

        self.memory.update(text, state)
        risk, emergency = self.safety.assess(text, intent)
        state.last_risk = risk
        missing_questions = self.missing.ask_missing(intent, state)
        if missing_questions:
            return "Mình chưa đủ dữ kiện để kết luận. " + " ".join(missing_questions)

        kb_item = self.retrieval.find(text, intent)
        diagnosis = self.diagnosis.build(intent, kb_item, emergency)
        source = f"{kb_item.get('source')} ({kb_item.get('last_updated')})" if kb_item else None
        return self.response.render(diagnosis, risk, source)


ORCHESTRATOR = AgenticOrchestrator()


def detect_intent(text: str) -> str:
    return ORCHESTRATOR.intent.detect(normalize_text(text))


def response_for_intent(intent: str, text: str, state: ChatState) -> str:
    # compatibility wrapper for tests
    if intent == "greeting":
        return "Xin chào 👋 Mình hỗ trợ tư vấn sức khỏe giới tính, bạn cứ hỏi tự nhiên nhé."
    if intent == "ending":
        state.asked_slots.clear(); state.memory.clear(); state.last_risk = "Chưa đánh giá"
        return "Mình đã kết thúc ca tư vấn hiện tại. Khi cần bạn quay lại nhé."
    if intent == "out_of_scope":
        return OUT_OF_SCOPE_REPLY

    # keep memory/risk behavior aligned with main orchestrator flow
    ORCHESTRATOR.memory.update(text, state)
    risk, emergency = ORCHESTRATOR.safety.assess(text, intent)
    state.last_risk = risk
    missing_questions = ORCHESTRATOR.missing.ask_missing(intent, state)
    if missing_questions:
        return "Mình chưa đủ dữ kiện để kết luận. " + " ".join(missing_questions)
    kb_item = ORCHESTRATOR.retrieval.find(text, intent)
    diagnosis = ORCHESTRATOR.diagnosis.build(intent, kb_item, emergency)
    source = f"{kb_item.get('source')} ({kb_item.get('last_updated')})" if kb_item else None
    return ORCHESTRATOR.response.render(diagnosis, risk, source)


def chatbot(message: str, history: list, state: ChatState):
    reply = ORCHESTRATOR.reply(message, state)
    return reply, state, state.last_risk


custom_css = """
:root {--bg:#0b1220;--surface:#111827;--text:#e5e7eb;--accent:#22c55e;}
.gradio-container {background: linear-gradient(135deg, #0b1220 0%, #111827 100%) !important;}
footer {visibility:hidden}
#title-banner {padding:14px;border-radius:14px;background:#0f172a;color:var(--text);border:1px solid #1f2937;}
"""

if gr is not None:
    with gr.Blocks(css=custom_css, theme=gr.themes.Soft(primary_hue="green")) as demo:
        gr.Markdown(f"## {APP_NAME}", elem_id="title-banner")
        state = gr.State(ChatState())
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Sidebar mức độ nguy cơ")
                gr.Markdown("- 🔴 Cao: dấu hiệu cấp cứu/nguy cơ cao\n- 🟠 Trung bình: cần theo dõi sát và có thể cần khám\n- 🟢 Thấp: tư vấn theo dõi tại nhà")
                risk_box = gr.Textbox(value="Chưa đánh giá", label="Mức nguy cơ hiện tại", interactive=False)
            with gr.Column(scale=4):
                gr.ChatInterface(
                    fn=chatbot,
                    additional_inputs=[state],
                    additional_outputs=[state, risk_box],
                    textbox=gr.Textbox(placeholder="Nhập câu hỏi của bạn...", lines=2),
                )

if __name__ == "__main__" and gr is not None:
    demo.launch()
