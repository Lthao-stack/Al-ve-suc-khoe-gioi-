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


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower().strip()
    return re.sub(r"\s+", " ", text)


df = pd.DataFrame(knowledge_base)
documents = (df["topic"] + " " + df["answer"]).tolist()
vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2))
tfidf_matrix = vectorizer.fit_transform(documents)

GREETING_PATTERNS = [r"^(xin chào|chào|hello|hi|hey|bot ơi)[\s!,.]*$"]
END_KEYWORDS = ["tạm biệt", "bye", "kết thúc", "dừng", "hết rồi", "không hỏi nữa", "xong rồi", "cảm ơn"]
ADVICE_KEYWORDS = ["nên làm gì", "làm sao", "làm thế nào", "có sao không", "nguy hiểm không", "có cần đi khám không", "tư vấn"]
WARNING_KEYWORDS = ["đau dữ dội", "chảy máu nhiều", "dịch lạ", "mùi hôi", "bị ép", "xâm hại", "lạm dụng"]


def classify_topic(text: str) -> str:
    if any(k in text for k in ["gay", "lesbian", "bisexual", "lgbt", "come out", "đồng tính", "song tính", "chuyển giới"]):
        return "LGBTQ+ và bản dạng giới"
    if any(k in text for k in ["dậy thì", "vỡ giọng", "mọc lông", "mộng tinh", "thủ dâm"]):
        return "Tuổi dậy thì và phát triển cơ thể"
    if any(k in text for k in ["bao cao su", "tránh thai", "thuốc tránh thai", "có thai", "que thử thai"]):
        return "Tình dục an toàn và tránh thai"
    if any(k in text for k in ["sti", "bệnh lây", "ngứa", "rát", "dịch", "mùi hôi", "viêm"]):
        return "Sức khỏe sinh sản và bệnh lý"
    return "Sức khỏe giới tính chung"


def retrieve_answer(question: str):
    user_vec = vectorizer.transform([question])
    scores = cosine_similarity(user_vec, tfidf_matrix).flatten()
    idx = int(scores.argmax())
    if scores[idx] < 0.1:
        return None
    return df.iloc[idx]["answer"]


def generate_advice(text: str) -> str:
    if not any(k in text for k in ADVICE_KEYWORDS):
        return ""
    advice = ["- Theo dõi triệu chứng và mức độ ảnh hưởng trong sinh hoạt hằng ngày.", "- Tránh tự điều trị theo nguồn không rõ ràng.", "- Nếu lo lắng kéo dài, nên gặp nhân viên y tế để được tư vấn trực tiếp."]
    return "\n\n**Lời khuyên:**\n" + "\n".join(advice)


def generate_warning(text: str) -> str:
    if any(k in text for k in WARNING_KEYWORDS):
        return "\n\n**Cảnh báo:** Có dấu hiệu cần được hỗ trợ sớm. Nếu tình trạng nặng hơn hoặc liên quan ép buộc/xâm hại, hãy tìm trợ giúp ngay từ cơ sở y tế hoặc người đáng tin cậy."
    return ""


@dataclass
class ChatState:
    asked_questions: list[str] = field(default_factory=list)

    def is_repeat(self, message: str) -> bool:
        if not self.asked_questions:
            return False
        texts = self.asked_questions + [message]
        temp = vectorizer.transform(texts)
        scores = cosine_similarity(temp[-1], temp[:-1]).flatten()
        return bool(scores.max() > 0.9)


def chatbot(message: str, history: list, state: ChatState):
    text = normalize_text(message or "")
    if not text:
        return "Bạn hãy nhập câu hỏi để mình hỗ trợ nhé.", state

    if any(re.match(p, text) for p in GREETING_PATTERNS):
        return "Xin chào 👋 Mình có thể hỗ trợ các câu hỏi thường gặp về sức khỏe giới tính.", state

    if any(k in text for k in END_KEYWORDS):
        state.asked_questions.clear()
        return "Mình đã kết thúc cuộc trò chuyện và làm mới ngữ cảnh. Cảm ơn bạn!", state

    if state.is_repeat(text):
        return "Bạn đã hỏi nội dung tương tự trước đó. Hãy thử bổ sung thêm chi tiết để mình hỗ trợ chính xác hơn nhé.", state

    state.asked_questions.append(text)
    answer = retrieve_answer(text)
    if not answer:
        return "Mình chưa có dữ liệu phù hợp trong bộ tri thức hiện tại. Bạn có thể mô tả cụ thể hơn không?", state

    topic = classify_topic(text)
    response = f"### Nhóm nội dung: {topic}\n\n{answer}"
    response += generate_advice(text)
    response += generate_warning(text)
    response += "\n\n> Lưu ý: Thông tin chỉ mang tính tham khảo, không thay thế tư vấn y tế chuyên môn."
    return response, state


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
        examples=["Dậy thì bình thường bắt đầu lúc mấy tuổi?", "Dấu hiệu STI thường gặp là gì?", "Nếu bị ép buộc quan hệ thì nên làm gì?"],
    )

if __name__ == "__main__":
    demo.launch()
