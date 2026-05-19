# Al-ve-suc-khoe-gioi-

Chatbot Gradio hỗ trợ trả lời câu hỏi thường gặp về sức khỏe giới tính.

## Điểm đã hoàn thiện
- Tách mã từ notebook sang `app.py` để dễ bảo trì.
- Cải thiện giao diện: theme mềm, banner rõ ràng, nền gradient.
- Sửa lỗi trạng thái hội thoại: mỗi phiên có bộ nhớ riêng (`gr.State`) thay vì dùng biến global.
- Tăng tính ổn định xử lý câu hỏi lặp và chuẩn hoá văn bản tiếng Việt.
- Bổ sung kiến trúc **agentic** theo pipeline nhiều tác tử chuyên trách.
- Mở rộng nhận diện tâm lý/cảm xúc, tình huống nguy hiểm và câu hỏi ngoài chuyên môn.

## Kiến trúc Agentic Agent của hệ thống
Hệ thống xử lý theo pipeline:

1. **Scope Guard Agent**: chặn câu hỏi ngoài phạm vi sức khỏe giới tính/sinh sản.
2. **Planner Agent**: phân loại nhóm chủ đề, bao gồm nhóm tâm lý & cảm xúc.
3. **Retriever Agent**: TF-IDF retrieval trả lời từ kho tri thức.
4. **Safety Agent**: phát hiện tình huống nguy hiểm và phân mức rủi ro:
   - 🟢 nhẹ
   - 🟡 trung bình
   - 🟠 cần khám sớm
   - 🔴 nguy hiểm
5. **Psychological Support Agent**: phản hồi đồng cảm, trấn an và hướng dẫn an toàn tâm lý.
6. **Responder Agent**: tổng hợp trả lời, thêm câu hỏi follow-up trọng tâm khi thiếu dữ kiện.

## Các cải tiến mới
- Tăng vốn từ tiếng Việt đời thường/slang cho các nhóm:
  - lo lắng mang thai, áp lực quan hệ, tự ti cơ thể, bị ép quan hệ.
  - nghi ngờ STI/HIV, rách bao cao su, sốt/chảy máu/đau dữ dội sau quan hệ.
- Nhận diện phủ định triệu chứng cơ bản (`không đau`, `không bị ngứa`, `chưa quan hệ`) để giảm kết luận sai.
- Khi gặp câu ngoài chuyên môn (game, thời tiết, công nghệ, tiền bạc, lập trình...), bot từ chối lịch sự và mời quay lại đúng lĩnh vực.

## Chạy ứng dụng
```bash
pip install gradio scikit-learn pandas
python app.py
```
