# Al-ve-suc-khoe-gioi-

Chatbot Gradio hỗ trợ trả lời câu hỏi thường gặp về sức khỏe giới tính.

## Điểm đã hoàn thiện
- Tách mã từ notebook sang `app.py` để dễ bảo trì.
- Cải thiện giao diện: theme mềm, banner rõ ràng, nền gradient.
- Sửa lỗi trạng thái hội thoại: mỗi phiên có bộ nhớ riêng (`gr.State`) thay vì dùng biến global.
- Tăng tính ổn định xử lý câu hỏi lặp và chuẩn hoá văn bản tiếng Việt.

## Chạy ứng dụng
```bash
pip install gradio scikit-learn pandas
python app.py
```
