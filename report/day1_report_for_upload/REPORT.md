# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy: 06/11/2026**

**Runtime Colab: 4T CPU** CPU/GPU

**Python / PyTorch / Ultralytics:Python 3**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=… ; `class_name`=… ; `rank`=1 ; `score`=… ; `taxonomy_name`=…
{
    "taxonomy_name": "ImageNet-1K",
    "rank": 1,
    "class_id": 468,
    "class_name": "cab",
    "score": 0.510915
}
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
cả bức ảnh
Vì trường task có giá trị image_classification, nghĩa là mô hình phân loại toàn bộ hình ảnh.
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
do taxonomy/checkpoint
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
dùng cả ba sẽ tránh trùng tên lớp, giúp dễ xử lý dữ bằng máy và đảm bảo khả năng đối chiếu
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
Guideline phải thống nhất quy tắc chọn một lớp duy nhất dựa trên chủ thể chính, mức độ nổi bật và ngữ cảnh tổng thể, tránh việc mỗi người gán nhãn chọn một đối tượng khác nhau cho cùng một ảnh.
6. Vì sao không được lấy `score` của model làm ground truth?
vì score chỉ thể hiện mức độ tin tưởng của mô hình đối với dự đoán, không đảm bảo dự đoán đó đúng.
## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=… ; `score`=… ; `bbox_xyxy`=[…, …, …, …] ; `bbox_width`=… ; `bbox_height`=…
{
  "sample_id": "kitchen",
  "class_name": "person",
  "score": 0.912625,
  "bbox_xyxy": [
    385.33,
    69.24,
    498.92,
    348.92
  ],
  "bbox_width": 113.58,
  "bbox_height": 279.68
}
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
    Box trên xác định vị trí người đứng ở giữa, hơi lệch về bên phải bức ảnh.
    - Góc trên-trái: X = 385.33 px, Y = 69.24 px.
    - Góc dưới-phải: X = 498.92 px, Y = 348.92 px.
    - Chiều rộng: 113.58 px.
    - Chiều cao: 279.68 px.
    Hiểu đơn giản: Hộp chữ nhật bắt đầu cách mép trái ảnh khoảng 385 px, cách mép trên khoảng 69 px, bao quanh người và kết thúc tại tọa độ (498.92, 348.92).
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
 Trong file JSON được cung cấp, có 11 prediction của ảnh kitchen, tất cả được xuất với score_threshold = 0.35.
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
    Ngưỡng càng thấp thì reviewer cần:
        Kiểm tra từng bounding box có thực sự bao quanh vật thể không.
        Kiểm tra nhãn lớp có chính xác không.
        Phát hiện và loại bỏ các box dự đoán sai (false positive).
        Kiểm tra các box trùng lặp hoặc chồng lấn không hợp lý.
        Điều chỉnh lại box nếu vị trí hoặc kích thước chưa chính xác.
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
    Người gán nhãn phải vẽ bounding box hình chữ nhật sát nhất có thể với đường bao của vật thể, bao phủ toàn bộ phần vật thể nhìn thấy được, hạn chế lấy quá nhiều vùng nền xung quanh và không bỏ sót phần vật thể có thể quan sát được.
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
    Mấy cái lò nướng (oven) ở mép ảnh bị cắt mất một nửa. Mình sẽ dùng escalation (hỏi Lab Coach) vì vật bị cắt mép thế này rất dễ gây tranh cãi, không rõ là nên vẽ hay bỏ qua.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=kitchen-001 ; `class_name`=person ; `score`=0.90 ; số điểm polygon=`polygon_point_count`= rất nhiều ; 3–5 điểm đầu của `polygon_xy`=[[496.0, 331.0], [496.0, 304.0], [495.0, 303.0], [495.0, 301.0], [494.0, 300.0]]
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
   Polygon vẽ ra đường viền chính xác theo hình dáng thật của vật thể, giúp tách vật đó ra khỏi nền xung quanh tốt hơn so với cái hộp chữ nhật.
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
   `instance_id` dùng để phân biệt các vật thể khác nhau dù chúng cùng một class (ví dụ có 2 người thì có 2 ID khác nhau). Nó không phải class ID (vì class ID chỉ nói đó là người) và không phải tracking ID (vì tracking dùng để theo dõi vật thể di chuyển qua các frame video).
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
   Vẽ đường viền men theo sát các pixel của vật thể, không để lấn sang vật thể bên cạnh hoặc lấn sang background.
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
   -> Mấy cái bát (bowl) nằm sát nhau trên bàn làm đường viền mask bị dính vào nhau. Mình dùng escalation vì ranh giới chạm nhau rất khó vẽ đúng, dễ bị thừa thiếu pixel.


## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một lớp cho toàn bộ ảnh (VD: cab) | Ảnh traffic có nhiều loại xe, dễ đoán sai nếu không có guideline rõ ràng. | Xác định vật thể chính của bức ảnh và gán nhãn. | Xem nhãn đó có đúng là tâm điểm của ảnh không. |
| Phát hiện vật thể | Hộp (bbox) kèm nhãn class. | Ở ngưỡng 0.20, model bắt được cả cái cốc, cái thìa (rất nhỏ), dễ bị nhầm là rác. | Vẽ hộp bao sát vật thể. | Kiểm tra xem có bỏ sót vật thể nào không, hộp vẽ có bị thừa/thiếu không. |
| Instance segmentation | Đa giác (polygon) bao quanh vật thể. | Các vật thể chạm nhau (như mấy cái bát) làm đường viền mask bị dính vào nhau. | Vẽ mask men theo viền vật thể. | Kiểm tra độ chính xác của các điểm ảnh trên đường viền. |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
    Chỉ dùng ảnh COCO công khai có sẵn trong notebook; tuyệt đối không tải ảnh cá nhân hay dữ liệu nội bộ lên Colab/GitHub.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
    Mình sẽ dừng thao tác lại và báo ngay cho Lab Coach hoặc giảng viên phụ trách để xin hướng dẫn.
## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [x] Đã nộp **link repository** trên LMS
