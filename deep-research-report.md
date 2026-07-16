# Nghiên cứu sâu về loại bỏ keyframe trùng lặp bằng embedding cho bài toán keyframe retrieval từ prompt

## Bối cảnh và kết luận nhanh

Từ mô tả dự án của bạn, đây không phải là bài toán “xóa ảnh giống nhau” theo nghĩa đơn giản, mà là bài toán **giảm dư thừa khung hình** trong một pipeline retrieval đa mô-đun gồm prompt-to-keyframe, track, shot, OCR, ASR và temporal search. Vì vậy, thuật toán tốt phải đồng thời làm được ba việc: giảm số keyframe để index và search nhanh hơn, giữ lại các frame đại diện cho từng cụm nội dung, và tránh xóa nhầm các frame tưởng giống nhau nhưng lại khác ở chi tiết quan trọng cho truy vấn như chữ trên màn hình, trạng thái vật thể, hay biến đổi ngắn trong cùng một shot. Các công trình gần đây về long-video understanding và text-video retrieval đều cho thấy temporal redundancy là nút thắt lớn, trong khi các chiến lược chỉ dựa trên sampling đều có thể bỏ sót frame quan trọng nếu không có cơ chế chọn representative frame hoặc query-aware reranking. citeturn14view1turn14view0turn6view6turn18view0

Nếu mục tiêu hiện tại của bạn là **code xử lý loại bỏ keyframe trùng lặp dựa trên embedding của keyframe**, thì khuyến nghị thực tế nhất không phải là chọn một thuật toán duy nhất, mà là dùng **pipeline hai tầng**. Tầng một dùng embedding similarity để loại bỏ redundancy “rõ ràng” trong phạm vi thời gian gần hoặc trong cùng track/shot; tầng hai dùng clustering hoặc graph grouping để gom các near-duplicate trên phạm vi rộng hơn, sau đó chọn một representative frame bằng **medoid** hoặc một tiêu chí query-aware. Với hệ retrieval từ prompt, first-pass embedding nên thuộc cùng họ với backbone retrieval, chẳng hạn CLIP hoặc SigLIP 2, vì đây là các mô hình được huấn luyện trong shared image-text space với cosine similarity và có năng lực tốt cho image-text retrieval. Nếu bài toán của bạn cần **strict dedup** ở mức “copy hoặc gần copy”, hãy thêm một nhánh xác minh bằng descriptor chuyên cho copy detection như SSCD hoặc xác minh cục bộ bằng OCR/SSIM cho các cặp borderline. citeturn7view1turn6view1turn16view0turn12view0

Nói ngắn gọn, với task của bạn, tổ hợp đáng làm nhất là: **normalize embedding → shot/track-aware local dedup → ANN kNN graph → connected components hoặc HDBSCAN → chọn medoid → OCR/SSIM hoặc SSCD verification cho các cụm khó**. Cách này bám rất sát những gì các nghiên cứu mới đang làm để giảm temporal redundancy mà vẫn giữ độ đại diện và độ đa dạng của frame. citeturn7view5turn17search0turn6view4turn14view0turn7view4turn12view0

## Phân rã đúng bài toán duplicate trong hệ của bạn

Điểm quan trọng nhất là phải tách rõ ba loại “trùng lặp”, vì mỗi loại cần một cách xử lý khác nhau. Loại thứ nhất là **exact hoặc near-exact duplicate**, ví dụ cùng một cảnh gần như không đổi, khác nhẹ do nén, resize, hoặc thay đổi màu nhỏ. Loại thứ hai là **temporal redundancy**, tức nhiều frame liên tiếp trong cùng track hoặc cùng shot có nội dung gần như giống nhau nhưng không hoàn toàn là copy. Loại thứ ba là **semantic overlap nhưng không phải duplicate thật**, ví dụ hai frame đều chứa “người cầm điện thoại”, nhưng một frame có subtitle, một frame có giao diện khác, hoặc một frame thể hiện trạng thái trước/sau của hành động. Các nghiên cứu gần đây nhấn mạnh rằng frame selection nếu quá dựa vào global semantic similarity sẽ dễ rơi vào “semantic traps”, tức chọn những frame có vẻ đúng theo từ khóa nhưng bỏ mất ngữ cảnh và các chi tiết cục bộ quan trọng. Đồng thời, work về visually identical image detection cho thấy global descriptor hoặc cosine threshold có thể bỏ qua các khác biệt nhỏ nhưng rất quan trọng về text, badge, title, hay overlay. citeturn14view0turn11view2turn12view0

Điều này dẫn tới một hệ quả thực tế: **embedding dedup không nên được dùng như một bộ lọc “xóa sạch” duy nhất**. CLIP, chẳng hạn, dùng L2-normalized image/text embeddings và tối ưu scaled pairwise cosine similarity trong một không gian chung ảnh–văn bản, nên rất phù hợp cho retrieval theo prompt. Nhưng chính vì nó ưu tiên semantic alignment, nếu bạn đặt threshold quá gắt thì nó có thể gộp những frame khác nhau về chi tiết thị giác nhưng giống nhau về nghĩa. SigLIP 2 tiếp tục cải thiện image-text retrieval và transfer performance, nên thích hợp làm backbone retrieval mạnh hơn, nhưng bản chất shared embedding vẫn khiến nó mạnh về semantic alignment hơn là strict copy verification. citeturn7view1turn7view2turn6view1

Ở chiều ngược lại, nếu mục tiêu chính là “các frame này có thực sự là copy/near-copy không”, descriptor chuyên cho copy detection lại phù hợp hơn. SSCD được thiết kế riêng cho image copy detection, dùng compact descriptor cho web-scale search, nhấn mạnh bài toán hard match/non-match threshold, và cho thấy entropy regularization giúp khoảng cách giữa descriptor nhất quán hơn giữa các vùng embedding khác nhau. Nói cách khác, nếu bạn muốn một **ngưỡng toàn cục** ổn định hơn để nói “trùng / không trùng”, SSCD là một lựa chọn đáng cân nhắc bổ sung bên cạnh CLIP/SigLIP. citeturn16view0

Với pipeline keyframe retrieval từ prompt, mình sẽ phân vai như sau: **CLIP hoặc SigLIP 2 cho retrieval và coarse semantic dedup; SSCD hoặc OCR/SSIM verification cho strict duplicate checking; temporal logic theo shot/track để tránh xóa nhầm frame ngắn nhưng quan trọng**. Đây là cách cân bằng tốt nhất giữa retrieval relevance và dedup precision. citeturn6view1turn16view0turn12view0turn6view5

## Các nghiên cứu học thuật nổi bật về Keyframe Selection và Redundancy Reduction

Trong các nghiên cứu gần đây (giai đoạn 2023 - 2025) phục vụ cho Video-LLM và Video Retrieval, việc loại bỏ dư thừa temporal (temporal redundancy) và semantic redundancy đang là chủ đề được quan tâm lớn. Dưới đây là các phương pháp tiêu biểu từ các paper nổi bật:

### 1. RETAKE (DPSelect & PivotKV)
* **Ý tưởng cốt lõi**: Giảm thiểu đồng thời cả dư thừa thị giác cấp thấp (low-level temporal redundancy) và dư thừa tri thức cấp cao (high-level knowledge redundancy).
* **Cơ chế DPSelect (Dist-Peak Select)**: Thay vì dùng uniform sampling hoặc ngưỡng similarity cố định, DPSelect tính toán cosine dissimilarity (khoảng cách cosine ngược) giữa các frame liền kề và tìm các **local maximum peaks** (điểm cực đại địa phương). Các đỉnh này tương ứng với thời điểm có sự thay đổi đột ngột về hành động hoặc bối cảnh thị giác.
* **Ý nghĩa**: Rất thích hợp làm bộ lọc cục bộ (local dedup) để giữ lại các frame chứa biến đổi động thực sự trong shot mà không bị mất chi tiết quan trọng.

### 2. KeyVideoLLM (Text-Video Semantic Similarity)
* **Ý tưởng cốt lõi**: Tận dụng trực tiếp sự tương quan giữa text query và video frames thông qua không gian embedding chung của CLIP.
* **Cơ chế**: Tính toán cosine similarity giữa visual embedding của các frame và text embedding của truy vấn (hoặc câu hỏi/câu trả lời giả định). Chỉ các frame có độ tương quan cao mới được giữ lại làm keyframe đại diện. Phương pháp này đạt tỉ lệ nén lên đến 60.9x mà vẫn duy trì/cải thiện chất lượng bài toán QA.
* **Ý nghĩa**: Phù hợp cho pha "Query-aware frame selection" ở tầng re-ranking/retrieval cuối cùng.

### 3. MaxInfo (Diversity-Aware Selection)
* **Ý tưởng cốt lõi**: Khắc phục nhược điểm của Top-K (dễ chọn phải các frame giống nhau nhưng đều có điểm tương đồng cao với query).
* **Cơ chế**: Sử dụng SVD (Singular Value Decomposition) trên ma trận feature của CLIP để phân tích thông tin bao phủ hình học (geometric volume maximization). Mục tiêu là chọn ra tập frame có tổng thể tích không gian embedding lớn nhất, qua đó tối đa hóa độ đa dạng (diversity) của nội dung video.
* **Ý nghĩa**: Phù hợp khi cần summarize hoặc rút gọn video thành một số lượng keyframe cố định nhưng phải đại diện được toàn bộ nội dung video (global representation).

### 4. LMSKE (Large Model based Sequential Keyframe Extraction)
* **Ý tưởng cốt lõi**: Kết hợp mô hình phân cảnh (Shot Boundary Detection) và phân cụm thích ứng (Adaptive Clustering).
* **Cơ chế**: Dùng TransNetV2 để chia shot thô. Trích xuất CLIP embedding cho các frame trong shot, sau đó áp dụng phân cụm thích ứng (adaptive clustering) cho từng shot độc lập để loại bỏ redundant frames. Điều này giúp giữ cấu trúc tuần tự thời gian nguyên bản của video.

### 5. FOCUS (Frame-Optimistic Confidence Upper-bound Selection)
* **Ý tưởng cốt lõi**: Tiếp cận việc chọn keyframe dưới dạng bài toán Combinatorial Exploration.
* **Cơ chế**: Sử dụng thuật toán Multi-armed Bandit để xác định các vùng thời gian (temporal regions) có giá trị cao trước khi đi vào chọn các frame cụ thể, tối ưu hóa giữa việc khám phá nội dung mới (exploration) và khai thác nội dung liên quan (exploitation).

## Các họ thuật toán phù hợp nhất

### So khớp ngưỡng trực tiếp trên cosine similarity

Đây là cách đơn giản nhất: chuẩn hóa embedding, tính cosine similarity, rồi xóa hoặc gộp những frame vượt ngưỡng. Vì CLIP vốn sử dụng normalized embeddings và cosine similarity trong huấn luyện, cách này rất tự nhiên khi bạn đang dùng CLIP-like backbone. Nó đặc biệt hiệu quả cho **local temporal dedup**, tức các frame ở gần nhau trong cùng track hoặc shot. Các công trình như RETAKE cũng cho thấy temporal redundancy có thể được khai thác trực tiếp từ vision encoder features; thay vì chọn top khoảng cách lớn nhất toàn cục, họ dùng local peaks để giữ frame có thay đổi thực sự đáng kể. citeturn7view1turn14view1

Ưu điểm của phương án này là dễ code, chạy nhanh, dễ debug, và rất phù hợp làm lớp lọc đầu tiên. Nhược điểm là threshold cố định khó ổn định trên toàn bộ dữ liệu, đặc biệt khi video có nhiều miền nội dung khác nhau hoặc khi có text overlay, UI, subtitle, watermark. Chính vì vậy, threshold cosine chỉ nên là **tầng một**, không nên là toàn bộ giải pháp. citeturn16view0turn12view0

### Đồ thị tương đồng và connected components

Một bước tiến thực dụng hơn là xây **kNN graph** trên embedding: mỗi frame nối với k láng giềng gần nhất, sau đó chỉ giữ các cạnh vượt ngưỡng similarity và lấy connected components làm cụm duplicate/near-duplicate. Đây là một suy luận triển khai rất hợp lý từ các thư viện/vector-index như Faiss và HNSW, vốn được xây dựng cho vector similarity search ở quy mô lớn. Faiss cung cấp các primitive để search, cluster, compress và transform dense vectors; HNSW là một graph-based ANN index có khả năng scaling rất mạnh và được xây dựng tăng dần theo cấu trúc nhiều lớp. citeturn7view5turn17search0turn17search1turn17search6

Với bài toán của bạn, graph-based grouping thường tốt hơn pairwise thresholding thuần túy vì duplicate hiếm khi đi theo quan hệ “một-một”. Thực tế thường là một cụm gồm nhiều frame gần nhau: frame A giống B, B giống C, dù A có thể không vượt ngưỡng trực tiếp với C. Connected components xử lý kiểu lan truyền này rất gọn. Nếu dữ liệu tăng liên tục, HNSW phù hợp cho online-ish search; nếu bạn chạy batch offline lớn và có GPU, Faiss thường tiện hơn. citeturn17search0turn17search1turn7view5

### Clustering mật độ: DBSCAN, HDBSCAN, OPTICS

Khi không muốn chốt trước số cụm, density clustering là họ thuật toán rất hợp với dedup. DBSCAN được thiết kế để phát hiện cụm có hình dạng bất kỳ, xử lý noise tốt và hoạt động hiệu quả trên cơ sở dữ liệu lớn. Điểm yếu của DBSCAN là chọn epsilon khó khi mật độ thay đổi. HDBSCAN được phát triển để giảm khó khăn này, xử lý tốt hơn dữ liệu có mật độ biến thiên và yêu cầu một bộ tham số trực quan, tương đối robust hơn. OPTICS cũng cùng tinh thần: không ép phải chốt một ngưỡng mật độ duy nhất ngay từ đầu mà tạo ra ordering phản ánh cấu trúc mật độ ở nhiều mức. citeturn6view3turn6view4turn3search9

Trong thực tế dự án của bạn, **HDBSCAN thường là điểm cân bằng tốt nhất** nếu corpus chưa quá khổng lồ và bạn cần chống noise. Nó đặc biệt hợp khi embedding của bạn đến từ nhiều loại video, nhiều domain, hoặc mỗi track/shot có mật độ điểm khác nhau. DBSCAN phù hợp hơn khi dữ liệu đồng nhất và bạn đoán được ngưỡng khá ổn. OPTICS hữu ích khi bạn muốn phân tích dữ liệu trước để hiểu ngưỡng thay vì commit ngay. citeturn6view4turn3search9

### Chọn representative frame bằng medoid hoặc maximum-volume

Sau khi có cụm, bạn vẫn cần giữ lại một frame đại diện. Với dedup, **medoid** thường hợp hơn centroid vì medoid là một điểm thật trong dữ liệu, không phải vector trung bình. Các tiếp cận hiện đại trong video và re-localization vẫn dùng medoid/keyframe đại diện vì nó phù hợp với bước matching downstream. Một hướng mới hơn là **maximum-volume / MaxInfo**, chọn tập frame sao cho span của embedding space được phủ tốt nhất, giúp giữ cả tính đại diện lẫn đa dạng. MaxInfo cho thấy tối đa hóa geometric volume của các embedding được chọn giúp loại redundancy nhưng vẫn giữ coverage tốt của nội dung video. citeturn10search7turn10search1turn7view4

Nếu task của bạn chỉ cần **một frame đại diện cho một cụm duplicate**, dùng medoid là đủ, dễ cài và ổn định. Nếu bạn cần giữ **một tập nhỏ frame đa dạng** trong mỗi shot hoặc mỗi video để phục vụ retrieval/LLM/video understanding, MaxInfo đáng nghiên cứu hơn. Với retrieval từ prompt, một biến thể rất hay là **query-aware medoid**: chọn frame trong cụm có tổng khoảng cách tới cụm nhỏ nhất nhưng đồng thời có similarity cao với prompt embedding. Ý tưởng này phù hợp với xu hướng mới trong text-guided frame selection. citeturn7view4turn11view3turn7view3

### Phương án lai để tránh xóa nhầm các frame khác nhau ở chi tiết nhỏ

Đây là phần rất quan trọng nếu video của bạn có UI, subtitle, OCR text, scoreboard, caption hoặc biển báo. Nghiên cứu gần đây về visually identical image detection chỉ ra rằng global embeddings hoặc hash toàn cục có thể mất nhạy với khác biệt cục bộ nhưng có ý nghĩa, như title khác ngôn ngữ, badge xếp hạng, subtitle, hoặc vị trí text thay đổi. Phương án hiệu quả là **lọc thô bằng cosine distance trên embedding**, rồi xác minh lại các cặp quá gần nhau bằng **OCR-aware SSIM** hoặc structural comparison cục bộ. citeturn12view0

Nói cách khác, nếu retrieval system của bạn có track OCR riêng thì đừng để module dedup làm mất thông tin OCR. Hãy coi OCR như một “bộ phanh” trong khâu dedup: hai frame rất giống nhau về embedding nhưng khác text thì không nên gộp ngay. Đây là một trong những chỗ mà pipeline đa mô-đun của dự án chính là lợi thế kiến trúc của bạn. citeturn12view0

## Kiến trúc triển khai khuyến nghị cho task hiện tại

Pipeline mình khuyên dùng cho codebase hiện tại là một kiến trúc bốn bước.

Bước đầu tiên là **tách theo temporal context**. Nếu hệ đã có shot boundary hoặc track, hãy dedup ưu tiên **trong cùng track hoặc cùng shot**, rồi mới xét cross-shot. Các công trình như TAC-SUM và InfoShot đều cho thấy temporal/shot structure là tín hiệu rất mạnh để tránh làm mất các deviation ngắn nhưng quan trọng; trong video understanding, chiến lược shot-aware giữ được cả frame đại diện lẫn frame “bất thường” trong cùng shot tốt hơn so với sampling phẳng toàn video. citeturn6view5turn13search2

Bước thứ hai là **local dedup bằng cosine threshold trong một cửa sổ thời gian ngắn**. Ở bước này, bạn không cần ANN hay clustering phức tạp. Chỉ cần normalize embedding và so với các frame lân cận trong cùng track/shot. Mục tiêu là loại đi các frame liên tiếp gần như giống hệt nhau. Hướng của RETAKE, vốn chọn local maximum peak distance dựa trên visual features để phát hiện frame đáng giữ, gợi ý rất rõ rằng temporal locality nên được khai thác trước khi làm global grouping. citeturn14view1

Bước thứ ba là **global grouping bằng ANN + graph hoặc HDBSCAN**. Sau local pass, hãy index toàn bộ embedding còn lại bằng Faiss hoặc HNSW, truy vấn top-k láng giềng gần nhất cho từng frame, rồi hoặc dựng similarity graph để lấy connected components, hoặc chạy HDBSCAN trên embedding/KNN distances. Đây là chỗ hệ thống bắt đầu gom các duplicate không nằm kề nhau về thời gian, chẳng hạn cùng một background, cùng một object crop, hoặc cùng cảnh được lặp lại ở nhiều thời điểm. Với tập dữ liệu lớn, Faiss và HNSW là hai xương sống hợp lý nhất cho giai đoạn này. citeturn7view5turn17search0turn17search1turn6view4

Bước cuối cùng là **chọn representative frame và xác minh biên**. Chọn medoid nếu bạn muốn frame ổn định, thật, ít rủi ro. Nếu bạn có prompt của người dùng ngay tại thời điểm suy diễn, hãy chấm thêm điểm relevance giữa frame và prompt bằng CLIP/SigLIP để giữ frame vừa đại diện cụm vừa hữu ích cho retrieval. Đây là hướng nhất quán với KeyVideoLLM, TiFRe và ProCLIP, vốn đều dựa vào text-video similarity hoặc prompt-aware sampling để giữ những frame liên quan tới truy vấn mà vẫn giảm số frame đáng kể. Với các cụm mà khoảng cách quá sát ngưỡng hoặc có nhiều text/UI, hãy chạy bước verification bằng OCR/SSIM hoặc descriptor chuyên copy detection như SSCD. citeturn11view3turn7view3turn18view0turn18view3turn16view0turn12view0

Một pseudo-pipeline rất phù hợp với task này có thể tóm tắt như sau:

```python
# E: normalized keyframe embeddings
# meta: shot_id, track_id, timestamp, optional OCR text
# q: optional prompt embedding

for each group in group_by(track_id or shot_id):
    local_keep = temporal_dedup(group, sim_threshold=tau_local, window=w)

global_index = build_ann_index(local_keep.embeddings)  # Faiss/HNSW
edges = query_knn_edges(global_index, k=k)

# keep only strong similarity edges, optionally within same shot or nearby shots
edges = [(i, j) for (i, j, s) in edges if s >= tau_global]

clusters = connected_components(edges)
# or: clusters = HDBSCAN(...).fit_predict(local_keep.embeddings)

for cluster in clusters:
    rep = medoid(cluster)
    if prompt_available:
        rep = query_aware_rep(cluster, q, alpha=alpha)

    if borderline(cluster) or text_sensitive(cluster):
        rep = verify_with_ocr_ssim_or_sscd(cluster, rep)
```

Về bản chất, đây là cách kết hợp những gì mạnh nhất của temporal modeling, vector search, clustering và query-aware frame selection trong cùng một pipeline. citeturn6view5turn7view5turn6view4turn11view3turn18view3turn12view0

## Cách chọn ngưỡng và đánh giá sao cho có thể ra quyết định kỹ thuật

Phần khó nhất của dedup bằng embedding không nằm ở code, mà nằm ở **chọn ngưỡng đúng**. SSCD chỉ ra rất rõ rằng copy detection khác với retrieval ranking thuần túy, vì bạn cần một hard decision match/non-match chứ không chỉ một thứ tự tương đối. Cũng vì vậy, nếu bạn kỳ vọng một threshold toàn cục dùng tốt cho mọi loại video, thì chỉ cosine similarity của embedding semantic thường chưa đủ ổn định. Ngược lại, HDBSCAN và OPTICS hấp dẫn chính ở chỗ chúng giảm sự lệ thuộc vào một epsilon cố định khi dữ liệu có mật độ thay đổi. citeturn16view0turn6view4turn3search9

Trong thực nghiệm, bạn nên đánh giá theo **hai lớp metric**, không chỉ một. Lớp thứ nhất là metric nội tại của dedup: pair precision/recall/F1 cho nhãn duplicate, reduction ratio, số cluster, cluster purity, và tỉ lệ giữ lại frame đại diện đúng. Lớp thứ hai là metric downstream: latency index/search, memory footprint, và quan trọng nhất là **retrieval quality sau dedup** như Recall@K hoặc mAP cho bài toán prompt-to-keyframe. ProCLIP cho thấy rõ rằng giảm chi phí tính toán chỉ có ý nghĩa khi vẫn giữ được retrieval accuracy; họ dùng prompt-aware sampling kết hợp coarse pruning và CLIP reranking để giảm mạnh độ trễ nhưng vẫn duy trì chất lượng. citeturn18view1turn18view3

Về benchmark, nếu bạn cần proxy dataset cho strict duplicate/copy, **Copydays** và **DISC2021** là hai điểm tham chiếu tốt ở mức image copy detection; SSCD báo cáo mạnh trên cả DISC2021 và Copydays. Nếu bạn muốn nhìn bài toán ở mức video/segment, **VCSL** là một benchmark lớn cho video copy localization với 160k realistic video copy pairs và hơn 280k segment pairs, đi kèm evaluation protocol cho overlap-aware localization. Những benchmark này không trùng hẳn bài toán keyframe dedup của bạn, nhưng rất hữu ích để kiểm tra xem descriptor và threshold có đang thiên về semantic retrieval hay strict copy detection. citeturn16view0turn15search0turn15search5

Một quy trình chọn ngưỡng khá an toàn là thế này: trước hết gán nhãn thủ công một tập validation nhỏ nhưng khó, chia rõ các cặp thành “duplicate thật”, “near-duplicate vẫn nên gộp”, và “semantic giống nhưng không được gộp”. Sau đó quét lưới threshold cho local dedup và global dedup riêng biệt, tối ưu theo mục tiêu downstream chứ không chỉ F1 pairwise. Nếu hệ có OCR/ASR mạnh, hãy thêm penalty cho các trường hợp gộp nhầm frame có text khác nhau hoặc timestamp khác nhau về mặt ngữ nghĩa. Điều này phản ánh đúng rủi ro mà research về text-aware visual similarity đã nêu ra. citeturn12view0turn16view0

## Khuyến nghị chốt cho task của bạn

Nếu mình phải chốt một khuyến nghị thực chiến cho task hiện tại, mình sẽ đề xuất như sau.

**Nếu bạn cần một phiên bản đầu tiên, dễ cài và hiệu quả nhanh**, hãy làm local temporal dedup bằng cosine threshold trên embedding đã normalize, trong cùng track hoặc shot, sau đó dùng Faiss/HNSW để lấy kNN và dựng connected components. Cuối mỗi component, giữ medoid. Đây là cấu hình có tỷ lệ effort/hiệu quả rất tốt, dễ benchmark, và mở đường để scale lên sau này. citeturn7view1turn7view5turn17search0turn10search7

**Nếu dữ liệu của bạn đa dạng và threshold khó ổn định**, hãy thay connected components bằng HDBSCAN. Bạn sẽ mất thêm công tinh chỉnh và hiểu cluster labels/noise points, nhưng đổi lại sẽ đỡ phụ thuộc vào một ngưỡng similarity duy nhất, đặc biệt hữu ích khi mỗi video hoặc mỗi domain có mật độ embedding khác nhau. citeturn6view4turn6view3

**Nếu hệ retrieval phụ thuộc mạnh vào prompt của người dùng**, đừng chỉ dedup offline một lần rồi thôi. Hãy thêm một bước query-aware reranking hoặc query-aware representative selection ở top-K kết quả cuối. Các hướng như KeyVideoLLM, TiFRe và ProCLIP đều cho thấy văn bản truy vấn có thể hướng dẫn chọn frame liên quan hơn rất nhiều so với các chiến lược query-agnostic, đồng thời hai-stage pruning vẫn giảm được độ trễ đáng kể. citeturn11view3turn7view3turn18view1turn18view3

**Nếu video của bạn có nhiều text, subtitle, giao diện, bảng điểm, hoặc OCR là tín hiệu quan trọng**, phải có hybrid verification. Chỉ dùng global embedding để xóa frame là rủi ro cao. Hãy xác minh cặp borderline bằng OCR-aware SSIM hoặc một descriptor chuyên copy detection như SSCD. Đây là lớp bảo vệ giúp bạn không làm hỏng chất lượng retrieval chỉ vì module dedup “quá hăng”. citeturn12view0turn16view0

Và cuối cùng, nếu phải chọn một stack “đáng thử ngay tuần này”, mình sẽ xếp thứ tự như sau: **CLIP hoặc SigLIP 2 embedding cho retrieval-semantic space, local temporal cosine dedup, Faiss/HNSW kNN graph, HDBSCAN hoặc connected components, medoid representative, rồi OCR/SSIM hoặc SSCD verification cho cụm mơ hồ**. Đây là tổ hợp bám sát cả nền tảng lý thuyết lẫn hướng đi của các công trình mới về giảm temporal redundancy, query-aware frame selection, và large-scale embedding search. citeturn6view1turn7view1turn7view5turn17search0turn6view4turn11view3turn12view0turn16view0