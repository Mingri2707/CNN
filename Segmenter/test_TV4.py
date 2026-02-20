import cv2
from ultralytics import YOLO

from segmenter import Segmenter 

def main():
    # Đừng dẫn này đến file best mỗi người mỗi khác nhớ đổi lại

    model_path = r"D:\new_2025\2025 - 2026\DTU\DS 371 D\DAN\TrainYoLo\runs\detect\ket_qua_train_official\weights\best.pt"
    model = YOLO(model_path)

    # --- 2. ĐỌC ẢNH TEST ---
    duong_dan_anh = r"D:\new_2025\2025 - 2026\DTU\DS 371 D\DAN\Final\Nam\Segmenter\test_bien\5.png" 
    img = cv2.imread(duong_dan_anh)

    if img is None:
        print("Lỗi")
        return

    print("[YOLO] Đang tìm kiếm biển số trong ảnh...")
    results = model.predict(source=img, conf=0.6, show=False, verbose=False)

    # --- 3. KIỂM TRA & CẮT ẢNH ---
    if len(results[0].boxes) > 0:
        print("[YOLO] Đã bắt được biển số! Đang tiến hành cắt ảnh...")
        
        # Lấy tọa độ x, y của khung biển số
        box = results[0].boxes[0]
        x1, y1, x2, y2 = map(int, box.xyxy[0]) 

        # CẮT: Lấy cái biển số ra khỏi tấm ảnh to
        anh_bien_so_da_cat = img[y1:y2, x1:x2]

        print("[YOLO] Đã cắt xong biển số! Đang chuyển cho code của TV4 xử lý...\n")

        cong_cu_cua_tv4 = Segmenter()
        
        cac_ky_tu_cat_duoc = cong_cu_cua_tv4.segment(anh_bien_so_da_cat, debug=True)

        print(f"🎉 [TV4] Code TV4 đã bóc tách thành công {len(cac_ky_tu_cat_duoc)} ký tự!")

        # Hiện từng ký tự mà TV4 đã chặt ra lên màn hình
        for i, char_img in enumerate(cac_ky_tu_cat_duoc):
            cv2.imshow(f"Ky tu thu {i+1}", char_img)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    else:
        print("[YOLO] Không tìm thấy biển số nào trong ảnh này!")

if __name__ == '__main__':
    main()