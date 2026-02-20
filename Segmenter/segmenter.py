import cv2
import numpy as np
import os

class Segmenter:

    def __init__(self, char_size=(64, 128)):
        self.char_w, self.char_h = char_size
    def preprocess(self, img):
        # Chuyển sang ảnh đen trắng
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Khử nhiễu (7,7) là thông số phù hợp nhất nếu tăng thêm thì ảnh dễ mở và mất nét hơn, nếu giảm đi thì giữ được chi tiết nhưng còn nhiễu
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        # Cân bằng sáng (giới hạn độ tăng tương phản và chia ảnh thành ô 8x8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # áp dụng CLAHE lên ảnh đã khử nhiễu
        gray = clahe.apply(blur)
        return gray
    
    def binarize(self, gray):
        thresh = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, # Ngưỡng tại mỗi pixel được tính dựa trên trung bình có trọng số Gaussian của vùng lân cận
            cv2.THRESH_BINARY_INV, # Ký tự sẽ thành màu trắng, nền thành màu đen 
            11, # kích thước vùng tính ngưỡng
            2 # Giá trị trừ đi khỏi trung bình
        )
        return thresh
    
    def remove_noise(self, binary):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        return clean
    
    # Tìm tất cả vùng trắng - kiểm tra kích thước - kiểm tra hình dạng - giữ lại những vùng có khả năng là ký tự - trả về danh sách bounding box.
    def find_char_contours(self, clean_img, plate_shape):
        contours, _ = cv2.findContours(
            clean_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        plate_h, plate_w = plate_shape[:2]
        char_candidates = []

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            aspect_ratio = h / float(w + 1e-5)

            # Điều kiện lọc contour ký tự
            if (
                area > 100 and
                0.25 * plate_h < h < 0.95 * plate_h and
                w > 0.03 * plate_w and
                1.0 < aspect_ratio < 5.0
            ):
                char_candidates.append((x, y, w, h))

        return char_candidates
    
    #
    def split_if_needed(self, char_boxes):
        if len(char_boxes) <= 1:
            return char_boxes

        widths = [w for (_, _, w, _) in char_boxes]
        avg_w = np.mean(widths)

        final_boxes = []
        for (x, y, w, h) in char_boxes:
            # Nếu ký tự quá rộng - có khả năng dính
            if w > 1.7 * avg_w:
                # Tìm điểm giữa theo chiều ngang 
                mid = x + w // 2
                # Chia thành 2 phần box theo chiều dọc để tách ký tự 
                final_boxes.append((x, y, w // 2, h))
                final_boxes.append((mid, y, w // 2, h))
            else:
                final_boxes.append((x, y, w, h))

        return final_boxes
    
    #   
    def extract_characters(self, clean_img, char_boxes):
        char_images = []

        for (x, y, w, h) in char_boxes:
            char = clean_img[y:y + h  , x:x + w] # lấy the chiểu dọc, lấy theo chiều ngang
            char = cv2.resize(char, (self.char_w, self.char_h))
            char_images.append(char)

        return char_images
    
    #
    def segment(self, plate_img, debug=False):
        gray = self.preprocess(plate_img)
        binary = self.binarize(gray)
        clean = self.remove_noise(binary)

        char_boxes = self.find_char_contours(clean, plate_img.shape)
        char_boxes = self.split_if_needed(char_boxes)

        # Sắp xếp ký tự từ trên xuống dưới từ trái sang phải
        rows = []
        char_boxes = sorted(char_boxes, key=lambda b: (b[1]))

        for box in char_boxes:
            x, y, w, h = box
            placed = False
            
            for row in rows:
                # Nếu y gần với dòng đã có - cùng dòng
                if abs(row[0][1] - y) < h * 0.5:
                    row.append(box)
                    placed = True
                    break
                
            if not placed:
                rows.append([box])
                
        for row in rows:
            row.sort(key=lambda b: b[0])
        char_boxes = []

        for row in rows:
            for box in row:
                char_boxes.append(box)

        if debug:
            debug_img = plate_img.copy()
            for (x, y, w, h) in char_boxes:
                cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.imshow("Plate", plate_img)
            cv2.imshow("Binary", binary)
            cv2.imshow("Clean", clean)
            cv2.imshow("Characters", debug_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        char_images = self.extract_characters(clean, char_boxes)
        
        return char_images
    

if __name__ == "__main__":
    img = cv2.imread("img/images.jpg")

    segmenter = Segmenter()
    chars = segmenter.segment(img, debug=True)
   
    for i, ch in enumerate(chars):
        cv2.imshow(f"char_{i}", ch)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
