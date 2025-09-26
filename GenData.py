# GenData.py

import numpy as np
import cv2
import sys

# Biến toàn cục
MIN_CONTOUR_AREA = 40  # Diện tích tối thiểu của contour để được xem xét

RESIZED_IMAGE_WIDTH = 20      # Chiều rộng ảnh sau khi resize
RESIZED_IMAGE_HEIGHT = 30     # Chiều cao ảnh sau khi resize

def main():
    imgTrainingNumbers = cv2.imread("training_chars.png")  # Đọc ảnh chứa các ký tự mẫu

    imgGray = cv2.cvtColor(imgTrainingNumbers, cv2.COLOR_BGR2GRAY)  # Chuyển sang ảnh xám
    imgBlurred = cv2.GaussianBlur(imgGray, (5,5), 0)  # Làm mờ ảnh xám

    # Chuyển ảnh thành ảnh đen trắng bằng ngưỡng thích nghi (adaptive threshold)
    imgThresh = cv2.adaptiveThreshold(imgBlurred,
                                      255, 
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 
                                      11, 
                                      2)

    cv2.imshow("imgThresh", imgThresh)  # Hiển thị ảnh sau khi xử lý ngưỡng

    imgThreshCopy = imgThresh.copy()  # Tạo bản sao vì hàm findContours sẽ thay đổi ảnh

    # Tìm các đường viền ngoài cùng trong ảnh trắng đen
    npaContours, hierarchy = cv2.findContours(imgThreshCopy,
                                              cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_SIMPLE)

    # Tạo mảng rỗng để chứa ảnh sau khi được làm phẳng (flattened)
    npaFlattenedImages = np.empty((0, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))

    intClassifications = []  # Danh sách lưu nhãn cho từng ảnh mẫu

    # Danh sách các ký tự hợp lệ cần nhận dạng (theo mã ASCII)
    intValidChars = [ord('0'), ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6'), ord('7'), ord('8'), ord('9'),
                     ord('A'), ord('B'), ord('C'), ord('D'), ord('E'), ord('F'), ord('G'), ord('H'), ord('I'), ord('J'),
                     ord('K'), ord('L'), ord('M'), ord('N'), ord('O'), ord('P'), ord('Q'), ord('R'), ord('S'), ord('T'),
                     ord('U'), ord('V'), ord('W'), ord('X'), ord('Y'), ord('Z')]

    for npaContour in npaContours:
        if cv2.contourArea(npaContour) > MIN_CONTOUR_AREA:  # Chỉ xét các contour đủ lớn
            [intX, intY, intW, intH] = cv2.boundingRect(npaContour)  # Lấy hình chữ nhật bao quanh contour

            # Vẽ khung đỏ quanh ký tự để người dùng biết đang nhập ký tự nào
            cv2.rectangle(imgTrainingNumbers, (intX, intY), (intX+intW, intY+intH), (0, 0, 255), 2)

            # Cắt và resize vùng ảnh chứa ký tự
            imgROI = imgThresh[intY:intY+intH, intX:intX+intW]
            imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))

            cv2.imshow("imgROI", imgROI)                  # Hiển thị ảnh cắt ra
            cv2.imshow("imgROIResized", imgROIResized)    # Hiển thị ảnh sau khi resize
            cv2.imshow("training_numbers.png", imgTrainingNumbers)  # Ảnh ban đầu với khung đỏ

            intChar = cv2.waitKey(0)  # Chờ người dùng nhấn phím để nhập nhãn ký tự

            if intChar == 27:  # Nhấn ESC để thoát
                sys.exit()
            elif intChar in intValidChars:  # Nếu ký tự hợp lệ thì lưu nhãn và ảnh mẫu
                intClassifications.append(intChar)

                # Làm phẳng ảnh để lưu vào tệp
                npaFlattenedImage = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))

                # Thêm ảnh đã làm phẳng vào danh sách
                npaFlattenedImages = np.append(npaFlattenedImages, npaFlattenedImage, 0)

    # Chuyển danh sách nhãn sang mảng float và reshape để lưu
    fltClassifications = np.array(intClassifications, np.float32)
    npaClassifications = fltClassifications.reshape((fltClassifications.size, 1))

    print("\n\nQuá trình huấn luyện hoàn tất!!\n")

    # Ghi nhãn và ảnh mẫu vào tệp
    np.savetxt("classifications.txt", npaClassifications)
    np.savetxt("flattened_images.txt", npaFlattenedImages)

    cv2.destroyAllWindows()  # Đóng tất cả cửa sổ hiển thị

    return

if __name__ == "__main__":
    main()
# end if
