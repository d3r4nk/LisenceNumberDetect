import cv2
import numpy as np
import math

# Kích thước bộ lọc Gauss để làm mờ ảnh
GAUSSIAN_SMOOTH_FILTER_SIZE = (5, 5)
# Kích thước vùng lân cận và trọng số cho ngưỡng thích nghi
ADAPTIVE_THRESH_BLOCK_SIZE = 19 
ADAPTIVE_THRESH_WEIGHT = 9  

# Tiền xử lý ảnh để tách biển số
def preprocess(imgOriginal):
    imgGrayscale = extractValue(imgOriginal)  # Chuyển đổi ảnh sang mức xám
    imgMaxContrastGrayscale = maximizeContrast(imgGrayscale)  # Tăng độ tương phản
    height, width = imgGrayscale.shape
    imgBlurred = np.zeros((height, width, 1), np.uint8)
    imgBlurred = cv2.GaussianBlur(imgMaxContrastGrayscale, GAUSSIAN_SMOOTH_FILTER_SIZE, 0)  # Làm mịn ảnh
    imgThresh = cv2.adaptiveThreshold(imgBlurred, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_WEIGHT)  # Phân ngưỡng thích nghi
    return imgGrayscale, imgThresh

# Trích xuất kênh giá trị (độ sáng) từ ảnh HSV
def extractValue(imgOriginal):
    height, width, numChannels = imgOriginal.shape
    imgHSV = np.zeros((height, width, 3), np.uint8)
    imgHSV = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2HSV)  # Chuyển đổi sang hệ màu HSV
    imgHue, imgSaturation, imgValue = cv2.split(imgHSV)  # Tách các kênh màu
    return imgValue  # Trả về kênh giá trị (V)

# Tăng cường độ tương phản của ảnh mức xám
def maximizeContrast(imgGrayscale):
    height, width = imgGrayscale.shape
    imgTopHat = np.zeros((height, width, 1), np.uint8)
    imgBlackHat = np.zeros((height, width, 1), np.uint8)
    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # Tạo phần tử cấu trúc
    imgTopHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_TOPHAT, structuringElement, iterations = 10)  # Làm nổi bật vùng sáng
    imgBlackHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_BLACKHAT, structuringElement, iterations = 10)  # Làm nổi bật vùng tối
    imgGrayscalePlusTopHat = cv2.add(imgGrayscale, imgTopHat)  # Cộng thêm vùng sáng nổi bật
    imgGrayscalePlusTopHatMinusBlackHat = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)  # Trừ đi vùng tối nổi bật
    return imgGrayscalePlusTopHatMinusBlackHat
