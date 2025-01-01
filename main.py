import cv2
import numpy as np
from skimage.feature.texture import graycomatrix
from skimage.feature.texture import graycoprops


def process_image(image_path, output_path):
    # 讀取影像
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to read image.")
        return

    # 將影像轉換為灰階
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 設定濾波器大小和偏移量
    size_filter = 5
    offset = 3
    nr, nc = gray_image.shape[:2]
    x_filter = int(nr / offset)
    y_filter = int(nc / offset)

    # 初始化對比度和相關性矩陣
    co_contrast = np.zeros((x_filter, y_filter))
    co_correlation = np.zeros((x_filter, y_filter))

    # 計算行和列的結束位置
    end_r = nr - size_filter + offset
    end_c = nc - size_filter + offset

    # 計算 GLCM 的對比度和相關性
    for i in range(0, end_r, offset):
        for j in range(0, end_c, offset):
            f = gray_image[i : i + size_filter, j : j + size_filter]
            result = graycomatrix(f, [1], [0])
            co_contrast[int(i / offset), int(j / offset)] = float(
                graycoprops(result, "contrast")
            )
            co_correlation[int(i / offset), int(j / offset)] = float(
                graycoprops(result, "correlation")
            )

    # 創建一個全白的影像
    binary_image = np.ones_like(gray_image) * 255

    # 將異常點變成黑色
    for i in range(x_filter):
        for j in range(y_filter):
            if co_correlation[i, j] > 0.825 and co_contrast[i, j] > 40:
                binary_image[
                    i * offset : i * offset + size_filter,
                    j * offset : j * offset + size_filter,
                ] = 0

    # 儲存處理後的影像
    cv2.imwrite(output_path, binary_image)
    print(f"Processed image saved to {output_path}")

    # 顯示處理後的影像
    cv2.imshow("Processed Image", binary_image)
    cv2.waitKey(0)  # 等待按鍵事件
    cv2.destroyAllWindows()  # 關閉所有 OpenCV 視窗


# 使用範例
input_image_path = "test_img//image1.png"  # 請替換為您的影像路徑
output_image_path = "test_img//image1_result.png"  # 請替換為您希望儲存的影像路徑
process_image(input_image_path, output_image_path)
