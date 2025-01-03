import cv2
import numpy as np
from sklearn.metrics import confusion_matrix
from skimage.feature.texture import graycomatrix, graycoprops
from joblib import Parallel, delayed
import matplotlib.pyplot as plt


def display_images(input_image_path, output_image_path, ground_truth_path):
    input_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    output_image = cv2.imread(output_image_path, cv2.IMREAD_GRAYSCALE)
    ground_truth_image = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)

    if input_image is None or output_image is None or ground_truth_image is None:
        print("Error: Unable to read one or more images.")
        return

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(input_image, cmap="gray")
    plt.title("Input Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(output_image, cmap="gray")
    plt.title("Output Image")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(ground_truth_image, cmap="gray")
    plt.title("Ground Truth Image")
    plt.axis("off")

    plt.show()


def calculate_confusion_matrix(image1_path, image2_path):
    # 讀取兩張黑白圖片
    image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

    if image1 is None or image2 is None:
        print("Error: Unable to read one or both images.")
        return None, None

    # 確保兩張圖片的尺寸相同
    if image1.shape != image2.shape:
        print("Error: The images do not have the same dimensions.")
        return None, None

    # 將圖片二值化
    _, image1_bin = cv2.threshold(image1, 127, 255, cv2.THRESH_BINARY)
    _, image2_bin = cv2.threshold(image2, 127, 255, cv2.THRESH_BINARY)

    # 將圖片展平為一維陣列
    image1_flat = image1_bin.flatten()
    image2_flat = image2_bin.flatten()

    # 獲取唯一像素值
    unique_labels = np.unique(np.concatenate((image1_flat, image2_flat)))

    # 計算混淆矩陣
    cm = confusion_matrix(image1_flat, image2_flat, labels=unique_labels)

    # 如果唯一像素值包含 0 和 255，則計算準確率
    if 0 in unique_labels and 255 in unique_labels:
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        return cm, accuracy
    else:
        return cm, None


def process_block(gray_image, size_filter, offset, i, j):
    """ 計算灰度共生矩陣 """
    f = gray_image[i : i + size_filter, j : j + size_filter]
    result = graycomatrix(f, [1], [0])
    contrast = float(graycoprops(result, "contrast"))
    correlation = float(graycoprops(result, "correlation"))
    return (i, j, contrast, correlation)


def process_image(
    image_path,
    output_path,
    size_filter=25,
    offset=5,
    correlation_threshold=0.825,
    contrast_threshold=40,
):
    # 讀取影像
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to read image.")
        return

    # 將影像轉換為灰階
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 影像去噪
    denoised_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # 影像增強
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(denoised_image)

    # 設定濾波器大小和偏移量
    nr, nc = enhanced_image.shape[:2]
    x_filter = int((nr - size_filter) / offset) + 1
    y_filter = int((nc - size_filter) / offset) + 1

    # 初始化對比度和相關性矩陣
    co_contrast = np.zeros((x_filter, y_filter))
    co_correlation = np.zeros((x_filter, y_filter))

    # 使用並行處理計算 GLCM 的對比度和相關性
    results = Parallel(n_jobs=-1)(
        delayed(process_block)(enhanced_image, size_filter, offset, i, j)
        for i in range(0, nr - size_filter + 1, offset)
        for j in range(0, nc - size_filter + 1, offset)
    )

    contrast_values = []

    for i, j, contrast, correlation in results:
        co_contrast[int(i / offset), int(j / offset)] = contrast
        co_correlation[int(i / offset), int(j / offset)] = correlation
        contrast_values.append(contrast)


    contrast_values = np.array(contrast_values)

    # 創建一個全白影像
    binary_image = np.ones_like(enhanced_image) * 255

    # 將異常點變成黑色
    for i in range(x_filter):
        for j in range(y_filter):
            if (
                co_correlation[i, j] > correlation_threshold
                and co_contrast[i, j] > contrast_threshold
            ):
                binary_image[
                    i * offset : i * offset + size_filter,
                    j * offset : j * offset + size_filter,
                ] = 0

    # 儲存
    cv2.imwrite(output_path, binary_image)
    print(f"Processed image saved to {output_path}")


def main():
    """ 用for迴圈遍歷參數找最優解 """
    input_image_path = "test_img//image1.png"  # 請替換為您的影像路徑
    ground_truth_path = "test_img//image1_groundtruth.png"  # 請替換為您的影像路徑
    size_filter_values = [ 20, 25, 30, 35, 40]
    offset_values = [1, 2, 3]
    correlation_threshold_values = [0.75, 0.8, 0.85, 0.9]
    contrast_threshold_values = [30, 40, 50, 60, 70]

    with open("results2.txt", "w") as file:

        file.flush()
        for size_filter in size_filter_values:
            for offset in offset_values:
                for correlation_threshold in correlation_threshold_values:
                    for contrast_threshold in contrast_threshold_values:
                        output_image_path = "test_img//image1_result.png"
                        print(
                            f"Processing with parameters: size_filter={size_filter}, offset={offset}, correlation_threshold={correlation_threshold}, contrast_threshold={contrast_threshold}"
                        )
                        process_image(
                            input_image_path,
                            output_image_path,
                            size_filter=size_filter,
                            offset=offset,
                            correlation_threshold=correlation_threshold,
                            contrast_threshold=contrast_threshold,
                        )
                        cm, accuracy = calculate_confusion_matrix(
                            output_image_path, ground_truth_path
                        )
                        if cm is not None and accuracy is not None:
                            print(
                                f"Writing results for parameters: size_filter={size_filter}, offset={offset}, correlation_threshold={correlation_threshold}, contrast_threshold={contrast_threshold}"
                            )
                            file.write(
                                f"Parameters: size_filter={size_filter}, offset={offset}, correlation_threshold={correlation_threshold}, contrast_threshold={contrast_threshold}\n"
                            )
                            file.write(f"Accuracy: {accuracy:.4f}\n\n")
                            file.flush()
                        else:
                            print(
                                f"Skipping parameters due to None result: size_filter={size_filter}, offset={offset}, correlation_threshold={correlation_threshold}, contrast_threshold={contrast_threshold}"
                            )
                            file.write(
                                f"Parameters: size_filter={size_filter}, offset={offset}, correlation_threshold={correlation_threshold}, contrast_threshold={contrast_threshold}\n"
                            )
                            file.write("Accuracy: N/A\n\n")
                            file.flush()
    print("Results saved to results.txt")


if __name__ == "__main__":
    # main() 找最佳參數的function

    input_image_path = "test_img//image1.png"  # 初始圖片 
    ground_truth_path = "test_img//image1_groundtruth.png"  # 結果圖
    output_image_path = "test_img//image1_result.png" # 輸出圖片
    # Parameters: size_filter=20, offset=3, correlation_threshold=0.9, contrast_threshold=60 最優解
    process_image(
    input_image_path,
    output_image_path,
    size_filter=20,
    offset=3,
    correlation_threshold=0.9,
    contrast_threshold=60,
    )
    cm, accuracy = calculate_confusion_matrix(output_image_path, ground_truth_path)
    if cm is not None and accuracy is not None:
        print(f"Confusion Matrix:\n{cm}")
        print(f"Accuracy: {accuracy:.4f}")
    display_images(input_image_path, output_image_path, ground_truth_path)
    
