import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature.texture import graycomatrix, graycoprops
from joblib import Parallel, delayed
from sklearn.metrics import confusion_matrix


def calculate_confusion_matrix(image1_path, image2_path):
    # 讀取兩張黑白圖片
    image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

    if image1 is None or image2 is None:
        print("Error: Unable to read one or both images.")
        return

    # 確保兩張圖片的尺寸相同
    if image1.shape != image2.shape:
        print("Error: The images do not have the same dimensions.")
        return
    
    # 計算圖片中的像素總數
    total_pixels = image1.shape[0] * image1.shape[1]
    # print(f"Total number of pixels in the images: {total_pixels}")
    

    # 將圖片展平為一維陣列
    image1_flat = image1.flatten()
    image2_flat = image2.flatten()

    # 獲取唯一像素值
    unique_labels = np.unique(np.concatenate((image1_flat, image2_flat)))
    # print(f"Unique pixel values in the images: {unique_labels}")

    # 計算混淆矩陣
    cm = confusion_matrix(image1_flat, image2_flat, labels=unique_labels)
    # print(f"Confusion Matrix:\n{cm}")

    # 如果唯一像素值包含 0 和 255，則打印詳細的混淆矩陣
    if 0 in unique_labels and 255 in unique_labels:
        tn, fp, fn, tp = cm.ravel()
        # print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")

        accuracy = (tp + tn) / (tp + tn + fp + fn)
        # print(f"Accuracy: {accuracy:.2f}")
        return cm,accuracy
    else:
        return cm,None


def process_block(gray_image, size_filter, offset, i, j):
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

    # # 顯示原圖、去噪後的圖像和增強後的圖像
    # fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    # axs[0].imshow(gray_image, cmap="gray")
    # axs[0].set_title("Original Image")
    # axs[0].axis("off")

    # axs[1].imshow(denoised_image, cmap="gray")
    # axs[1].set_title("Denoised Image")
    # axs[1].axis("off")

    # axs[2].imshow(enhanced_image, cmap="gray")
    # axs[2].set_title("Enhanced Image")
    # axs[2].axis("off")

    # plt.show()

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

    # 計算對比度的分佈概率
    contrast_values = np.array(contrast_values)

    # 創建一個全白的影像
    binary_image = np.ones_like(enhanced_image) * 255

    # 將異常點變成黑色
    for i in range(x_filter):
        for j in range(y_filter):
            if (
                co_correlation[i, j] > correlation_threshold
                and co_contrast[i, j] > contrast_threshold
            ):
                # print(
                #     f"Block at ({i}, {j}) marked as abnormal: correlation={co_correlation[i, j]}, contrast={co_contrast[i, j]}"
                # )
                binary_image[
                    i * offset : i * offset + size_filter,
                    j * offset : j * offset + size_filter,
                ] = 0

    # 儲存處理後的影像
    cv2.imwrite(output_path, binary_image)
    # print(f"Processed image saved to {output_path}")

    # # 顯示直方圖和處理後的影像
    # fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # # 繪製直方圖
    # axs[0].hist(contrast_values, bins=50, density=True, alpha=0.75, color="blue")
    # axs[0].set_xlabel("Contrast")
    # axs[0].set_ylabel("Probability")
    # axs[0].set_title("Histogram of Contrast Values")
    # axs[0].grid(True)

    # # 顯示處理後的影像
    # axs[1].imshow(binary_image, cmap="gray")
    # axs[1].set_title("Processed Image")
    # axs[1].axis("off")

    # plt.show()


def unique_pixel_values(image_path):
    # 讀取影像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Unable to read image.")
        return

    # 獲取唯一像素值
    unique_values = np.unique(image)
    print(f"Unique pixel values in the image: {unique_values}")


# 使用範例
# input_image_path = "test_img//image1.png"  # 請替換為您的影像路徑
# output_image_path = "test_img//image1_result.png"  # 請替換為您希望儲存的影像路徑
# ground_truth_path = "test_img//image1_groundtruth.png"  # 請替換為您的影像路徑

# size_filter_values = [16, 25, 36]
# offset_values = [1, 2, 3]
# correlation_threshold_values = [0.8, 0.85, 0.9]
# contrast_threshold_values = [30, 40, 50]
# process_image(
#     input_image_path,
#     output_image_path,
#     size_filter=25,
#     offset=1,
#     correlation_threshold=0.9,
#     contrast_threshold=40,
# )
# calculate_confusion_matrix(output_image_path, ground_truth_path)
# unique_pixel_values(ground_truth_path)
def main(): 
    input_image_path = "test_img//image1.png"  # 請替換為您的影像路徑
    output_image_path = "test_img//image1_result.png"  # 請替換為您希望儲存的影像路徑
    ground_truth_path = "test_img//image1_groundtruth.png"  # 請替換為您的影像路徑

    size_filter_values = [16, 25, 36]
    offset_values = [1, 2, 3]
    correlation_threshold_values = [0.8, 0.85, 0.9]
    contrast_threshold_values = [30, 40, 50]
    with open("results.txt", "w") as file:
        for size_filter in size_filter_values:
            for offset in offset_values:
                for correlation_threshold in correlation_threshold_values:
                    for contrast_threshold in contrast_threshold_values:
                        output_image_path = "test_img//image1_result.png"
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
                        if accuracy is not None:
                            file.write(
                                f"Parameters: size_filter={size_filter}, offset={offset}, correlation_threshold={correlation_threshold}, contrast_threshold={contrast_threshold}\n"
                            )
                            file.write(f"Accuracy: {accuracy:.4f}\n\n")
                        else:
                            file.write(
                                f"Parameters: size_filter={size_filter}, offset={offset}, correlation_threshold={correlation_threshold}, contrast_threshold={contrast_threshold}\n"
                            )
                            file.write("Accuracy: N/A\n\n")
                        print("Results saved to results.txt")
if __name__ == "__main__":
    main()
