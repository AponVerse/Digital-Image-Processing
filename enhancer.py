import cv2
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# 1. Read Image & Preprocessing
# -------------------------------
def read_and_preprocess(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, cv2.GaussianBlur(gray, (3, 3), 0)

# -------------------------------
# 2. Detect Faces (Haar Cascade)
# -------------------------------
def detect_faces_haar(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                                         'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(image, 1.1, 5)
    mask = np.zeros(image.shape, dtype=np.uint8)
    for (x, y, w, h) in faces:
        mask[y:y+h, x:x+w] = 255
    if np.count_nonzero(mask) == 0:
        return None  # No face found
    mask = cv2.GaussianBlur(mask, (21, 21), 0)
    return mask

# -------------------------------
# 3. Detect Sky Region
# -------------------------------
def detect_sky(gray_img):
    h, w = gray_img.shape
    top_region = gray_img[:h//3, :]
    _, sky_mask = cv2.threshold(top_region, 180, 255, cv2.THRESH_BINARY)
    full_mask = np.zeros_like(gray_img)
    full_mask[:h//3, :] = sky_mask
    if np.count_nonzero(full_mask) == 0:
        return None  # No sky detected
    full_mask = cv2.GaussianBlur(full_mask, (21, 21), 0)
    return full_mask

# -------------------------------
# 4. Histogram Equalization
# -------------------------------
def apply_clahe(gray_img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(gray_img)

# -------------------------------
# 5. Gamma Correction
# -------------------------------
def gamma_correction(img, gamma):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(img, table)

# -------------------------------
# 6. Unsharp Mask
# -------------------------------
def unsharp_mask(img, kernel_size=(5,5), sigma=1.0, amount=1.0):
    blurred = cv2.GaussianBlur(img, kernel_size, sigma)
    sharp = cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)
    return sharp

# -------------------------------
# 7. Main Processing Pipeline
# -------------------------------
def process_image(image_path):
    _, gray = read_and_preprocess(image_path)

    # Step 1: Global contrast enhancement
    clahe_img = apply_clahe(gray)

    # Step 2: Detect zones
    face_mask = detect_faces_haar(gray)
    sky_mask = detect_sky(gray)

    # Step 3: If no zones found, do safe global enhancement
    if face_mask is None and sky_mask is None:
        final_img = unsharp_mask(clahe_img, amount=0.8)  # mild sharpen
        return gray, final_img

    result = clahe_img.copy()

    # Step 4: Enhance faces if found
    if face_mask is not None:
        face_enhanced = gamma_correction(clahe_img, gamma=1.2)
        alpha_face = face_mask.astype(np.float32) / 255
        result = cv2.convertScaleAbs(result * (1 - alpha_face) + face_enhanced * alpha_face)

    # Step 5: Enhance skies if found
    if sky_mask is not None:
        sky_enhanced = gamma_correction(clahe_img, gamma=1.1)
        alpha_sky = sky_mask.astype(np.float32) / 255
        result = cv2.convertScaleAbs(result * (1 - alpha_sky) + sky_enhanced * alpha_sky)

    # Step 6: Final mild sharpening
    final_img = unsharp_mask(result, amount=0.8)

    return gray, final_img

# -------------------------------
# 8. Run & Show Result
# -------------------------------
if __name__ == "__main__":
    input_image_path = "old_photo.jpg"  # Your scanned photo
    original, enhanced = process_image(input_image_path)

    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.title("Original")
    plt.imshow(original, cmap='gray')
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.title("Enhanced")
    plt.imshow(enhanced, cmap='gray')
    plt.axis('off')
    plt.show()

    cv2.imwrite("enhanced_output.jpg", enhanced)
