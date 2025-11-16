# enhancer_fixed.py
import cv2
import numpy as np

# -------------------------------
# Utility / I/O helpers
# -------------------------------
def read_image(path):
    img = cv2.imread(r"C:\Users\User\Documents\MyPython programme\Photo_inhancement\Old_1.jpg")
    if img is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray

# -------------------------------
# 1. Preprocessing: denoise + normalize
# -------------------------------
def denoise_and_normalize(gray):
    # Non-local means denoising - preserves edges better than Gaussian for textures/scratches
    denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
    # Light contrast normalization (avoid over-boost)
    norm = cv2.normalize(denoised, None, 0, 255, cv2.NORM_MINMAX)
    return denoised, norm

# -------------------------------
# 2. Scratch/spot detection & inpainting (robust)
# -------------------------------
def detect_and_inpaint_scratches(gray):
    """
    Robust scratch detection:
      - combine morphological top-hat (detect bright small features)
      - Canny to get thin edges
      - keep connected components that are thin / small (likely scratches/dust)
      - remove very large components (avoid inpainting large areas)
    Returns:
      scratch_mask, inpainted_image
    """
    h, w = gray.shape
    # top-hat to find bright small features
    kernel_size = max(9, min(25, int(min(h, w) * 0.02)))  # adaptive kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)

    # normalize and threshold to obtain candidate bright spots
    th_val = 20
    _, th = cv2.threshold(tophat, th_val, 255, cv2.THRESH_BINARY)

    # detect edges (thin scratches)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)), iterations=1)

    # combine thin edges and bright spots (unions often catch scratches)
    candidate = cv2.bitwise_or(th, edges)

    # small morphological cleanup
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    candidate = cv2.medianBlur(candidate, 3)

    # connected components analysis: keep components that are narrow/relatively small
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(candidate, connectivity=8)
    mask = np.zeros_like(candidate)
    img_area = h * w
    kept_area = 0

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        wbox = stats[i, cv2.CC_STAT_WIDTH]
        hbox = stats[i, cv2.CC_STAT_HEIGHT]

        # compute aspect ratio: scratches tend to have high aspect ratio or thin bounding boxes
        aspect = max(wbox / (hbox + 1e-6), hbox / (wbox + 1e-6))

        # heuristics: keep components that are small-ish and/or thin (likely scratches)
        if area < 1000 and (area < 200 or aspect > 2.0):
            mask[labels == i] = 255
            kept_area += area

    # If nothing kept, return empty mask and the original image
    if np.count_nonzero(mask) < 50:
        return np.zeros_like(mask), gray.copy()

    # Final cleanup: dilate slightly to cover the line width
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)), iterations=1)
    mask = cv2.medianBlur(mask, 5)

    # Safety: if mask covers too much (e.g. >12% of image) reduce it by removing largest components
    max_allowed = int(0.12 * img_area)
    if np.count_nonzero(mask) > max_allowed:
        # remove largest components until below threshold
        num_labels2, labels2, stats2, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        # gather (label, area) excluding background
        comps = [(i, stats2[i, cv2.CC_STAT_AREA]) for i in range(1, num_labels2)]
        comps_sorted = sorted(comps, key=lambda x: x[1], reverse=True)
        cur_mask = mask.copy()
        for label, area in comps_sorted:
            if np.count_nonzero(cur_mask) <= max_allowed:
                break
            # remove this component
            cur_mask[labels2 == label] = 0
        mask = cur_mask

    # Inpaint with Telea (fast & good for small defects)
    inpainted = cv2.inpaint(gray, mask, 3, cv2.INPAINT_TELEA)

    return mask, inpainted

# -------------------------------
# 3. Face detection (Haar) -> smooth mask
# -------------------------------
def detect_faces_haar_smooth(gray):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # run detector on the denoised/normalized image (input should be pre-denoise but BEFORE inpaint)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=4, minSize=(30,30), maxSize=(1500,1500))
    mask = np.zeros_like(gray, dtype=np.uint8)

    for (x, y, w, h) in faces:
        # expand rectangle slightly to include cheeks/forehead
        pad_w, pad_h = int(0.15*w), int(0.18*h)
        x1, y1 = max(0, x - pad_w), max(0, y - pad_h)
        x2, y2 = min(gray.shape[1], x + w + pad_w), min(gray.shape[0], y + h + pad_h)
        mask[y1:y2, x1:x2] = 255

    if np.count_nonzero(mask) == 0:
        return None

    # close small holes and blur to get soft alpha mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25,25))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.GaussianBlur(mask, (31,31), 0)
    return mask

# -------------------------------
# 4. CLAHE (safe, mild)
# -------------------------------
def apply_clahe(gray, clip=1.8, grid=(8,8)):
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
    return clahe.apply(gray)

# -------------------------------
# 5. Gamma correction
# -------------------------------
def gamma_correction(img, gamma=1.0):
    if gamma == 1.0:
        return img.copy()
    inv = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img, table)

# -------------------------------
# 6. Unsharp mask (gentle)
# -------------------------------
def unsharp_mask(img, amount=0.6, kernel_size=(5,5), sigma=1.0):
    blurred = cv2.GaussianBlur(img, kernel_size, sigma)
    sharp = cv2.addWeighted(img.astype(np.float32), 1.0 + amount, blurred.astype(np.float32), -amount, 0.0)
    sharp = np.clip(sharp, 0, 255).astype(np.uint8)
    return sharp

# -------------------------------
# 7. Main pipeline (returns intermediate results)
# -------------------------------
def process_image(image_path):
    # Read
    color, gray_orig = read_image(image_path)

    # Step A: Preprocess (denoise + normalize)
    denoised, normalized = denoise_and_normalize(gray_orig)

    # Step B: Face detection BEFORE inpainting (so faces are not lost)
    face_mask = detect_faces_haar_smooth(denoised)  # pass denoised (better detection)

    # Step C: Scratch detection & inpaint -> reduces artifacts before CLAHE
    scratch_mask, inpainted = detect_and_inpaint_scratches(normalized)

    # Step D: CLAHE applied to inpainted image (mild)
    clahe_img = apply_clahe(inpainted, clip=1.8, grid=(8,8))

    # Step E: Face-specific enhancement (selective gamma + mild local contrast)
    face_enhanced = clahe_img.copy()
    if face_mask is not None:
        # slightly stronger gamma for faces but gentle overall
        face_gamma = gamma_correction(clahe_img, gamma=1.18)
        a = (face_mask.astype(np.float32) / 255.0)
        face_enhanced = cv2.convertScaleAbs(clahe_img * (1 - a) + face_gamma * a)

    # Step F: Global mild gamma to brighten shadows slightly (avoid double brightening faces)
    global_gamma = gamma_correction(face_enhanced, gamma=1.04)

    # Step G: Final mild sharpening (unsharp)
    final = unsharp_mask(global_gamma, amount=0.6)

    # Prepare displayable masks (as images)
    scratch_mask_show = (scratch_mask.astype(np.uint8) * 1) if scratch_mask is not None else np.zeros_like(gray_orig)
    face_mask_show = (face_mask.astype(np.uint8) * 1) if face_mask is not None else np.zeros_like(gray_orig)

    results = {
        "Original": gray_orig,
        "Denoised": denoised,
        "Normalized": normalized,
        "Face Mask (before inpaint)": face_mask_show,
        "Scratch Mask": scratch_mask_show,
        "Inpainted": inpainted,
        "CLAHE": clahe_img,
        "Face Enhanced (local gamma)": face_enhanced,
        "Global Gamma": global_gamma,
        "Final Output": final
    }

    return results

# -------------------------------
# If run as script show images
# -------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import sys

    # Use argument path or change here
    img_path = "old_photo.jpg"
    if len(sys.argv) > 1:
        img_path = sys.argv[1]

    res = process_image(img_path)

    # Display in grid
    titles = list(res.keys())
    n = len(titles)
    cols = 3
    rows = (n + cols - 1) // cols

    plt.figure(figsize=(14, 5 * rows))
    for i, k in enumerate(titles):
        plt.subplot(rows, cols, i + 1)
        plt.title(k)
        plt.imshow(res[k], cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

    # Save final output for convenience
    cv2.imwrite("enhanced_fixed_output.jpg", res["Final Output"])
