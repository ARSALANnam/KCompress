import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def compress_image_kmeans(image_bytes, k_clusters=16):

    np_array = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise valueError("Could not decode image from provided bytes. Please check the image file.")
    
    img_rgb = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB)

    pixels = img_rgb.reshape(-1,3)

    kmeans = KMeans(n_clusters=k_clusters ,random_state=42, n_init=10)
    kmeans.fit(pixels)

    new_colors = kmeans.cluster_centers_[kmeans.labels_]
    compressed_img_rgb = new_colors.reshape(img_rgb.shape).astype(np.uint8)
    
    compressed_img_bgr = cv2.cvtColor(compressed_img_rgb, cv2.COLOR_RGB2BGR)
    is_success, buffer = cv2.imencode(".jpg", compressed_img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
    if not is_success:
        raise Exception("Could not encode compressed image to PNG.")

    return buffer.tobytes(), compressed_img_bgr
