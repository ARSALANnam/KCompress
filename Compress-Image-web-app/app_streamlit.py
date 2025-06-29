import streamlit as st
from PIL import Image
import io
import time
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from image_processor import compress_image_kmeans

# Streamlit تنظیمات صفحه
st.set_page_config(
    page_title="فشرده‌سازی عکس با K-Means",
    page_icon="🎨",
    layout="centered",
    initial_sidebar_state="expanded"
)

# عنوان و توضیحات
st.title("🎨 K-Means فشرده‌سازی تصویر با ")
st.write(".عکس خود را آپلود کنید، تعداد رنگ‌های مورد نظر را انتخاب کنید و تصویر را فشرده‌سازی کنید")
st.markdown("---")

# آپلود فایل 
uploaded_file = st.file_uploader(
    ":(JPG, JPEG, PNG, GIF)یک عکس برای فشرده سازی انتخاب کنید",
    type=["jpg", "jpeg", "png", "gif"]
)

# بررسی وجود فایل آپلود شده 
if uploaded_file is not None:
    # Read the image bytes once
    original_image_bytes = uploaded_file.read()

    # نمایش عکس اصلی
    st.subheader("عکس اصلی")
    try:
        # Use PIL for displaying as it's generally more robust with Streamlit
        original_image_pil = Image.open(io.BytesIO(original_image_bytes))
        st.image(original_image_pil, caption="عکس آپلود شده", use_container_width=True)
    except Exception as e:
        st.error(f"خطا در بارگذاری عکس اصلی: {e}")
        st.stop()

    st.markdown("---")

    # انتخاب تعداد رنگ (K) 
    st.subheader("تنظیمات فشرده‌سازی")
    k_clusters = st.slider(
        "تعداد رنگ‌های نهایی (K):",
        min_value=2,
        max_value=128, # می توانید این مقدار را تغییر دهید
        value=16, # مقدار پیش‌فرض
        step=1,
        help=" .کمتر باشد، فشرده‌سازی بیشتر و جزئیات رنگ کمتر می‌شود Kهرچه"
    )

    # دکمه فشرده‌سازی
    if st.button("🖼️ فشرده‌سازی عکس"):
        with st.spinner("در حال فشرده‌سازی عکس... لطفاً صبر کنید."):
            try:
                
                # تبدیل تصویر اصلی به numpy
                original_image_np = np.array(original_image_pil)
                original_image_np_bgr = cv2.cvtColor(original_image_np, cv2.COLOR_RGB2BGR)

                start_time = time.time()
                for i in range(1000000):
                    pass


                # Call the compression function
                compressed_image_bytes, compressed_np = compress_image_kmeans(original_image_bytes, k_clusters)

                end_time = time.time()
                processing_time = end_time - start_time
                
                # SSIM
                compressed_rgb = cv2.cvtColor(compressed_np, cv2.COLOR_BGR2RGB)
                ssim_value = ssim(original_image_np, compressed_rgb, multichannel=True, data_range=255, win_size=3)

                # محاسبه حجم
                original_size_kb = len(original_image_bytes) / 1024
                compressed_size_kb = len(compressed_image_bytes) / 1024
               
                compression_ratio = original_size_kb / compressed_size_kb
                reduction_percentage = ((original_size_kb - compressed_size_kb) / original_size_kb) * 100
                
                st.success("فشرده‌سازی با موفقیت انجام شد!")

                #  نمایش عکس فشرده شده 
                st.subheader("عکس فشرده شده")
                compressed_image_pil = Image.open(io.BytesIO(compressed_image_bytes))
                st.image(compressed_image_pil, caption=f"عکس فشرده شده (K={k_clusters} رنگ)", use_container_width=True)

                # دکمه دانلود
                st.download_button(
                    label="📥 ذخیره عکس فشرده شده",
                    data=compressed_image_bytes,
                    file_name=f"compressed_image_k{k_clusters}.JPG",
                    mime="image/jpg",
                    help="تصویر فشرده شده را با فرمت JPG ذخیره می‌کند."
                )
                
                # داشبورد 
                st.markdown("---")
                st.subheader("📊 داشبورد فشرده‌سازی")

                with st.container():
                    st.metric("SSIM", f"{ssim_value:.4f}", help="مقدار شباهت ساختاری تصویر")
                    st.metric("⏱️ زمان اجرا", f"{processing_time:.4f} ثانیه")
                    st.metric("حجم اولیه عکس", f"{original_size_kb:.2f} KB") 
                    st.metric("حجم فشرده عکس", f"{compressed_size_kb:.2f} KB")
                    st.metric("📦 نسبت فشرده‌سازی", f"{compression_ratio:.2f}x")
                    st.metric("⬇️ درصد کاهش", f"{reduction_percentage:.2f}%")

            except ValueError as ve:
                st.error(f"خطای ورودی: {ve}")
            except Exception as e:
                st.error(f"خطایی در حین فشرده‌سازی رخ داد: {e}")
                st.exception(e) # نمایش جزئیات خطا برای دیباگ

else:
    st.info("لطفاً یک عکس برای شروع فشرده‌سازی آپلود کنید.")

st.markdown("---")
st.markdown(" ⚡ Streamlit و ARSALANnam(Echolyno) ساخته شده توسط ")
st.markdown("Github : https://github.com/ARSALANnam")