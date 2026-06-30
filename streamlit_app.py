import streamlit as st
import cv2
import numpy as np
import re
from PIL import Image
from ultralytics import YOLO

# إعداد الصفحة وتصميمها
st.set_page_config(
    page_title="كاشف اللوحات المصرية الذكي",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# إضافة CSS مخصص لإضفاء مظهر احترافي ومميز (Premium Design)
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@300;400;600;700&display=swap');
        
        * {
            font-family: 'Cairo', sans-serif;
        }
        
        .main-title {
            text-align: center;
            color: #0078D7;
            font-weight: 700;
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
        }
        
        .sub-title {
            text-align: center;
            color: #555555;
            font-size: 1.2rem;
            margin-bottom: 2rem;
        }
        
        .card-container {
            background-color: #f8f9fa;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            margin-bottom: 1.5rem;
            border-right: 5px solid #0078D7;
        }
        
        .card-title {
            color: #0078D7;
            font-weight: 600;
            font-size: 1.1rem;
            margin-bottom: 1rem;
        }
        
        /* تصميم لوحة الترخيص المصرية */
        .egypt-plate {
            border: 4px solid #111111;
            border-radius: 10px;
            width: 320px;
            background-color: #FFFFFF;
            margin: 20px auto;
            overflow: hidden;
            box-shadow: 0 8px 16px rgba(0,0,0,0.15);
        }
        
        .egypt-plate-header {
            background-color: #0078D7; /* اللون الأزرق للسيارات الملاكي */
            height: 35px;
            color: #FFFFFF;
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0 15px;
            font-weight: 700;
            font-size: 14px;
            border-bottom: 2px solid #111111;
        }
        
        .egypt-plate-body {
            display: flex;
            justify-content: space-around;
            align-items: center;
            height: 80px;
            font-size: 32px;
            font-weight: 700;
            color: #111111;
            direction: rtl;
            padding: 0 10px;
        }
        
        .plate-divider {
            width: 3px;
            height: 100%;
            background-color: #111111;
        }
    </style>
""", unsafe_allow_html=True)

# دالة تحميل الموديلات مع التخزين المؤقت لتجنب التحميل المتكرر
@st.cache_resource
def load_models():
    try:
        # تحميل الموديلات محلياً
        plate_model = YOLO("best.pt")
        char_model = YOLO("fastapi_app/characters_yolo12n_best.pt")
        return plate_model, char_model
    except Exception as e:
        st.error(f"خطأ أثناء تحميل الموديلات: {e}")
        return None, None

plate_model, char_model = load_models()

# دالة تصحيح ميل اللوحة
def deskew_plate(plate_img):
    try:
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.bilateralFilter(gray, 9, 75, 75)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        coords = np.column_stack(np.where(thresh > 0))
        if len(coords) == 0:
            return plate_img
            
        coords = np.fliplr(coords) # تحويل من (y, x) إلى (x, y)
        rect = cv2.minAreaRect(coords)
        angle = rect[-1]
        
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
            
        if 1.0 < abs(angle) < 30.0:
            (h, w) = plate_img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(plate_img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            return rotated
    except Exception as e:
        st.sidebar.error(f"فشل تصحيح الميل: {e}")
    return plate_img

# دالة التحقق من صحة وقبول صيغة اللوحة المصرية
def is_valid_egyptian_plate(text: str) -> bool:
    if not (5 <= len(text) <= 7):
        return False
        
    allowed_chars_pattern = re.compile(r'^[أببتثجحخدذرزسشصضطظعغفقكلمنھوي0-9]+$')
    if not allowed_chars_pattern.match(text):
        return False
        
    letters = re.findall(r'[أببتثجحخدذرزسشصضطظعغفقكلمنھوي]', text)
    digits = re.findall(r'[0-9]', text)
    
    if len(letters) in [2, 3] and len(digits) in [3, 4]:
        return True
        
    return False

# تقسيم الحروف والأرقام لغرض العرض الجميل على اللوحة الافتراضية
def split_plate_text(text: str):
    letters = "".join(re.findall(r'[أببتثجحخدذرزسشصضطظعغفقكلمنھوي]', text))
    digits = "".join(re.findall(r'[0-9]', text))
    return letters, digits

# --- تصميم القائمة الجانبية (Sidebar) ---
st.sidebar.markdown("<h2 style='text-align: center; color: #0078D7;'>إعدادات التحكم</h2>", unsafe_allow_html=True)
st.sidebar.markdown("---")

plate_conf = st.sidebar.slider("حد ثقة كشف اللوحة (Plate Confidence)", 0.0, 1.0, 0.25, 0.05)
char_conf = st.sidebar.slider("حد ثقة قراءة الحروف (Character Confidence)", 0.0, 1.0, 0.25, 0.05)
apply_deskew = st.sidebar.checkbox("تفعيل تصحيح ميل اللوحة (Deskewing)", value=True)

st.sidebar.markdown("---")
st.sidebar.info(
    "💡 هذا النظام يقوم بالتعرف على السيارات واكتشاف لوحاتها المعدنية "
    "ثم استخلاص الأرقام والحروف ومطابقتها بالصيغ المعتمدة داخل جمهورية مصر العربية."
)

# --- الواجهة الرئيسية ---
st.markdown("<h1 class='main-title'>🏎️ نظام كشف وقراءة لوحات المرور المصرية</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>مشروع التخرج الذكي لمعالجة الصور والتعرف على النصوص لحظياً</p>", unsafe_allow_html=True)

# اختيار مصدر الصورة
source_option = st.radio(
    "اختر مصدر صورة السيارة:",
    ("رفع صورة من جهازك 📁", "التقاط صورة حية بالكاميرا 📷")
)

uploaded_file = None
camera_file = None

if "رفع صورة من جهازك 📁" in source_option:
    uploaded_file = st.file_uploader("قم باختيار صورة سيارة...", type=["jpg", "jpeg", "png"])
else:
    camera_file = st.camera_input("التقط صورة للوحة السيارة")

# تحديد الملف النشط للمعالجة
target_file = uploaded_file if uploaded_file is not None else camera_file

if target_file is not None:
    # فتح وقراءة الصورة
    image = Image.open(target_file)
    img_array = np.array(image)
    
    # تحويل من RGB (PIL) إلى BGR (OpenCV)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    st.markdown("---")
    
    # المعالجة
    with st.spinner("جاري كشف وقراءة اللوحة باستخدام الذكاء الاصطناعي..."):
        plate_detections = plate_model(img_bgr, conf=plate_conf, verbose=False)
        
        plates_found = False
        
        for p_res in plate_detections:
            if len(p_res.boxes) > 0:
                plates_found = True
                
                # إنشاء أعمدة لعرض تفاصيل النتائج
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown("<div class='card-container'><div class='card-title'>📍 كشف اللوحة في الصورة الأصلية</div></div>", unsafe_allow_html=True)
                    annotated_img = img_bgr.copy()
                    
                    for p_box in p_res.boxes:
                        bx1, by1, bx2, by2 = map(int, p_box.xyxy[0])
                        # رسم مربع حول اللوحة المكتشفة
                        cv2.rectangle(annotated_img, (bx1, by1), (bx2, by2), (0, 255, 0), 4)
                        
                        plate_crop = img_bgr[by1:by2, bx1:bx2]
                        
                        if plate_crop.size > 0:
                            # 1. تصحيح الميل إذا تم تفعيله
                            if apply_deskew:
                                processed_plate = deskew_plate(plate_crop)
                            else:
                                processed_plate = plate_crop
                            
                            # 2. كشف وقراءة الحروف
                            char_res = char_model(processed_plate, conf=char_conf, verbose=False)
                            detected_chars = []
                            annotated_plate = processed_plate.copy()
                            
                            for c_res in char_res:
                                for c_box in c_res.boxes:
                                    cx1, cy1, cx2, cy2 = map(int, c_box.xyxy[0])
                                    char_class = char_model.names[int(c_box.cls[0])]
                                    detected_chars.append((float(cx1), char_class))
                                    # رسم مربع حول كل حرف
                                    cv2.rectangle(annotated_plate, (cx1, cy1), (cx2, cy2), (255, 0, 0), 2)
                            
                            # ترتيب الحروف من اليسار لليمين بناءً على إحداثي X
                            detected_chars.sort(key=lambda x: x[0])
                            plate_text = "".join([c[1] for c in detected_chars])
                            
                    # تحويل الصورة المعلمة للرسم في Streamlit
                    annotated_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                    st.image(annotated_rgb, caption="مكان اللوحة المكتشفة", use_container_width=True)
                
                with col2:
                    st.markdown("<div class='card-container'><div class='card-title'>🔍 تفاصيل المعالجة والقراءة</div></div>", unsafe_allow_html=True)
                    
                    # عرض مراحل المعالجة
                    st.write("**صورة اللوحة المقصوصة (Crop):**")
                    st.image(cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB), use_container_width=True)
                    
                    if apply_deskew:
                        st.write("**صورة اللوحة بعد تصحيح الميل (Deskewed):**")
                        st.image(cv2.cvtColor(processed_plate, cv2.COLOR_BGR2RGB), use_container_width=True)
                    
                    st.write("**تحديد الحروف المكتشفة:**")
                    st.image(cv2.cvtColor(annotated_plate, cv2.COLOR_BGR2RGB), use_container_width=True)
                    
                    # عرض النص واللوحة الافتراضية
                    st.markdown("---")
                    st.write("**القراءة المبدئية للـ OCR:**")
                    st.code(plate_text if plate_text else "لا توجد حروف مكتشفة")
                    
                    # التحقق من صلاحية اللوحة المصرية الرسمية
                    if is_valid_egyptian_plate(plate_text):
                        st.success("✅ اللوحة مطابقة للمواصفات وصيغ المرور المصرية الرسمية!")
                        letters, digits = split_plate_text(plate_text)
                        
                        # رسم اللوحة التفاعلية
                        st.markdown(f"""
                            <div class='egypt-plate'>
                                <div class='egypt-plate-header'>
                                    <span>EGYPT</span>
                                    <span>مصر</span>
                                </div>
                                <div class='egypt-plate-body'>
                                    <span style='letter-spacing: 12px;'>{digits}</span>
                                    <div class='plate-divider'></div>
                                    <span style='letter-spacing: 12px;'>{letters}</span>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.warning("⚠️ النص المستخلص لا يطابق قواعد اللوحات المصرية الرسمية بالكامل (تأكد من وضوح الصورة وحد الثقة).")
                        
        if not plates_found:
            st.error("❌ لم يتمكن النظام من اكتشاف أي لوحة معدنية في الصورة. يرجى تعديل زاوية الالتقاط أو تقليل حد ثقة كشف اللوحة.")
