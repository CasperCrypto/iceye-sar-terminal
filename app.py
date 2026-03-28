import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import time
import pandas as pd
import json
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from streamlit_image_comparison import image_comparison
import io
from scipy.ndimage import uniform_filter

# -----------------------------------------------------------------------------
# Design System: ICEYE Professional Terminal Production (V12.5)
# -----------------------------------------------------------------------------
st.set_page_config(page_title="ICEYE ANALYSIS | TERMINAL", page_icon="🛰️", layout="wide", initial_sidebar_state="expanded")

# MONOCHROME LUX CSS (Shadcn-Dark Inspired)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Outfit:wght@500;600;700&display=swap');
    
    html, body, [data-testid="stAppViewContainer"] { background-color: #000000 !important; color: #FAFAFA; font-family: 'Inter', sans-serif; }
    [data-testid="stSidebar"] { background-color: rgba(9, 9, 11, 0.75) !important; border-right: 1px solid #27272A !important; backdrop-filter: blur(20px); }
    
    .brand-header { font-family: 'Outfit', sans-serif; font-weight: 700; font-size: 2.2rem; color: #FFFFFF; letter-spacing: -0.05em; margin-bottom: 8px; }
    .brand-subheader { font-family: 'Inter', sans-serif; color: #71717A; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.25em; margin-bottom: 50px; }
    .section-label { font-family: 'Inter', sans-serif; color: #A1A1AA; font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.15em; margin-bottom: 15px; }

    div[data-testid="stFileUploader"] section {
        background-color: rgba(255, 255, 255, 0.04) !important;
        border: 1px solid #27272A !important; border-radius: 0.75rem !important; padding: 4rem 2rem !important; backdrop-filter: blur(20px);
    }
    .stButton>button {
        background: #FFFFFF !important; color: #000000 !important; border-radius: 0.5rem !important;
        font-weight: 600 !important; font-size: 0.85rem !important; padding: 0.75rem 2.5rem !important;
        text-transform: uppercase; letter-spacing: 0.05em; transition: all 0.2s;
    }
    .stButton>button:hover { transform: scale(1.02); box-shadow: 0 0 25px rgba(255,255,255,0.1); }
    [data-testid="stMetricValue"] { font-family: 'Outfit', sans-serif; font-weight: 600; font-size: 1.8rem; }
    .stDataFrame { border: 1px solid #27272A !important; border-radius: 0.75rem; }
    .stTabs [data-baseweb="tab"] { padding: 12px 48px; color: #52525B; font-family: 'Outfit', sans-serif; font-size: 0.9rem; }
    .stTabs [aria-selected="true"] { color: #FFFFFF !important; border-bottom-color: #FFFFFF !important; }
    
    .status-dot { height: 8px; width: 8px; background-color: #FFFFFF; border-radius: 50%; display: inline-block; margin-right: 8px; box-shadow: 0 0 8px #FFFFFF; }
    .tech-hud { background: rgba(255,255,255,0.02); border: 1px solid #27272A; padding: 15px; border-radius: 8px; margin-top: 40px; }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# ENGINEARTIFACTS: ANALYSIS CORE
# -----------------------------------------------------------------------------
def get_hydro_mask(gray_img):
    try:
        blur = cv2.GaussianBlur(gray_img, (101, 101), 0)
        _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((25, 25), np.uint8))
    except: return np.zeros_like(gray_img)

def classify_shape(contour, l_m, w_m):
    try:
        area = cv2.contourArea(contour); hull = cv2.convexHull(contour); hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0; aspect_ratio = l_m / w_m if w_m > 0 else 0
        if l_m > 70: return "LARGE VESSEL" if aspect_ratio > 1.8 else "STRATEGIC OBJECT"
        if (15 < l_m < 60) and (solidity < 0.55): return "AIRCRAFT"
        if l_m < 15 and aspect_ratio < 1.3: return "VEHICLE"
        if aspect_ratio > 1.6: return "SHIP"
        return "CARGO / OBJECT"
    except: return "PROC ERR"

def lee_filter(img_array, size=5):
    try:
        img = img_array.astype(np.float32)
        img_mean = uniform_filter(img, (size, size)); img_sqr_mean = uniform_filter(img**2, (size, size))
        img_variance = img_sqr_mean - img_mean**2; over_var = np.var(img)
        weights = img_variance / (img_variance + over_var + 1e-5)
        return np.clip(img_mean + weights * (img - img_mean), 0, 255).astype(np.uint8)
    except: return img_array

def process_tiled(img_gray, gain, gsd, mask):
    try:
        h, w = img_gray.shape; tile_size = 640; overlap = 50; all_cnts = []
        _, th_global = cv2.threshold(img_gray, int(255-(gain*105)), 255, cv2.THRESH_BINARY)
        for y in range(0, h, tile_size - overlap):
            for x in range(0, w, tile_size - overlap):
                y2, x2 = min(y + tile_size, h), min(x + tile_size, w)
                tile_th = th_global[y:y2, x:x2]
                cnts, _ = cv2.findContours(tile_th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in cnts:
                    cnt[:, :, 0] += x; cnt[:, :, 1] += y; all_cnts.append(cnt)
        return all_cnts
    except: return []

# -----------------------------------------------------------------------------
# MISSION INFRASTRUCTURE
# -----------------------------------------------------------------------------
class MissionControl:
    BUCKET_NAME = "iceye-open-data-catalog"
    REGION = "us-west-2"
    def __init__(self):
        self.s3 = boto3.client('s3', region_name=self.REGION, config=Config(signature_version=UNSIGNED))
    @st.cache_data(ttl=3600)
    def fetch_image(_self, key):
        try:
            res = _self.s3.get_object(Bucket=_self.BUCKET_NAME, Key=key)
            return res['Body'].read()
        except: return None
    @st.cache_data(ttl=3600)
    def list_missions(_self):
        try:
            res = _self.s3.list_objects_v2(Bucket=_self.BUCKET_NAME, Delimiter='/')
            return [p.get('Prefix') for p in res.get('CommonPrefixes', [])]
        except: return []
    @st.cache_data(ttl=3600)
    def list_files(_self, prefix):
        try:
            res = _self.s3.list_objects_v2(Bucket=_self.BUCKET_NAME, Prefix=prefix, MaxKeys=100)
            return [{'Key': obj.get('Key'), 'Size': obj.get('Size')} for obj in res.get('Contents', []) if obj.get('Key').lower().endswith(('.tif', '.png', '.jpg'))]
        except: return []

mc = MissionControl()

# -----------------------------------------------------------------------------
# SIDEBAR: DATA SOURCE & FILTERS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<p class="brand-header" style="font-size: 1.2rem; margin-top:20px;">MISSION SELECT</p>', unsafe_allow_html=True)
    m_pfxs = mc.list_missions()
    if m_pfxs:
        s_ms = st.selectbox("CATEGORY", m_pfxs)
        s_fs = mc.list_files(s_ms)
        if s_fs:
            s_f = st.selectbox("DATASET", s_fs, format_func=lambda x: f"{x['Key'].split('/')[-1]} ({round(x['Size']/1024/1024,1)}MB)")
            if st.button("RUN STREAM"):
                st.session_state['remote_key'] = s_f['Key']
                st.session_state['remote_trigger'] = True

    st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)
    st.markdown('<p class="section-label">FILTERS</p>', unsafe_allow_html=True)
    h_on = st.toggle("WATER FILTER", value=True)
    s_on = st.toggle("HIGH RESOLUTION SCAN", value=True)
    l_on = st.toggle("NOISE REDUCTION", value=True)
    c_gn = st.slider("GAIN", 0.1, 1.0, 0.55, 0.05)
    g_rs = st.number_input("RESOLUTION SCALE", value=3.0, step=0.5)
    
    st.markdown('<div class="tech-hud">', unsafe_allow_html=True)
    st.markdown('<span class="status-dot"></span><span style="font-size: 0.7rem; color: #71717A; font-weight: 600;">MISSION CONNECTED</span>', unsafe_allow_html=True)
    st.markdown('<p style="font-size: 0.65rem; color: #52525B; margin-top: 10px;">CORE: V12.5-PASS<br>S3: READ-ONLY (PUBLIC)<br>STATUS: PRODUCTION READY</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# MAIN APP: ICEYE ANALYSIS TERMINAL
# -----------------------------------------------------------------------------
st.markdown('<p class="brand-header">ICEYE ANALYSIS TERMINAL</p>', unsafe_allow_html=True)
st.markdown('<p class="brand-subheader">Professional satellite analysis and structural variance</p>', unsafe_allow_html=True)

tabs = st.tabs(["DETECTION", "CHANGES", "OPTICAL"])

# -- TAB 1: DETECTION --
with tabs[0]:
    up1 = st.file_uploader("DROP SATELLITE IMAGE HERE", type=['jpg', 'png', 'tiff', 'tif', 'webp'], label_visibility="collapsed")
    img = None
    if up1: img = Image.open(up1).convert('RGB')
    elif st.session_state.get('remote_trigger'):
        with st.spinner("STREAMING FROM CATALOG..."):
            raw = mc.fetch_image(st.session_state.get('remote_key'))
            if raw: img = Image.open(io.BytesIO(raw)).convert('RGB')

    if img:
        st.markdown("<div style='height: 60px;'></div>", unsafe_allow_html=True)
        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.markdown('<p class="section-label">SAR IMAGE</p>', unsafe_allow_html=True)
            st.image(img, use_container_width=True)
        with col2:
            st.markdown('<p class="section-label">ANALYSIS OVERLAY</p>', unsafe_allow_html=True)
            with st.spinner("Processing satellite returns..."):
                start = time.time(); arr = np.array(img); gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
                if l_on: gray = lee_filter(gray)
                h_mask = get_hydro_mask(gray)
                if s_on: cnts = process_tiled(gray, c_gn, g_rs, h_mask)
                else: 
                    _, th = cv2.threshold(gray, int(255-(c_gn*105)), 255, cv2.THRESH_BINARY)
                    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                plot, data = arr.copy(), []
                for i, cnt in enumerate(cnts):
                    area = cv2.contourArea(cnt)
                    if 10 < area < 12000:
                        x, y, w, h = cv2.boundingRect(cnt)
                        cx, cy = min(x+w//2, gray.shape[1]-1), min(y+h//2, gray.shape[0]-1)
                        if h_on and h_mask[cy, cx] == 255: continue 
                        rect = cv2.minAreaRect(cnt); _, (sw, sh), ang = rect
                        l_m, w_m = max(sw, sh)*g_rs, min(sw, sh)*g_rs
                        cls = classify_shape(cnt, l_m, w_m)
                        m = {"ID": f"ICE-{100+i}", "CLASS": cls, "LENGTH_M": round(l_m, 1), "WIDTH_M": round(w_m, 1)}
                        data.append(m); cv2.rectangle(plot,(x,y),(x+w,y+h),(255,255,255), 2 if l_m > 40 else 1)
                        cv2.putText(plot, m['CLASS'], (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
                st.image(plot, use_container_width=True)
        if data:
            st.markdown("<div style='height: 60px;'></div>", unsafe_allow_html=True)
            st.markdown('<p class="section-label">OBJECT REPORT</p>', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(data), use_container_width=True)
            st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)
            m1, m2, m3 = st.columns(3)
            m1.metric("OBJECTS", len(data)); m2.metric("LATENCY", f"{(time.time()-start)*1000:.0f} ms"); m3.metric("INTELLIGENCE", "BASIC")
    else:
        st.markdown("<div style='height: 100px;'></div>", unsafe_allow_html=True)
        st.markdown('<p style="color: #52525B; font-size: 1.2rem; font-weight: 500; text-align: center;">UPLOAD SAR MISSION DATA OR SELECT FROM CATALOG TO BEGIN ANALYSIS</p>', unsafe_allow_html=True)

# -- TAB 2: CHANGES --
with tabs[1]:
    st.markdown('<p class="section-label">STRUCTURAL CHANGE ANALYSIS</p>', unsafe_allow_html=True)
    c1u, c2u = st.columns(2, gap="large")
    with c1u: f_a = st.file_uploader("BASELINE (T-1)", type=['jpg', 'png', 'tiff', 'tif', 'webp'], key="vA")
    with c2u: f_b = st.file_uploader("CURRENT (T-0)", type=['jpg', 'png', 'tiff', 'tif', 'webp'], key="vB")
    if f_a and f_b:
        with st.spinner("Analyzing structural shifts..."):
            ia, ib = np.array(Image.open(f_a).convert('RGB')), np.array(Image.open(f_b).convert('RGB'))
            if ia.shape != ib.shape: ia = cv2.resize(ia, (ib.shape[1], ib.shape[0]))
            ga, gb = cv2.cvtColor(ia, cv2.COLOR_RGB2GRAY), cv2.cvtColor(ib, cv2.COLOR_RGB2GRAY)
            dv = 50 + int((1.0 - c_gn)*50); _, ta = cv2.threshold(ga, dv, 255, cv2.THRESH_BINARY); _, tb = cv2.threshold(gb, dv, 255, cv2.THRESH_BINARY)
            dif = cv2.absdiff(ta, tb); kr = np.ones((25,25), np.uint8)
            dr = cv2.dilate(cv2.morphologyEx(dif, cv2.MORPH_CLOSE, kr), np.ones((5,5), np.uint8), iterations=1)
            cts, _ = cv2.findContours(dr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); ovl = ib.copy()
            for ct in cts:
                if cv2.contourArea(ct) > 30:
                    x, y, w, h = cv2.boundingRect(ct); cv2.rectangle(ovl, (x,y), (x+w,y+h), (255,255,255), 3)
            st.markdown("<div style='height: 60px;'></div>", unsafe_allow_html=True)
            c1v, c2v = st.columns(2, gap="large")
            with c1v: st.image(ia, use_container_width=True, caption="BASELINE")
            with c2v: st.image(ovl, use_container_width=True, caption="RE-ANALYSIS OVERLAY")
    else:
        st.markdown("<div style='height: 100px;'></div>", unsafe_allow_html=True)
        st.markdown('<p style="color: #52525B; font-size: 1.2rem; font-weight: 500; text-align: center;">UPLOAD BASELINE AND CURRENT DATA TO SCAN FOR STRUCTURAL VARIANCE</p>', unsafe_allow_html=True)

# -- TAB 3: OPTICAL --
with tabs[2]:
    st.markdown('<p class="section-label">OPTICAL TRANSLATION</p>', unsafe_allow_html=True)
    coL, coR = st.columns([1, 2], gap="large")
    with coL:
        omode = st.radio("OPTICAL MODE", ["NATURAL", "THERMAL"], label_visibility="visible")
        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
        fopt = st.file_uploader("UPLOAD RAW SAR", type=['jpg', 'png', 'tiff', 'tif', 'webp'], key="vO")
    with coR:
        if fopt:
            ao = np.array(Image.open(fopt).convert('L'))
            with st.spinner("Translating..."):
                clh = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)); gray = clh.apply(ao)
                h, w = gray.shape; syn = np.zeros((h, w, 3), dtype=np.uint8)
                if omode == "NATURAL":
                    syn[:,:,0] = np.where(gray < 60, np.clip(gray*1.8 + 30, 0, 255), gray*0.8)
                    syn[:,:,1] = np.where((gray >= 60) & (gray < 170), np.clip(gray*1.3 + 10, 0, 255), gray*0.9)
                    syn[:,:,2] = np.where(gray >= 170, np.clip(gray*1.1, 0, 255), gray*0.6)
                else:
                    syn = cv2.addWeighted(cv2.applyColorMap(gray, cv2.COLORMAP_JET), 0.7, cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), 0.3, 0)
                image_comparison(img1=Image.fromarray(ao), img2=Image.fromarray(cv2.cvtColor(cv2.bilateralFilter(syn,5,35,35), cv2.COLOR_BGR2RGB)), label1="RAW SAR", label2=f"{omode} OPTICAL", make_responsive=True, in_memory=True)
        else:
            st.markdown("<div style='height: 100px;'></div>", unsafe_allow_html=True)
            st.markdown('<p style="color: #52525B; font-size: 1.2rem; font-weight: 500; text-align: center;">UPLOAD RAW SAR TO GENERATE SYNTHETIC OPTICAL SPECTRA</p>', unsafe_allow_html=True)
