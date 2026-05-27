import streamlit as st
import pandas as pd
import numpy as np
import io
import ezdxf
import re
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import minimize

# 1. 페이지 설정
st.set_page_config(
    layout="wide", 
    page_title="JOINT AI - Process Optimization Suite",
    page_icon="⚡"
)

# 2. CSS 스타일
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Inter:wght@400;500;600;700&display=swap');
    .stApp { background-color: #090d16 !important; color: #e2e8f0 !important; font-family: 'Inter', sans-serif; }
    [data-testid="stSidebar"] { background-color: #0f1524 !important; border-right: 1px solid #1e293b; min-width: 360px !important; }
    h1, h2, h3, h4 { font-family: 'Inter', sans-serif; font-weight: 600 !important; letter-spacing: -0.01em; }
    .glass-card { background: #131b2e; border: 1px solid #223154; border-radius: 6px; padding: 12px 16px; margin-bottom: 12px; }
    .glass-card-title { color: #38bdf8; font-size: 0.85rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 0px; padding: 2px 0; }
    .stButton>button, .stDownloadButton>button { height: 2.8rem !important; font-size: 0.9rem !important; border-radius: 4px !important; background: #10b981 !important; color: #ffffff !important; font-weight: 600; border: none !important; width: 100%; }
    .stButton>button:hover { background: #059669 !important; }
    .stNumberInput label, .stSlider label { color: #94a3b8 !important; font-weight: 500 !important; font-size: 0.82rem !important; }
    button[data-baseweb="tab"] { font-size: 0.9rem !important; font-weight: 600 !important; height: 2.8rem !important; color: #64748b !important; }
    button[data-baseweb="tab"][aria-selected="true"] { color: #38bdf8 !important; border-bottom: 2px solid #38bdf8 !important; }
    </style>
""", unsafe_allow_html=True)

# 3. 인증 패널
if "authenticated" not in st.session_state: st.session_state.authenticated = False
if not st.session_state.authenticated:
    _, center, _ = st.columns([1, 1.8, 1])
    with center:
        st.markdown("<br><br><div class='glass-card' style='text-align: center; padding: 40px;'><h2 style='color: #10b981;'>JOINT PROCESS INTELLIGENCE</h2></div>", unsafe_allow_html=True)
        pwd = st.text_input("Enter Password", type="password")
        if st.button("AUTHENTICATE SYSTEM"):
            if pwd == "admin1234": st.session_state.authenticated = True; st.rerun()
            else: st.error("Invalid credentials.")
    st.stop()

# 4. 세션 초기화
if 'model_tq' not in st.session_state:
    st.session_state.update({
        'model_tq': None, 'model_ed': None, 'scaler': None, 'df_caulking': pd.DataFrame(),
        'process_vars': ['Caulking_Distance', 'Stud_Center', 'Aging_Status'],
        'data_bounds': {'Caulking_Distance': (4.0, 7.0), 'Stud_Center': (1.5, 3.5), 'Aging_Status': (0, 1)},
        'optimizer_status': "STANDBY", 'dxf_params': {"S_L": 0.0, "T_R": 0.0},
        'm_cd_min': 4.0, 'm_cd_max': 7.0, 'm_sc_min': 1.5, 'm_sc_max': 3.5,
        't_tq_min': 35.0, 't_tq_max': 37.0, 't_ed_min': 125000, 't_ed_max': 126000,
        'sim_cd': 5.5, 'sim_sc': 2.5, 'opt_result_x': None, 'confidence_score': None
    })

# 5. 사이드바
with st.sidebar:
    st.markdown("<h2 style='color: #ffffff; font-size:1.15rem; margin-bottom: 20px;'>CONTROL CONSOLE</h2>", unsafe_allow_html=True)
    with st.expander("Master Data Stream", expanded=True):
        u_input = st.file_uploader("Upload Log File (CSV, XLSX)", type=['csv','xlsx'])
        dxf_input = st.file_uploader("Upload CAD Spec (DXF)", type=['dxf'])

    if st.button("RUN ENGINE INITIALIZATION", type="primary"):
        # DXF 파싱
        params = {"S_L": 0.0, "T_R": 0.0}
        if dxf_input:
            try:
                doc = ezdxf.read(dxf_input)
                msp = doc.modelspace()
                for entity in msp.query('TEXT MTEXT'):
                    txt = entity.dxf.text if entity.dxftype() == 'TEXT' else entity.text
                    match_sl = re.search(r'S_L\s*[:=]?\s*([\d\.]+)', txt, re.IGNORECASE)
                    match_tr = re.search(r'T_R\s*[:=]?\s*([\d\.]+)', txt, re.IGNORECASE)
                    if match_sl: params["S_L"] = float(match_sl.group(1))
                    if match_tr: params["T_R"] = float(match_tr.group(1))
                st.session_state['dxf_params'] = params
            except Exception as e: st.error(f"DXF Parsing Error: {e}")

        # 데이터 로드
        if u_input:
            df_master = pd.read_csv(u_input) if u_input.name.endswith('csv') else pd.read_excel(u_input)
            df_comb = df_master.dropna(subset=['Torque', 'Endurance'])
            X_list = st.session_state['process_vars']
            scaler = MinMaxScaler().fit(df_comb[X_list])
            model_tq = LinearRegression().fit(scaler.transform(df_comb[X_list]), df_comb['Torque'])
            model_ed = LinearRegression().fit(scaler.transform(df_comb[X_list]), df_comb['Endurance'])
            st.session_state.update({'model_tq': model_tq, 'model_ed': model_ed, 'scaler': scaler, 'df_caulking': df_comb, 'optimizer_status': "ENGINE READY"})
            st.rerun()

# 6. 메인 뷰포트
if st.session_state['model_tq']:
    dxf = st.session_state['dxf_params']
    st.success(f"System Initialized. [CAD Specs: S_L={dxf['S_L']}, T_R={dxf['T_R']}]")
    
    tab1, tab2, tab3 = st.tabs(["QUALITY INVERSE TARGETING", "REAL-TIME WHAT-IF SIMULATOR", "FACTORY DATALAKE LOGS"])
    
    with tab1:
        st.write("기존 Tab1 로직을 여기에 그대로 붙여넣으세요.")
    with tab2:
        st.write("기존 Tab2 로직을 여기에 그대로 붙여넣으세요.")
    with tab3:
        st.dataframe(st.session_state['df_caulking'], use_container_width=True)
else:
    st.info("CORE ENGINE INACTIVE: Please upload data and CAD logs.")
