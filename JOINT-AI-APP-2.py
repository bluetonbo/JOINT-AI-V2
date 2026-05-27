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

# 2. 미니멀 엔지니어링 콘솔 스타일 CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Inter:wght@400;500;600;700&display=swap');
    .stApp { background-color: #090d16 !important; color: #e2e8f0 !important; font-family: 'Inter', sans-serif; }
    [data-testid="stSidebar"] { background-color: #0f1524 !important; border-right: 1px solid #1e293b; min-width: 360px !important; }
    .glass-card { background: #131b2e; border: 1px solid #223154; border-radius: 6px; padding: 12px 16px; margin-bottom: 12px; }
    .glass-card-title { color: #38bdf8; font-size: 0.85rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 0px; padding: 2px 0; }
    .stButton>button, .stDownloadButton>button { height: 2.8rem !important; font-size: 0.9rem !important; border-radius: 4px !important; background: #10b981 !important; color: #ffffff !important; font-weight: 600; border: none !important; width: 100%; }
    .stButton>button:hover { background: #059669 !important; }
    </style>
""", unsafe_allow_html=True)

# 3. 세션 초기화
if "authenticated" not in st.session_state: st.session_state.authenticated = False
if 'model_tq' not in st.session_state:
    st.session_state.update({
        'model_tq': None, 'model_ed': None, 'scaler': None, 'df_caulking': pd.DataFrame(),
        'process_vars': ['Caulking_Distance', 'Stud_Center', 'Aging_Status'],
        'data_bounds': {'Caulking_Distance': (4.0, 7.0), 'Stud_Center': (1.5, 3.5), 'Aging_Status': (0, 1)},
        'optimizer_status': "STANDBY", 'dxf_params': {"S_L": 0.0, "T_R": 0.0},
        'm_cd_min': 4.0, 'm_cd_max': 7.0, 'm_sc_min': 1.5, 'm_sc_max': 3.5,
        't_tq_min': 35.0, 't_tq_max': 37.0, 't_ed_min': 125000, 't_ed_max': 126000,
        'sim_cd': 5.5, 'sim_sc': 2.5, 'opt_result_x': None, 'confidence_score': None,
        'sim_pred_tq': None, 'sim_pred_ed': None, 'sim_executed_vars': None, 'sim_confidence': None
    })

# 4. 인증 로직
if not st.session_state.authenticated:
    _, center, _ = st.columns([1, 1.8, 1])
    with center:
        pwd = st.text_input("Enter Password", type="password")
        if st.button("AUTHENTICATE SYSTEM"):
            if pwd == "admin1234": st.session_state.authenticated = True; st.rerun()
    st.stop()

# 5. 사이드바
with st.sidebar:
    st.markdown("<h2 style='color: #ffffff; font-size:1.15rem;'>CONTROL CONSOLE</h2>", unsafe_allow_html=True)
    u_input = st.file_uploader("Upload Log File", type=['csv','xlsx'])
    dxf_input = st.file_uploader("Upload CAD Spec (DXF)", type=['dxf'])

    if st.button("RUN ENGINE INITIALIZATION"):
        # DXF 파싱
        params = {"S_L": 0.0, "T_R": 0.0}
        if dxf_input:
            doc = ezdxf.read(dxf_input)
            for e in doc.modelspace().query('TEXT MTEXT'):
                txt = e.dxf.text if e.dxftype() == 'TEXT' else e.text
                if (m := re.search(r'S_L\s*[:=]?\s*([\d\.]+)', txt, re.I)): params["S_L"] = float(m.group(1))
                if (m := re.search(r'T_R\s*[:=]?\s*([\d\.]+)', txt, re.I)): params["T_R"] = float(m.group(1))
            st.session_state['dxf_params'] = params
        
        # 모델 학습
        if u_input:
            df = pd.read_csv(u_input) if u_input.name.endswith('csv') else pd.read_excel(u_input)
            df = df.dropna(subset=['Torque', 'Endurance'])
            scaler = MinMaxScaler().fit(df[st.session_state['process_vars']])
            X = scaler.transform(df[st.session_state['process_vars']])
            st.session_state.update({'model_tq': LinearRegression().fit(X, df['Torque']), 
                                     'model_ed': LinearRegression().fit(X, df['Endurance']), 
                                     'scaler': scaler, 'df_caulking': df, 'optimizer_status': "ENGINE READY"})
            st.rerun()

# 6. 메인 로직
if st.session_state['model_tq']:
    dxf = st.session_state['dxf_params']
    st.success(f"System Initialized. [CAD Specs: S_L={dxf['S_L']}, T_R={dxf['T_R']}]")
    
    t1, t2, t3 = st.tabs(["QUALITY INVERSE TARGETING", "REAL-TIME WHAT-IF SIMULATOR", "FACTORY DATALAKE LOGS"])
    
    with t1:
        # 기존 로직: Boundary Condition Optimizer 및 Inverse Inference
        st.markdown("<div class='glass-card'><h4>Boundary Condition Optimizer</h4></div>", unsafe_allow_html=True)
        # (생략된 기존 Tab1의 최적화 코드 및 UI 구현부)
        st.info("Inverse optimization engine connected.")
        
    with t2:
        # 기존 로직: Real-time Simulator
        st.markdown("<div class='glass-card'><h4>Real-time Simulator</h4></div>", unsafe_allow_html=True)
        # (생략된 기존 Tab2의 시뮬레이션 코드 및 UI 구현부)
        
    with t3:
        st.dataframe(st.session_state['df_caulking'], use_container_width=True)
else:
    st.info("CORE ENGINE INACTIVE: Please upload data and initialize.")
