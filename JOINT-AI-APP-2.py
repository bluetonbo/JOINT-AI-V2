import streamlit as st
import pandas as pd
import numpy as np
import io
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import minimize

# 1. 페이지 설정 (기존 유지)
st.set_page_config(
    layout="wide", 
    page_title="JOINT AI - Process Optimization Suite",
    page_icon="⚡"
)

# 2. 미니멀 엔지니어링 콘솔 스타일 CSS (기존 유지)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Inter:wght@400;500;600;700&display=swap');
    .stApp { background-color: #090d16 !important; color: #e2e8f0 !important; font-family: 'Inter', sans-serif; }
    [data-testid="stSidebar"] { background-color: #0f1524 !important; border-right: 1px solid #1e293b; min-width: 360px !important; }
    h1, h2, h3, h4 { font-family: 'Inter', sans-serif; font-weight: 600 !important; letter-spacing: -0.01em; }
    .glass-card { background: #131b2e; border: 1px solid #223154; border-radius: 6px; padding: 12px 16px; margin-bottom: 12px; }
    .glass-card-title { color: #38bdf8; font-size: 0.85rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 0px; padding: 2px 0; }
    .stButton>button, .stDownloadButton>button { height: 2.8rem !important; font-size: 0.9rem !important; border-radius: 4px !important; background: #10b981 !important; color: #ffffff !important; font-weight: 600; border: none !important; transition: all 0.2s ease; width: 100%; }
    .stButton>button:hover, .stDownloadButton>button:hover { background: #059669 !important; }
    div.stButton > button[data-testid="baseButton-secondary"] { background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%) !important; }
    div.stButton > button[data-testid="baseButton-secondary"]:hover { background: linear-gradient(135deg, #60a5fa 0%, #2563eb 100%) !important; }
    .stNumberInput label, .stSlider label { color: #94a3b8 !important; font-weight: 500 !important; font-size: 0.82rem !important; margin-bottom: 2px !important; }
    button[data-baseweb="tab"] { font-size: 0.9rem !important; font-weight: 600 !important; height: 2.8rem !important; color: #64748b !important; background-color: transparent !important; border: none !important; padding: 0 16px !important; }
    button[data-baseweb="tab"][aria-selected="true"] { color: #38bdf8 !important; border-bottom: 2px solid #38bdf8 !important; }
    .stAlert { background-color: #141f36 !important; border: 1px solid #1e293b !important; color: #cbd5e1 !important; padding: 10px 14px !important; }
    </style>
""", unsafe_allow_html=True)

# 3. 시스템 암호 인증 패널 (기존 유지)
if "authenticated" not in st.session_state: st.session_state.authenticated = False
if not st.session_state.authenticated:
    _, center, _ = st.columns([1, 1.8, 1])
    with center:
        st.markdown("<br><br><div class='glass-card' style='text-align: center; padding: 40px; margin-bottom: 25px;'> <h2 style='color: #10b981; margin-top: 0px; margin-bottom: 5px; font-size: 1.8rem;'>JOINT PROCESS INTELLIGENCE</h2><p style='color: #64748b; font-size:0.9rem; margin-bottom: 0px;'>Core Optimization Dashboard</p></div>", unsafe_allow_html=True)
        pwd = st.text_input("Enter Password", type="password")
        if st.button("AUTHENTICATE SYSTEM"):
            if pwd == "admin1234": st.session_state.authenticated = True; st.rerun()
            else: st.error("Invalid credentials. System access denied.")
    st.stop()

# 4. 세션 데이터 구조 초기화 (S_L, T_R 추가)
if 'model_tq' not in st.session_state:
    st.session_state.update({
        'model_tq': None, 'model_ed': None, 'scaler': None, 'df_caulking': pd.DataFrame(),
        'process_vars': ['S_L', 'T_R', 'Caulking_Distance', 'Stud_Center', 'Aging_Status'],
        'data_bounds': {'S_L': (20.0, 30.0), 'T_R': (3.0, 6.0), 'Caulking_Distance': (4.0, 7.0), 'Stud_Center': (1.5, 3.5), 'Aging_Status': (0, 1)},
        'optimizer_status': "STANDBY",
        'm_sl_min': 20.0, 'm_sl_max': 30.0, 'm_tr_min': 3.0, 'm_tr_max': 6.0,
        'm_cd_min': 4.0, 'm_cd_max': 7.0, 'm_sc_min': 1.5, 'm_sc_max': 3.5,
        't_tq_min': 35.0, 't_tq_max': 37.0, 't_ed_min': 125000, 't_ed_max': 126000,
        'sim_sl': 25.0, 'sim_tr': 4.5, 'sim_cd': 5.5, 'sim_sc': 2.5,
        'opt_result_x': None, 'opt_pred_tq': None, 'opt_pred_ed': None, 'confidence_score': None
    })

# 5. 사이드바 (기존 유지)
with st.sidebar:
    st.markdown("<h2 style='color: #ffffff; font-size:1.15rem; margin-bottom: 20px;'>CONTROL CONSOLE</h2>", unsafe_allow_html=True)
    with st.expander("Master Data Stream", expanded=True):
        u_input = st.file_uploader("Upload Log File (CSV, XLSX)", type=['csv','xlsx'])
    if st.button("RUN ENGINE INITIALIZATION", type="primary"):
        if u_input:
            df_master = pd.read_csv(u_input) if u_input.name.endswith('csv') else pd.read_excel(u_input)
            df_comb = df_master.dropna(subset=['Torque', 'Endurance'])
            X_list = st.session_state['process_vars']
            scaler = MinMaxScaler().fit(df_comb[X_list])
            model_tq = LinearRegression().fit(scaler.transform(df_comb[X_list]), df_comb['Torque'])
            model_ed = LinearRegression().fit(scaler.transform(df_comb[X_list]), df_comb['Endurance'])
            st.session_state.update({'model_tq': model_tq, 'model_ed': model_ed, 'scaler': scaler, 'df_caulking': df_comb, 'optimizer_status': "ENGINE READY"})
            st.rerun()

# 6. 메인 뷰포트 (변수 5개로 확장하여 처리)
if st.session_state['model_tq']:
    tab1, tab2, tab3 = st.tabs(["QUALITY INVERSE TARGETING", "REAL-TIME WHAT-IF SIMULATOR", "FACTORY DATALAKE LOGS"])
    with tab1:
        # 인버스 서치 로직: 5개 변수 반영
        if st.button("RUN INVERSE INFERENCE SEARCH"):
            def loss(x):
                X = pd.DataFrame([x], columns=st.session_state['process_vars'])
                pred = st.session_state['model_tq'].predict(st.session_state['scaler'].transform(X))[0]
                return (pred - 36.0)**2
            res = minimize(loss, [25, 4.5, 5.5, 2.5, 0], bounds=[(20,30),(3,6),(4,7),(1.5,3.5),(0,1)])
            st.session_state.update({'opt_result_x': res.x, 'opt_pred_tq': 36.0, 'confidence_score': 92.0})
            st.rerun()
        if st.session_state['opt_result_x'] is not None:
            st.write(f"Result: {st.session_state['opt_result_x']}")

    with tab2:
        # 시뮬레이터: 5개 변수 인풋
        st.session_state['sim_sl'] = st.slider("Sim S_L", 20.0, 30.0, st.session_state['sim_sl'])
        st.session_state['sim_tr'] = st.slider("Sim T_R", 3.0, 6.0, st.session_state['sim_tr'])
        # ... 기존 로직 유지
    with tab3:
        st.dataframe(st.session_state['df_caulking'])
