import streamlit as st
import pandas as pd
import numpy as np
import io
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

# 3. 인증 및 초기화
if "authenticated" not in st.session_state: st.session_state.authenticated = False
if not st.session_state.authenticated:
    _, center, _ = st.columns([1, 1.8, 1])
    with center:
        st.markdown("<div class='glass-card' style='text-align: center; padding: 40px;'><h2 style='color: #10b981;'>JOINT PROCESS INTELLIGENCE</h2></div>", unsafe_allow_html=True)
        pwd = st.text_input("Enter Password", type="password")
        if st.button("AUTHENTICATE SYSTEM"):
            if pwd == "admin1234": st.session_state.authenticated = True; st.rerun()
    st.stop()

# 4. 세션 초기화 (S_L, T_R 변수 반영)
if 'model_tq' not in st.session_state:
    st.session_state.update({
        'model_tq': None, 'model_ed': None, 'scaler': None, 'df_caulking': pd.DataFrame(),
        'process_vars': ['Caulking_Distance', 'Stud_Center', 'Aging_Status', 'S_L', 'T_R'],
        'optimizer_status': "STANDBY", 'sim_cd': 5.5, 'sim_sc': 2.5, 'sim_sl': 5.0, 'sim_tr': 5.0,
        'target_tq_range': (35.0, 37.0), 'target_ed_range': (125000.0, 126000.0),
        'opt_result_x': None, 'opt_pred_tq': None, 'opt_pred_ed': None, 'confidence_score': None
    })

# 5. 사이드바
with st.sidebar:
    st.markdown("<h2 style='color: #ffffff; font-size:1.15rem;'>CONTROL CONSOLE</h2>", unsafe_allow_html=True)
    u_input = st.file_uploader("Upload Log File (CSV, XLSX)", type=['csv','xlsx'])
    if st.button("RUN ENGINE INITIALIZATION", type="primary"):
        if u_input:
            df = pd.read_csv(u_input) if u_input.name.endswith('csv') else pd.read_excel(u_input)
            X_vars = st.session_state['process_vars']
            scaler = MinMaxScaler().fit(df[X_vars])
            X_scaled = scaler.transform(df[X_vars])
            st.session_state.update({'model_tq': LinearRegression().fit(X_scaled, df['Torque']), 
                                     'model_ed': LinearRegression().fit(X_scaled, df['Endurance']),
                                     'scaler': scaler, 'df_caulking': df, 'optimizer_status': "ENGINE READY"})
            st.rerun()

# 6. 메인
if st.session_state['model_tq']:
    h_left, h_right = st.columns([2, 1])
    with h_right:
        st.markdown(f"<div style='text-align:right;'>{st.session_state['optimizer_status']}</div>", unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["QUALITY INVERSE TARGETING", "REAL-TIME WHAT-IF SIMULATOR", "FACTORY DATALAKE LOGS"])
    
    with tab1:
        if st.button("RUN INVERSE INFERENCE SEARCH", type="secondary"):
            def loss(x):
                pred = st.session_state['model_tq'].predict(st.session_state['scaler'].transform([x]))[0]
                return (pred - 36)**2
            res = minimize(loss, [5, 2, 0, 5, 5], method='SLSQP')
            st.session_state['opt_result_x'] = res.x
        if st.session_state['opt_result_x'] is not None:
            st.write(f"Result: {st.session_state['opt_result_x']}")

    with tab2:
        st.session_state['sim_cd'] = st.slider("CD", 0.0, 10.0, st.session_state['sim_cd'])
        st.session_state['sim_sc'] = st.slider("SC", 0.0, 10.0, st.session_state['sim_sc'])
        st.session_state['sim_sl'] = st.slider("S_L", 0.0, 10.0, st.session_state['sim_sl'])
        st.session_state['sim_tr'] = st.slider("T_R", 0.0, 10.0, st.session_state['sim_tr'])
        if st.button("EXECUTE PREDICTIVE SIMULATION"):
            query = [[st.session_state['sim_cd'], st.session_state['sim_sc'], 0, st.session_state['sim_sl'], st.session_state['sim_tr']]]
            st.session_state['sim_pred_tq'] = st.session_state['model_tq'].predict(st.session_state['scaler'].transform(query))[0]
            st.rerun()
            
    with tab3:
        st.dataframe(st.session_state['df_caulking'])
