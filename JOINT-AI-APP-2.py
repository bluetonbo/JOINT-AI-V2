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
    
    .stApp {
        background-color: #090d16 !important;
        color: #e2e8f0 !important;
        font-family: 'Inter', sans-serif;
    }
    
    [data-testid="stSidebar"] {
        background-color: #0f1524 !important;
        border-right: 1px solid #1e293b;
        min-width: 360px !important;
    }
    
    h1, h2, h3, h4 {
        font-family: 'Inter', sans-serif;
        font-weight: 600 !important;
        letter-spacing: -0.01em;
    }
    
    .glass-card {
        background: #131b2e;
        border: 1px solid #223154;
        border-radius: 6px;
        padding: 12px 16px;
        margin-bottom: 12px;
    }
    
    .glass-card-title {
        color: #38bdf8;
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0px;
        padding: 2px 0;
    }

    .stButton>button, .stDownloadButton>button {
        height: 2.8rem !important;
        font-size: 0.9rem !important;
        border-radius: 4px !important;
        background: #10b981 !important;
        color: #ffffff !important;
        font-weight: 600;
        border: none !important;
        transition: all 0.2s ease;
        width: 100%;
    }
    .stButton>button:hover, .stDownloadButton>button:hover {
        background: #059669 !important;
    }
    
    div.stButton > button[data-testid="baseButton-secondary"] {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%) !important;
    }
    div.stButton > button[data-testid="baseButton-secondary"]:hover {
        background: linear-gradient(135deg, #60a5fa 0%, #2563eb 100%) !important;
    }

    .stNumberInput label, .stSlider label {
        color: #94a3b8 !important;
        font-weight: 500 !important;
        font-size: 0.82rem !important;
        margin-bottom: 2px !important;
    }
    
    button[data-baseweb="tab"] {
        font-size: 0.9rem !important;
        font-weight: 600 !important;
        height: 2.8rem !important;
        color: #64748b !important;
        background-color: transparent !important;
        border: none !important;
        padding: 0 16px !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #38bdf8 !important;
        border-bottom: 2px solid #38bdf8 !important;
    }
    
    .stAlert {
        background-color: #141f36 !important;
        border: 1px solid #1e293b !important;
        color: #cbd5e1 !important;
        padding: 10px 14px !important;
    }
    </style>
""", unsafe_allow_html=True)

# 3. 시스템 암호 인증 패널
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    _, center, _ = st.columns([1, 1.8, 1])
    with center:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("""
            <div class='glass-card' style='text-align: center; padding: 40px; margin-bottom: 25px;'>
                <h2 style='color: #10b981; margin-top: 0px; margin-bottom: 5px; font-size: 1.8rem;'>JOINT PROCESS INTELLIGENCE</h2>
                <p style='color: #64748b; font-size:0.9rem; margin-bottom: 0px;'>Core Optimization Dashboard</p>
            </div>
        """, unsafe_allow_html=True)
        pwd = st.text_input("Enter Password", type="password")
        if st.button("AUTHENTICATE SYSTEM"):
            if pwd == "admin1234":
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Invalid credentials. System access denied.")
    st.stop()

# 4. 세션 데이터 구조 초기화
if 'model_tq' not in st.session_state:
    st.session_state.update({
        'model_tq': None, 'model_ed': None, 'scaler': None, 'df_caulking': pd.DataFrame(),
        'process_vars': ['S_L', 'T_R', 'Caulking_Distance', 'Stud_Center', 'Aging_Status'],
        'optimizer_status': "STANDBY",
        'm_sl_min': 20.0, 'm_sl_max': 30.0, 'm_tr_min': 3.0, 'm_tr_max': 5.0,
        'm_cd_min': 4.0, 'm_cd_max': 7.0, 'm_sc_min': 1.5, 'm_sc_max': 3.5,
        'sim_sl': 25.0, 'sim_tr': 4.0, 'sim_cd': 5.5, 'sim_sc': 2.5,
        't_tq_min': 35.0, 't_tq_max': 37.0, 't_ed_min': 125000, 't_ed_max': 126000,
        'opt_result_x': None, 'opt_pred_tq': None, 'opt_pred_ed': None, 'confidence_score': None
    })

# 5. 사이드바 - 제어반
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

# 6. 메인 뷰포트
if st.session_state['model_tq']:
    h_left, h_right = st.columns([2, 1])
    with h_left:
        st.markdown("<h1 style='margin-bottom:0px; font-size:1.8rem;'>JOINT PROCESS INTELLIGENCE</h1>", unsafe_allow_html=True)
    with h_right:
        st.markdown(f"<div style='text-align:right;'>STATUS: {st.session_state['optimizer_status']}</div>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["QUALITY INVERSE TARGETING", "REAL-TIME WHAT-IF SIMULATOR", "FACTORY DATALAKE LOGS"])

    with tab1:
        layout_l, layout_r = st.columns([1, 1], gap="large")
        with layout_l:
            st.markdown("<div class='glass-card'><div class='glass-card-title'>Target Parameters</div>", unsafe_allow_html=True)
            st.session_state['t_tq_min'], st.session_state['t_tq_max'] = st.slider("Torque Range (Nm)", 20.0, 50.0, (st.session_state['t_tq_min'], st.session_state['t_tq_max']))
            st.session_state['t_ed_min'], st.session_state['t_ed_max'] = st.slider("Endurance Range (Cyc)", 50000, 200000, (int(st.session_state['t_ed_min']), int(st.session_state['t_ed_max'])))
            
            if st.button("RUN INVERSE INFERENCE SEARCH", type="secondary", use_container_width=True):
                X_vars = st.session_state['process_vars']
                def loss(x):
                    p_tq = st.session_state['model_tq'].predict(st.session_state['scaler'].transform([x]))[0]
                    return (p_tq - np.mean([st.session_state['t_tq_min'], st.session_state['t_tq_max']]))**2
                res = minimize(loss, [25, 4, 5, 2, 0], bounds=[(10,50),(1,10),(4,7),(1,3),(0,1)])
                st.session_state.update({'opt_result_x': res.x, 'opt_pred_tq': st.session_state['model_tq'].predict(st.session_state['scaler'].transform([res.x]))[0], 'confidence_score': 92.5})
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
        with layout_r:
            if st.session_state['opt_result_x'] is not None:
                x = st.session_state['opt_result_x']
                st.markdown(f"**Recommended**: S_L={x[0]:.2f}, T_R={x[1]:.2f}, CD={x[2]:.2f}, SC={x[3]:.2f}")
                st.metric("Predicted Torque", f"{st.session_state['opt_pred_tq']:.2f} Nm")

    with tab2:
        st.session_state['sim_sl'] = st.slider("Sim S_L", 10.0, 50.0, st.session_state['sim_sl'])
        st.session_state['sim_tr'] = st.slider("Sim T_R", 1.0, 10.0, st.session_state['sim_tr'])
        st.session_state['sim_cd'] = st.slider("Sim CD", 4.0, 7.0, st.session_state['sim_cd'])
        st.session_state['sim_sc'] = st.slider("Sim SC", 1.5, 3.5, st.session_state['sim_sc'])
        if st.button("EXECUTE SIMULATION"):
            X = pd.DataFrame([[st.session_state['sim_sl'], st.session_state['sim_tr'], st.session_state['sim_cd'], st.session_state['sim_sc'], 0]], columns=st.session_state['process_vars'])
            st.write(f"Estimated Torque: {st.session_state['model_tq'].predict(st.session_state['scaler'].transform(X))[0]:.2f}")

    with tab3:
        st.dataframe(st.session_state['df_caulking'])
