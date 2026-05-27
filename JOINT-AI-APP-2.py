import streamlit as st
import pandas as pd
import numpy as np
import io
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import minimize

# 1. 페이지 설정
st.set_page_config(layout="wide", page_title="JOINT AI - Process Optimization Suite", page_icon="⚡")

# 2. CSS 스타일
st.markdown("""
    <style>
    .stApp { background-color: #090d16 !important; color: #e2e8f0 !important; font-family: 'Inter', sans-serif; }
    [data-testid="stSidebar"] { background-color: #0f1524 !important; border-right: 1px solid #1e293b; min-width: 360px !important; }
    .glass-card { background: #131b2e; border: 1px solid #223154; border-radius: 6px; padding: 12px 16px; margin-bottom: 12px; }
    .glass-card-title { color: #38bdf8; font-size: 0.85rem; font-weight: 700; text-transform: uppercase; margin-bottom: 8px; }
    .stButton>button { height: 2.8rem !important; background: #10b981 !important; color: #ffffff !important; font-weight: 600; border: none !important; width: 100%; }
    </style>
""", unsafe_allow_html=True)

# 3. 인증 시스템
if "authenticated" not in st.session_state: st.session_state.authenticated = False
if not st.session_state.authenticated:
    _, center, _ = st.columns([1, 1.8, 1])
    with center:
        st.markdown("<div class='glass-card' style='text-align:center; padding:40px;'><h2>JOINT PROCESS INTELLIGENCE</h2></div>", unsafe_allow_html=True)
        pwd = st.text_input("Enter Password", type="password")
        if st.button("AUTHENTICATE SYSTEM"):
            if pwd == "admin1234": st.session_state.authenticated = True; st.rerun()
    st.stop()

# 4. 세션 초기화 (5개 변수 구조)
if 'model_tq' not in st.session_state:
    st.session_state.update({
        'model_tq': None, 'model_ed': None, 'scaler': None, 'df_caulking': pd.DataFrame(),
        'process_vars': ['Caulking_Distance', 'Stud_Center', 'Aging_Status', 'S_L', 'T_R'],
        'optimizer_status': "STANDBY",
        'sim_vals': {'Caulking_Distance': 5.5, 'Stud_Center': 2.5, 'Aging_Status': 0, 'S_L': 30.0, 'T_R': 15.0}
    })

# 5. 사이드바 - 제어반
with st.sidebar:
    st.markdown("## CONTROL CONSOLE")
    u_input = st.file_uploader("Upload Log File (CSV, XLSX)", type=['csv','xlsx'])
    if st.button("RUN ENGINE INITIALIZATION", type="primary"):
        if u_input:
            df = pd.read_csv(u_input) if u_input.name.endswith('csv') else pd.read_excel(u_input)
            required_cols = st.session_state['process_vars'] + ['Torque', 'Endurance']
            df_comb = df.dropna(subset=required_cols)
            X_list = st.session_state['process_vars']
            scaler = MinMaxScaler().fit(df_comb[X_list])
            X_scaled = scaler.transform(df_comb[X_list])
            st.session_state.update({
                'model_tq': LinearRegression().fit(X_scaled, df_comb['Torque']),
                'model_ed': LinearRegression().fit(X_scaled, df_comb['Endurance']),
                'scaler': scaler, 'df_caulking': df_comb, 'optimizer_status': "ENGINE READY"
            })
            st.rerun()

# 6. 메인 뷰포트
if st.session_state['model_tq']:
    tab1, tab2, tab3 = st.tabs(["QUALITY INVERSE TARGETING", "REAL-TIME WHAT-IF SIMULATOR", "FACTORY DATALAKE LOGS"])
    
    with tab1:
        st.markdown("<div class='glass-card'><div class='glass-card-title'>Multi-Variable Inverse Optimizer</div>", unsafe_allow_html=True)
        # 타겟 범위 입력 생략 (기존 로직 유지)
        if st.button("RUN 5-VAR INVERSE INFERENCE"):
            X_vars = st.session_state['process_vars']
            def loss_func(x):
                pred = st.session_state['model_tq'].predict(st.session_state['scaler'].transform(pd.DataFrame([x], columns=X_vars)))
                return (pred[0] - 36.0)**2 # 예시 타겟 36Nm
            
            res = minimize(loss_func, x0=[5.5, 2.5, 0, 30.0, 15.0], bounds=[(4,7), (1.5,3.5), (0,1), (10,50), (5,25)])
            st.session_state['opt_result_x'] = res.x
            st.rerun()

    with tab2:
        st.markdown("<div class='glass-card'><div class='glass-card-title'>Real-time Parameter Inputs</div>", unsafe_allow_html=True)
        cols = st.columns(5)
        sim_input = {}
        with cols[0]: sim_input['Caulking_Distance'] = st.number_input("CD", 4.0, 7.0, 5.5)
        with cols[1]: sim_input['Stud_Center'] = st.number_input("SC", 1.5, 3.5, 2.5)
        with cols[2]: sim_input['Aging_Status'] = st.selectbox("Aging", [0, 1])
        with cols[3]: sim_input['S_L'] = st.number_input("S_L", 10.0, 50.0, 30.0)
        with cols[4]: sim_input['T_R'] = st.number_input("T_R", 5.0, 25.0, 15.0)
        
        if st.button("EXECUTE SIMULATION"):
            df_q = pd.DataFrame([sim_input])[st.session_state['process_vars']]
            pred = st.session_state['model_tq'].predict(st.session_state['scaler'].transform(df_q))
            st.metric("Predicted Torque", f"{pred[0]:.2f} Nm")

    with tab3:
        st.dataframe(st.session_state['df_caulking'], use_container_width=True)
else:
    st.info("System Ready. Upload data via Sidebar.")
