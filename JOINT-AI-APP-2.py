import streamlit as st
import pandas as pd
import numpy as np
import io
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import minimize

# 1. 페이지 설정 (유지)
st.set_page_config(
    layout="wide", 
    page_title="JOINT AI - Process Optimization Suite",
    page_icon="⚡"
)

# 2. 미니멀 엔지니어링 콘솔 스타일 CSS (유지)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Inter:wght@400;500;600;700&display=swap');
    
    .stApp { background-color: #090d16 !important; color: #e2e8f0 !important; font-family: 'Inter', sans-serif; }
    [data-testid="stSidebar"] { background-color: #0f1524 !important; border-right: 1px solid #1e293b; min-width: 360px !important; }
    .glass-card { background: #131b2e; border: 1px solid #223154; border-radius: 6px; padding: 12px 16px; margin-bottom: 12px; }
    .glass-card-title { color: #38bdf8; font-size: 0.85rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 0px; padding: 2px 0; }
    .stButton>button { height: 2.8rem !important; font-size: 0.9rem !important; border-radius: 4px !important; background: #10b981 !important; color: #ffffff !important; font-weight: 600; border: none !important; width: 100%; }
    .stButton>button:hover { background: #059669 !important; }
    </style>
""", unsafe_allow_html=True)

# 3. 시스템 암호 인증 (유지)
if "authenticated" not in st.session_state: st.session_state.authenticated = False
if not st.session_state.authenticated:
    _, center, _ = st.columns([1, 1.8, 1])
    with center:
        pwd = st.text_input("Enter Password", type="password")
        if st.button("AUTHENTICATE SYSTEM"):
            if pwd == "admin1234": st.session_state.authenticated = True; st.rerun()
    st.stop()

# 4. 세션 데이터 구조 초기화 (S_L, T_R 추가 반영)
if 'model_tq' not in st.session_state:
    st.session_state.update({
        'model_tq': None, 'model_ed': None, 'scaler': None, 'df_caulking': pd.DataFrame(),
        'process_vars': ['S_L', 'T_R', 'Caulking_Distance', 'Stud_Center', 'Aging_Status'],
        'optimizer_status': "STANDBY",
        'sim_sl': 25.0, 'sim_tr': 4.0, 'sim_cd': 5.5, 'sim_sc': 2.5,
        'opt_result_x': None
    })

# 5. 사이드바 - 제어반
with st.sidebar:
    u_input = st.file_uploader("Upload Log File (CSV, XLSX)", type=['csv','xlsx'])
    if st.button("RUN ENGINE INITIALIZATION", type="primary"):
        if u_input:
            df = pd.read_csv(u_input) if u_input.name.endswith('csv') else pd.read_excel(u_input)
            df = df.dropna(subset=['Torque', 'Endurance'])
            X_list = st.session_state['process_vars']
            scaler = MinMaxScaler().fit(df[X_list])
            X_scaled = scaler.transform(df[X_list])
            st.session_state.update({
                'model_tq': LinearRegression().fit(X_scaled, df['Torque']),
                'model_ed': LinearRegression().fit(X_scaled, df['Endurance']),
                'scaler': scaler, 'df_caulking': df, 'optimizer_status': "ENGINE READY"
            })
            st.rerun()

# 6. 메인 로직 (최적화 및 시뮬레이션에 S_L, T_R 통합)
if st.session_state['model_tq']:
    tab1, tab2, tab3 = st.tabs(["QUALITY INVERSE TARGETING", "REAL-TIME WHAT-IF SIMULATOR", "FACTORY DATALAKE LOGS"])
    
    with tab1:
        if st.button("RUN INVERSE INFERENCE SEARCH"):
            def loss(x):
                s = st.session_state['scaler'].transform([x])
                tq = st.session_state['model_tq'].predict(s)[0]
                ed = st.session_state['model_ed'].predict(s)[0]
                return (tq - 36)**2 + ((ed - 125000)/1000)**2
            # 5개 변수 최적화 (S_L, T_R, CD, SC, AG)
            res = minimize(loss, [25, 4, 5, 2, 0], bounds=[(20,30), (3,5), (4,7), (1.5,3.5), (0,1)])
            st.session_state['opt_result_x'] = res.x
            st.rerun()
            
    with tab2:
        st.session_state['sim_sl'] = st.number_input("S_L", value=st.session_state['sim_sl'])
        st.session_state['sim_tr'] = st.number_input("T_R", value=st.session_state['sim_tr'])
        st.session_state['sim_cd'] = st.slider("CD", 4.0, 7.0, st.session_state['sim_cd'])
        st.session_state['sim_sc'] = st.slider("SC", 1.5, 3.5, st.session_state['sim_sc'])
        
        if st.button("EXECUTE PREDICTIVE SIMULATION"):
            vals = [st.session_state['sim_sl'], st.session_state['sim_tr'], 
                    st.session_state['sim_cd'], st.session_state['sim_sc'], 0]
            s = st.session_state['scaler'].transform([vals])
            st.write(f"Result Torque: {st.session_state['model_tq'].predict(s)[0]:.2f} Nm")

    with tab3:
        st.dataframe(st.session_state['df_caulking'])
