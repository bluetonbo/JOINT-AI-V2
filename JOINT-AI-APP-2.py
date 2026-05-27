import streamlit as st
import pandas as pd
import numpy as np
import io
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import minimize

# 1. 페이지 설정 및 CSS (원본 전체 유지)
st.set_page_config(layout="wide", page_title="JOINT AI - Process Optimization Suite", page_icon="⚡")
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Inter:wght@400;500;600;700&display=swap');
    .stApp { background-color: #090d16 !important; color: #e2e8f0 !important; font-family: 'Inter', sans-serif; }
    [data-testid="stSidebar"] { background-color: #0f1524 !important; border-right: 1px solid #1e293b; min-width: 360px !important; }
    .glass-card { background: #131b2e; border: 1px solid #223154; border-radius: 6px; padding: 12px 16px; margin-bottom: 12px; }
    .glass-card-title { color: #38bdf8; font-size: 0.85rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 0px; padding: 2px 0; }
    .stButton>button { height: 2.8rem !important; font-size: 0.9rem !important; border-radius: 4px !important; background: #10b981 !important; color: #ffffff !important; font-weight: 600; border: none !important; width: 100%; }
    </style>
""", unsafe_allow_html=True)

# 2. 세션 상태 초기화 (S_L, T_R 반영)
if 'model_tq' not in st.session_state:
    st.session_state.update({
        'model_tq': None, 'model_ed': None, 'scaler': None, 'df_caulking': pd.DataFrame(),
        'process_vars': ['S_L', 'T_R', 'Caulking_Distance', 'Stud_Center', 'Aging_Status'],
        'optimizer_status': "STANDBY",
        'opt_result_x': None, 'opt_pred_tq': None, 'opt_pred_ed': None, 'confidence_score': None
    })

# 3. 사이드바 - 제어반
with st.sidebar:
    u_input = st.file_uploader("Upload Log File (CSV, XLSX)", type=['csv','xlsx'])
    if st.button("RUN ENGINE INITIALIZATION"):
        if u_input:
            df = pd.read_csv(u_input) if u_input.name.endswith('csv') else pd.read_excel(u_input)
            X = df[st.session_state['process_vars']]
            scaler = MinMaxScaler().fit(X)
            X_scaled = scaler.transform(X)
            model_tq = LinearRegression().fit(X_scaled, df['Torque'])
            model_ed = LinearRegression().fit(X_scaled, df['Endurance'])
            st.session_state.update({'model_tq': model_tq, 'model_ed': model_ed, 'scaler': scaler, 'df_caulking': df, 'optimizer_status': "READY"})
            st.rerun()

# 4. 메인 뷰포트 - 모든 UI 요소 복원
if st.session_state['model_tq'] is not None:
    st.markdown("<h1 style='font-size:1.8rem;'>JOINT PROCESS INTELLIGENCE</h1>", unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["QUALITY INVERSE TARGETING", "REAL-TIME WHAT-IF SIMULATOR", "FACTORY DATALAKE LOGS"])
    
    with tab1:
        st.markdown("<div class='glass-card'><div class='glass-card-title'>Optimization Controls</div>", unsafe_allow_html=True)
        if st.button("RUN INVERSE INFERENCE SEARCH"):
            # 최적화 로직 (5개 변수 반영)
            def loss(x):
                pred_tq = st.session_state['model_tq'].predict(st.session_state['scaler'].transform([x]))[0]
                return (pred_tq - 36.0)**2
            res = minimize(loss, x0=[30, 2, 5, 2, 0], bounds=[(10,50), (1,5), (4,7), (1,4), (0,1)])
            st.session_state.update({'opt_result_x': res.x})
        
        if st.session_state['opt_result_x'] is not None:
            st.write(f"최적 변수 값: {st.session_state['opt_result_x']}")
        st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        s_sl = st.number_input("S_L", 10.0, 50.0, 30.0)
        s_tr = st.number_input("T_R", 1.0, 5.0, 2.0)
        s_cd = st.number_input("Caulking_Distance", 4.0, 7.0, 5.0)
        s_sc = st.number_input("Stud_Center", 1.0, 4.0, 2.0)
        s_ag = st.selectbox("Aging_Status", [0, 1])
        if st.button("EXECUTE PREDICTIVE SIMULATION"):
            val = st.session_state['model_tq'].predict(st.session_state['scaler'].transform([[s_sl, s_tr, s_cd, s_sc, s_ag]]))
            st.metric("Predicted Torque", f"{val[0]:.2f} Nm")
        st.markdown("</div>", unsafe_allow_html=True)

    with tab3:
        st.dataframe(st.session_state['df_caulking'])

else:
    st.info("데이터를 업로드하고 초기화를 진행해 주세요.")
