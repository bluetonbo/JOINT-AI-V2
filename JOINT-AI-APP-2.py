import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import minimize

# 페이지 및 스타일 설정
st.set_page_config(layout="wide", page_title="JOINT AI - Process Optimization Suite", page_icon="⚡")
st.markdown("""
    <style>
    .stApp { background-color: #090d16 !important; color: #e2e8f0 !important; font-family: 'Inter', sans-serif; }
    [data-testid="stSidebar"] { background-color: #0f1524 !important; }
    .glass-card { background: #131b2e; border: 1px solid #223154; border-radius: 6px; padding: 16px; margin-bottom: 20px; }
    .glass-card-title { color: #38bdf8; font-size: 0.9rem; font-weight: 700; margin-bottom: 15px; text-transform: uppercase; }
    </style>
""", unsafe_allow_html=True)

# 초기화
if 'model_tq' not in st.session_state:
    st.session_state.update({'model_tq': None, 'process_vars': ['S_L', 'T_R', 'Caulking_Distance', 'Stud_Center', 'Aging_Status'], 'opt_result': None})

# 사이드바
with st.sidebar:
    st.title("CONTROL CONSOLE")
    u_input = st.file_uploader("Upload Log File (CSV, XLSX)", type=['csv','xlsx'])
    if st.button("RUN ENGINE INITIALIZATION"):
        if u_input:
            df = pd.read_csv(u_input) if u_input.name.endswith('csv') else pd.read_excel(u_input)
            scaler = MinMaxScaler().fit(df[st.session_state['process_vars']])
            model = LinearRegression().fit(scaler.transform(df[st.session_state['process_vars']]), df['Torque'])
            st.session_state.update({'model_tq': model, 'scaler': scaler, 'df': df})
            st.rerun()

# 메인 UI
if st.session_state['model_tq']:
    st.title("JOINT PROCESS INTELLIGENCE")
    tab1, tab2, tab3 = st.tabs(["QUALITY INVERSE TARGETING", "REAL-TIME WHAT-IF SIMULATOR", "FACTORY DATALAKE LOGS"])
    
    with tab1:
        st.markdown("<div class='glass-card'><div class='glass-card-title'>BOUNDARY CONDITION OPTIMIZER</div>", unsafe_allow_html=True)
        st.write("Safety Bound Limit Mode: Auto Mode")
        st.info("Auto-Bound Enabled: S_L(10~50), T_R(1~5), CD(4~7), SC(1~4), Aging(0~1)")
        
        st.markdown("</div><div class='glass-card'><div class='glass-card-title'>TARGET QUALITY KPI RANGE</div>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        tq_min = col1.number_input("Target Torque Min", 30.0, 40.0, 35.0)
        tq_max = col2.number_input("Target Torque Max", 30.0, 40.0, 37.0)
        
        if st.button("RUN INVERSE INFERENCE SEARCH"):
            def loss(x):
                pred = st.session_state['model_tq'].predict(st.session_state['scaler'].transform([x]))[0]
                return (pred - ((tq_min + tq_max)/2))**2
            res = minimize(loss, x0=[30, 2, 5, 2, 0], bounds=[(10,50),(1,5),(4,7),(1,4),(0,1)])
            st.session_state['opt_result'] = res.x
        
        if st.session_state['opt_result'] is not None:
            res_df = pd.DataFrame([st.session_state['opt_result']], columns=st.session_state['process_vars'])
            st.dataframe(res_df.style.format("{:.2f}"), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        # 시뮬레이터 입력창들... (기존 구조와 동일하게 배치)
        st.markdown("</div>", unsafe_allow_html=True)
