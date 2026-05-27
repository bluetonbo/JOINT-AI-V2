import streamlit as st
import pandas as pd
import numpy as np
import io
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import minimize

# 1. 페이지 설정
st.set_page_config(layout="wide", page_title="JOINT AI - Process Optimization Suite", page_icon="⚡")

# 2. CSS (원본 유지)
st.markdown("""
    <style>
    .stApp { background-color: #090d16 !important; color: #e2e8f0 !important; font-family: 'Inter', sans-serif; }
    [data-testid="stSidebar"] { background-color: #0f1524 !important; border-right: 1px solid #1e293b; min-width: 360px !important; }
    .glass-card { background: #131b2e; border: 1px solid #223154; border-radius: 6px; padding: 12px 16px; margin-bottom: 12px; }
    .glass-card-title { color: #38bdf8; font-size: 0.85rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 0px; padding: 2px 0; }
    .stButton>button { height: 2.8rem !important; font-size: 0.9rem !important; border-radius: 4px !important; background: #10b981 !important; color: #ffffff !important; font-weight: 600; border: none !important; width: 100%; }
    .stNumberInput label, .stSlider label { color: #94a3b8 !important; font-weight: 500 !important; font-size: 0.82rem !important; margin-bottom: 2px !important; }
    </style>
""", unsafe_allow_html=True)

# 3. 인증 (원본 유지)
if "authenticated" not in st.session_state: st.session_state.authenticated = False
if not st.session_state.authenticated:
    _, center, _ = st.columns([1, 1.8, 1])
    with center:
        pwd = st.text_input("Enter Password", type="password")
        if st.button("AUTHENTICATE SYSTEM"):
            if pwd == "admin1234": st.session_state.authenticated = True; st.rerun()
    st.stop()

# 4. 세션 초기화 (S_L, T_R 추가)
if 'model_tq' not in st.session_state:
    st.session_state.update({
        'model_tq': None, 'model_ed': None, 'scaler': None, 'df_caulking': pd.DataFrame(),
        'process_vars': ['Caulking_Distance', 'Stud_Center', 'Aging_Status', 'S_L', 'T_R'],
        'data_bounds': {'Caulking_Distance': (4.0, 7.0), 'Stud_Center': (1.5, 3.5), 'Aging_Status': (0, 1), 'S_L': (10.0, 50.0), 'T_R': (5.0, 25.0)},
        'optimizer_status': "STANDBY",
        'm_cd_min': 4.0, 'm_cd_max': 7.0, 'm_sc_min': 1.5, 'm_sc_max': 3.5,
        't_tq_min': 35.0, 't_tq_max': 37.0, 't_ed_min': 125000, 't_ed_max': 126000,
        'sim_cd': 5.5, 'sim_sc': 2.5, 'sim_sl': 30.0, 'sim_tr': 15.0,
        'target_tq_range': (35.0, 37.0), 'target_ed_range': (125000.0, 126000.0),
        'opt_result_x': None, 'opt_pred_tq': None, 'opt_pred_ed': None, 'confidence_score': None,
        'sim_pred_tq': None, 'sim_pred_ed': None, 'sim_executed_vars': None, 'sim_confidence': None
    })

# 5. 사이드바 (S_L, T_R 포함된 로직)
with st.sidebar:
    u_input = st.file_uploader("Upload Log File", type=['csv','xlsx'])
    if st.button("RUN ENGINE INITIALIZATION", type="primary"):
        if u_input:
            df_master = pd.read_csv(u_input) if u_input.name.endswith('csv') else pd.read_excel(u_input)
            df_comb = df_master.dropna(subset=st.session_state['process_vars'] + ['Torque', 'Endurance'])
            X_list = st.session_state['process_vars']
            scaler = MinMaxScaler().fit(df_comb[X_list])
            X_scaled = scaler.transform(df_comb[X_list])
            model_tq = LinearRegression().fit(X_scaled, df_comb['Torque'])
            model_ed = LinearRegression().fit(X_scaled, df_comb['Endurance'])
            st.session_state.update({
                'model_tq': model_tq, 'model_ed': model_ed, 'scaler': scaler, 'df_caulking': df_comb,
                'optimizer_status': "ENGINE READY"
            })
            st.rerun()

# 6. 메인 뷰포트 (Tab 1, 2 에 S_L, T_R 입력 및 최적화 로직 적용)
if st.session_state['model_tq']:
    tab1, tab2, tab3 = st.tabs(["QUALITY INVERSE TARGETING", "REAL-TIME WHAT-IF SIMULATOR", "FACTORY DATALAKE LOGS"])
    
    with tab1: # 최적화 루프
        if st.button("RUN INVERSE INFERENCE SEARCH"):
            X_vars = st.session_state['process_vars']
            def target_loss_function(x):
                pred = st.session_state['model_tq'].predict(st.session_state['scaler'].transform(pd.DataFrame([x], columns=X_vars)))
                return (pred[0] - 36.0)**2
            
            # S_L(10-50), T_R(5-25) 추가
            res = minimize(target_loss_function, x0=[5.5, 2.5, 0, 30.0, 15.0], bounds=[(4,7), (1.5,3.5), (0,1), (10,50), (5,25)])
            st.session_state.update({'opt_result_x': res.x})
            st.rerun()

    with tab2: # 시뮬레이터 UI
        c1, c2, c3, c4, c5 = st.columns(5)
        st.session_state['sim_cd'] = c1.number_input("CD", 4.0, 7.0, st.session_state['sim_cd'])
        st.session_state['sim_sc'] = c2.number_input("SC", 1.5, 3.5, st.session_state['sim_sc'])
        ag = c3.selectbox("Aging", [0, 1])
        st.session_state['sim_sl'] = c4.number_input("S_L", 10.0, 50.0, st.session_state['sim_sl'])
        st.session_state['sim_tr'] = c5.number_input("T_R", 5.0, 25.0, st.session_state['sim_tr'])
        
        if st.button("EXECUTE SIMULATION"):
            df_q = pd.DataFrame([[st.session_state['sim_cd'], st.session_state['sim_sc'], ag, st.session_state['sim_sl'], st.session_state['sim_tr']]], columns=st.session_state['process_vars'])
            pred = st.session_state['model_tq'].predict(st.session_state['scaler'].transform(df_q))
            st.metric("Predicted Torque", f"{pred[0]:.2f} Nm")
