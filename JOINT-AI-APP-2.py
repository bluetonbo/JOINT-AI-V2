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
    .stApp { background-color: #090d16 !important; color: #e2e8f0 !important; }
    [data-testid="stSidebar"] { background-color: #0f172a !important; border-right: 1px solid #1e293b; }
    .glass-card { background: #131b2e; border: 1px solid #223154; border-radius: 6px; padding: 16px; margin-bottom: 12px; }
    .glass-card-title { color: #38bdf8; font-size: 0.85rem; font-weight: 700; text-transform: uppercase; margin-bottom: 8px; }
    </style>
""", unsafe_allow_html=True)

# 3. 인증
if "authenticated" not in st.session_state: st.session_state.authenticated = False
if not st.session_state.authenticated:
    _, center, _ = st.columns([1, 1.8, 1])
    with center:
        st.markdown("<br><br><div class='glass-card' style='text-align:center;'><h2>JOINT PROCESS INTELLIGENCE</h2></div>", unsafe_allow_html=True)
        pwd = st.text_input("Enter Password", type="password")
        if st.button("AUTHENTICATE SYSTEM"):
            if pwd == "admin1234": st.session_state.authenticated = True; st.rerun()
            else: st.error("Access denied.")
    st.stop()

# 4. 세션 초기화
if 'model_tq' not in st.session_state:
    st.session_state.update({
        'model_tq': None, 'model_ed': None, 'scaler': None, 'df_caulking': pd.DataFrame(),
        'process_vars': ['S_L', 'T_R', 'Caulking_Distance', 'Stud_Center', 'Aging_Status'],
        'optimizer_status': "STANDBY",
        'sim_vals': {'S_L': 25.0, 'T_R': 4.0, 'CD': 5.5, 'SC': 2.5, 'AG': 0},
        'opt_result_x': None
    })

# 5. 사이드바 - 제어
with st.sidebar:
    st.markdown("## CONTROL CONSOLE")
    u_input = st.file_uploader("Upload Log File", type=['csv','xlsx'])
    if st.button("RUN ENGINE INITIALIZATION", type="primary"):
        if u_input:
            df = pd.read_csv(u_input) if u_input.name.endswith('csv') else pd.read_excel(u_input)
            X_list = st.session_state['process_vars']
            scaler = MinMaxScaler().fit(df[X_list])
            X_scaled = scaler.transform(df[X_list])
            st.session_state.update({
                'model_tq': LinearRegression().fit(X_scaled, df['Torque']),
                'model_ed': LinearRegression().fit(X_scaled, df['Endurance']),
                'scaler': scaler, 'df_caulking': df, 'optimizer_status': "ENGINE READY"
            })
            st.rerun()

# 6. 메인 로직
if st.session_state['model_tq']:
    tab1, tab2, tab3 = st.tabs(["INVERSE TARGETING", "WHAT-IF SIMULATOR", "DATALAKE"])
    
    with tab1:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("<div class='glass-card'><div class='glass-card-title'>Target KPIs</div>", unsafe_allow_html=True)
            tq_range = st.slider("Target Torque (Nm)", 20.0, 50.0, (35.0, 37.0))
            ed_range = st.slider("Target Endurance (Cyc)", 50000, 200000, (120000, 130000))
            if st.button("RUN OPTIMIZATION"):
                def loss(x):
                    s = st.session_state['scaler'].transform([x])
                    tq = st.session_state['model_tq'].predict(s)[0]
                    ed = st.session_state['model_ed'].predict(s)[0]
                    return ((tq - np.mean(tq_range))**2) + ((ed - np.mean(ed_range))/1000)**2
                
                res = minimize(loss, [25, 4, 5.5, 2.5, 0], bounds=[(0,50), (0,10), (0,10), (0,10), (0,1)])
                st.session_state['opt_result_x'] = res.x
                st.rerun()

        with c2:
            if st.session_state['opt_result_x'] is not None:
                x = st.session_state['opt_result_x']
                st.markdown(f"**Optimal Found:** S_L:{x[0]:.2f}, T_R:{x[1]:.2f}, CD:{x[2]:.2f}, SC:{x[3]:.2f}")
                # 엑셀 다운로드 로직 추가 가능

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            st.session_state.sim_vals['S_L'] = st.number_input("S_L", value=25.0)
            st.session_state.sim_vals['T_R'] = st.number_input("T_R", value=4.0)
            st.session_state.sim_vals['CD'] = st.number_input("CD", value=5.5)
            st.session_state.sim_vals['SC'] = st.number_input("SC", value=2.5)
            if st.button("SIMULATE"):
                vals = [st.session_state.sim_vals[k] for k in ['S_L', 'T_R', 'CD', 'SC', 'AG']]
                pred_s = st.session_state['scaler'].transform([vals])
                st.session_state.sim_res = (st.session_state['model_tq'].predict(pred_s)[0], 
                                          st.session_state['model_ed'].predict(pred_s)[0])
        with c2:
            if 'sim_res' in st.session_state:
                st.write(f"Est. Torque: {st.session_state.sim_res[0]:.2f}")
                st.write(f"Est. Endurance: {st.session_state.sim_res[1]:.0f}")

    with tab3:
        st.dataframe(st.session_state['df_caulking'])
