import streamlit as st
import pandas as pd
import numpy as np
import io
import ezdxf
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import minimize

# --- 0. DXF 파싱 함수 추가 ---
def extract_values_from_dxf(dxf_bytes):
    stream = io.BytesIO(dxf_bytes)
    try:
        doc = ezdxf.read(stream)
        msp = doc.modelspace()
        texts = [{'text': e.dxf.text, 'pos': e.dxf.insert} for e in msp.query('TEXT')]
        found = {}
        for label in ['S_L', 'T_R']:
            for item in texts:
                if item['text'] == label:
                    for other in texts:
                        if other['text'] not in ['S_L', 'T_R']:
                            dist = ((other['pos'].x - item['pos'].x)**2 + (other['pos'].y - item['pos'].y)**2)**0.5
                            if dist < 10.0:
                                try: found[label] = float(other['text'])
                                except: continue
        return found
    except: return {}

# 1. 페이지 설정
st.set_page_config(layout="wide", page_title="JOINT AI - Process Optimization Suite", page_icon="⚡")

# 2. CSS (기존 유지)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Inter:wght@400;500;600;700&display=swap');
    .stApp { background-color: #090d16 !important; color: #e2e8f0 !important; font-family: 'Inter', sans-serif; }
    [data-testid="stSidebar"] { background-color: #0f1524 !important; border-right: 1px solid #1e293b; min-width: 360px !important; }
    .glass-card { background: #131b2e; border: 1px solid #223154; border-radius: 6px; padding: 12px 16px; margin-bottom: 12px; }
    .glass-card-title { color: #38bdf8; font-size: 0.85rem; font-weight: 700; text-transform: uppercase; margin-bottom: 0px; padding: 2px 0; }
    .stButton>button { height: 2.8rem !important; background: #10b981 !important; color: #ffffff !important; font-weight: 600; border: none !important; width: 100%; }
    </style>
""", unsafe_allow_html=True)

# 3. 인증 패널
if "authenticated" not in st.session_state: st.session_state.authenticated = False
if not st.session_state.authenticated:
    _, center, _ = st.columns([1, 1.8, 1])
    with center:
        pwd = st.text_input("Enter Password", type="password")
        if st.button("AUTHENTICATE SYSTEM"):
            if pwd == "admin1234":
                st.session_state.authenticated = True
                st.rerun()
    st.stop()

# 4. 세션 초기화 (변수 확장)
if 'model_tq' not in st.session_state:
    st.session_state.update({
        'model_tq': None, 'model_ed': None, 'scaler': None, 'df_caulking': pd.DataFrame(),
        'process_vars': ['Caulking_Distance', 'Stud_Center', 'Aging_Status'],
        'design_vars': ['S_L', 'T_R'],
        'all_features': ['Caulking_Distance', 'Stud_Center', 'Aging_Status', 'S_L', 'T_R'],
        'optimizer_status': "STANDBY",
        'sim_cd': 5.5, 'sim_sc': 2.5, 'sim_sl': 100.0, 'sim_tr': 2.5,
        'opt_result_x': None
    })

# 5. 사이드바
with st.sidebar:
    st.markdown("## CONTROL CONSOLE")
    u_input = st.file_uploader("Upload Log File (CSV, XLSX)", type=['csv','xlsx'])
    dxf_input = st.file_uploader("Upload Blueprint (DXF)", type=['dxf'])
    
    if dxf_input:
        extracted = extract_values_from_dxf(dxf_input.getvalue())
        if extracted:
            if 'S_L' in extracted: st.session_state['sim_sl'] = extracted['S_L']
            if 'T_R' in extracted: st.session_state['sim_tr'] = extracted['T_R']
            st.success(f"도면 인식 완료: {extracted}")

    if st.button("RUN ENGINE INITIALIZATION", type="primary"):
        if u_input:
            df = pd.read_csv(u_input) if u_input.name.endswith('csv') else pd.read_excel(u_input)
            X_list = st.session_state['all_features']
            scaler = MinMaxScaler().fit(df[X_list])
            X_scaled = scaler.transform(df[X_list])
            st.session_state.update({
                'model_tq': LinearRegression().fit(X_scaled, df['Torque']),
                'model_ed': LinearRegression().fit(X_scaled, df['Endurance']),
                'scaler': scaler, 'df_caulking': df, 'optimizer_status': "ENGINE READY"
            })
            st.rerun()

# 6. 메인 뷰포트
if st.session_state['model_tq']:
    tab1, tab2, tab3 = st.tabs(["QUALITY INVERSE TARGETING", "REAL-TIME WHAT-IF SIMULATOR", "FACTORY DATALAKE LOGS"])
    
    with tab1:
        if st.button("RUN INVERSE INFERENCE SEARCH"):
            X_vars = st.session_state['all_features']
            def target_loss_function(x):
                scaled = st.session_state['scaler'].transform(pd.DataFrame([x], columns=X_vars))
                pred_tq = st.session_state['model_tq'].predict(scaled)[0]
                pred_ed = st.session_state['model_ed'].predict(scaled)[0]
                return (pred_tq - 36)**2 + ((pred_ed - 125500)/1000)**2
            
            # 5개 변수 범위 (기존 3개 + S_L, T_R)
            bounds = [(4,7), (1.5,3.5), (0,1), (50,150), (0.5,5)]
            res = minimize(target_loss_function, [5.5, 2.5, 0, 100, 2.5], method='SLSQP', bounds=bounds)
            st.session_state['opt_result_x'] = res.x
            st.rerun()
            
        if st.session_state['opt_result_x'] is not None:
            st.write(f"최적값: {st.session_state['opt_result_x']}")

    with tab2:
        st.number_input("S_L (Socket Length)", value=st.session_state['sim_sl'], key='sim_sl')
        st.number_input("T_R (Tool R)", value=st.session_state['sim_tr'], key='sim_tr')
        # 나머지 기존 시뮬레이션 슬라이더 코드들...
        
    with tab3:
        st.dataframe(st.session_state['df_caulking'])
