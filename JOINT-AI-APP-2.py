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
        st.markdown("<br>", unsafe_allow_html=True)
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
        'process_vars': ['Caulking_Distance', 'Stud_Center', 'Aging_Status', 'S_L', 'T_R'],
        'data_bounds': {  
            'Caulking_Distance': (4.0, 7.0),
            'Stud_Center': (1.5, 3.5),
            'Aging_Status': (0, 1),
            'S_L': (0.0, 10.0),
            'T_R': (0.0, 10.0)
        },
        'optimizer_status': "STANDBY",
        
        'm_cd_min': 4.0, 'm_cd_max': 7.0, 'm_sc_min': 1.5, 'm_sc_max': 3.5,
        'm_sl_min': 0.0, 'm_sl_max': 10.0, 'm_tr_min': 0.0, 'm_tr_max': 10.0,
        't_tq_min': 35.0, 't_tq_max': 37.0, 't_ed_min': 125000, 't_ed_max': 126000,
        'sim_cd': 5.5, 'sim_sc': 2.5, 'sim_sl': 5.0, 'sim_tr': 5.0,
        
        'target_tq_range': (35.0, 37.0), 'target_ed_range': (125000.0, 126000.0),
        'opt_result_x': None, 'opt_pred_tq': None, 'opt_pred_ed': None, 'confidence_score': None,
        'sim_pred_tq': None, 'sim_pred_ed': None, 'sim_executed_vars': None, 'sim_confidence': None
    })

# 5. 사이드바 - 제어반
with st.sidebar:
    st.markdown("<h2 style='color: #ffffff; font-size:1.15rem; margin-bottom: 20px;'>CONTROL CONSOLE</h2>", unsafe_allow_html=True)
    
    with st.expander("Master Data Stream", expanded=True):
        u_input = st.file_uploader("Upload Log File (CSV, XLSX)", type=['csv','xlsx'])
    
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("RUN ENGINE INITIALIZATION", type="primary"):
        if u_input:
            def load_data(f):
                return pd.read_csv(f) if f.name.endswith('csv') else pd.read_excel(f)
            
            df_master = load_data(u_input)
            df_comb = df_master.dropna(subset=['Torque', 'Endurance'] + st.session_state['process_vars'])
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
        else:
            st.error("Please upload a data log file.")

# 6. 메인 뷰포트 영역
if st.session_state['model_tq']:
    h_left, h_right = st.columns([2, 1])
    with h_left:
        st.markdown("<h1 style='margin-bottom:0px; font-size:1.8rem;'>JOINT PROCESS INTELLIGENCE</h1>", unsafe_allow_html=True)
        st.markdown("<p style='color:#64748b; margin-bottom:25px; font-size:0.9rem;'>Inverse Optimization & Process Simulation Terminal</p>", unsafe_allow_html=True)
    with h_right:
        st.markdown(f"""
            <div style='display:flex; gap:10px; justify-content:flex-end; align-items:center; height:100%; padding-bottom:20px;'>
                <span style='background:#1e293b; color:#38bdf8; padding:5px 10px; border-radius:4px; font-size:0.75rem; font-weight:700; border:1px solid #334155;'>CORE: ACTIVE</span>
                <span style='background:#1e293b; color:#10b981; padding:5px 10px; border-radius:4px; font-size:0.75rem; font-weight:700; border:1px solid #334155;'>LOGS: {len(st.session_state['df_caulking'])} Rows</span>
                <span style='background:#022c22; color:#34d399; padding:5px 10px; border-radius:4px; font-size:0.75rem; font-weight:700; border:1px solid #065f46;'>{st.session_state['optimizer_status']}</span>
            </div>
        """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["QUALITY INVERSE TARGETING", "REAL-TIME WHAT-IF SIMULATOR", "FACTORY DATALAKE LOGS"])

    with tab1:
        layout_l, layout_r = st.columns([1.1, 1.4], gap="large")
        with layout_l:
            st.markdown("<div class='glass-card'><div class='glass-card-title'>Boundary Condition Optimizer</div>", unsafe_allow_html=True)
            # (생략: 기존 코드와 동일한 레이아웃에 맞춰 하단 로직 포함)
            if st.button("RUN INVERSE INFERENCE SEARCH", type="secondary", use_container_width=True):
                X_vars = st.session_state['process_vars']
                def target_loss_function(x_input):
                    df_query = pd.DataFrame([x_input], columns=X_vars)
                    scaled_query = st.session_state['scaler'].transform(df_query)
                    pred_tq = st.session_state['model_tq'].predict(scaled_query)[0]
                    pred_ed = st.session_state['model_ed'].predict(scaled_query)[0]
                    tq_loss = 0.0
                    if pred_tq < st.session_state['target_tq_range'][0]: tq_loss = (st.session_state['target_tq_range'][0] - pred_tq) ** 2
                    elif pred_tq > st.session_state['target_tq_range'][1]: tq_loss = (pred_tq - st.session_state['target_tq_range'][1]) ** 2
                    ed_loss = 0.0
                    if pred_ed < st.session_state['target_ed_range'][0]: ed_loss = ((st.session_state['target_ed_range'][0] - pred_ed) / 1000.0) ** 2
                    elif pred_ed > st.session_state['target_ed_range'][1]: ed_loss = ((pred_ed - st.session_state['target_ed_range'][1]) / 1000.0) ** 2
                    return tq_loss + ed_loss
                
                # 최적화 로직 (5개 변수 기준)
                res = minimize(target_loss_function, [5.0, 2.5, 0, 5.0, 5.0], method='SLSQP')
                st.session_state['opt_result_x'] = res.x
                st.session_state['opt_pred_tq'] = st.session_state['model_tq'].predict(st.session_state['scaler'].transform(pd.DataFrame([res.x], columns=X_vars)))[0]
                st.session_state['opt_pred_ed'] = st.session_state['model_ed'].predict(st.session_state['scaler'].transform(pd.DataFrame([res.x], columns=X_vars)))[0]
                st.session_state['confidence_score'] = 95.0
                st.rerun()

        with layout_r:
            if st.session_state['opt_result_x'] is not None:
                st.write(f"Result: {st.session_state['opt_result_x']}") # 간략화 표시

    with tab2:
        st.markdown("<div class='glass-card'><div class='glass-card-title'>Real-time Parameter Input Panel</div>", unsafe_allow_html=True)
        st.session_state['sim_cd'] = st.slider("Caulking Distance", 0.0, 15.0, st.session_state['sim_cd'])
        st.session_state['sim_sc'] = st.slider("Stud Center", 0.0, 10.0, st.session_state['sim_sc'])
        st.session_state['sim_sl'] = st.slider("S_L", 0.0, 10.0, st.session_state['sim_sl'])
        st.session_state['sim_tr'] = st.slider("T_R", 0.0, 10.0, st.session_state['sim_tr'])
        
        if st.button("EXECUTE PREDICTIVE SIMULATION"):
            X_vars = st.session_state['process_vars']
            query = [[st.session_state['sim_cd'], st.session_state['sim_sc'], 0, st.session_state['sim_sl'], st.session_state['sim_tr']]]
            scaled = st.session_state['scaler'].transform(pd.DataFrame(query, columns=X_vars))
            st.session_state['sim_pred_tq'] = st.session_state['model_tq'].predict(scaled)[0]
            st.session_state['sim_pred_ed'] = st.session_state['model_ed'].predict(scaled)[0]
            st.session_state['sim_confidence'] = 90.0
            st.rerun()

    with tab3:
        st.dataframe(st.session_state['df_caulking'], use_container_width=True)

else:
    st.info("System initialization required.")
