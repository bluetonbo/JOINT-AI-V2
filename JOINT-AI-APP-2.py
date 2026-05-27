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

# 2. 미니멀 엔지니어링 콘솔 스타일 CSS (사각형 박스 슬림 최적화 반영)
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
    
    /* 대 타이틀 및 레이블 사각형 상자를 글자 크기에 맞춰 슬림하게 축소 */
    .glass-card {
        background: #131b2e;
        border: 1px solid #223154;
        border-radius: 6px;
        padding: 12px 16px; /* 위아래 패딩을 대폭 줄여 글자에 박스를 밀착 */
        margin-bottom: 12px; /* 컴포넌트 간 간격 최적화 */
    }
    
    /* 박스 내부 타이틀 텍스트 여백 제거 */
    .glass-card-title {
        color: #38bdf8;
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0px; /* 내부 공백을 제로화하여 박스가 비대해지는 것 방지 */
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

# 4. 세션 데이터 구조 초기화 및 초기 세팅값 바인딩
if 'model_tq' not in st.session_state:
    st.session_state.update({
        'model_tq': None, 'model_ed': None, 'scaler': None, 'df_caulking': pd.DataFrame(),
        'process_vars': ['Caulking_Distance', 'Stud_Center', 'Aging_Status', 'Stud_Length', 'Torque_Rate'],
        'data_bounds': {  
            'Caulking_Distance': (4.0, 7.0),
            'Stud_Center': (1.5, 3.5),
            'Aging_Status': (0, 1),
            'Stud_Length': (10.0, 50.0),
            'Torque_Rate': (0.5, 2.0)
        },
        'optimizer_status': "STANDBY",
        
        'm_cd_min': 4.0, 'm_cd_max': 7.0, 'm_sc_min': 1.5, 'm_sc_max': 3.5,
        'm_sl_min': 10.0, 'm_sl_max': 50.0, 'm_tr_min': 0.5, 'm_tr_max': 2.0,
        't_tq_min': 35.0, 't_tq_max': 37.0, 't_ed_min': 125000, 't_ed_max': 126000,
        'sim_cd': 5.5, 'sim_sc': 2.5, 'sim_sl': 30.0, 'sim_tr': 1.0,
        
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
            df_comb = df_master.dropna(subset=['Torque', 'Endurance'])
            X_list = st.session_state['process_vars']
            
            scaler = MinMaxScaler().fit(df_comb[X_list])
            X_scaled = scaler.transform(df_comb[X_list])
            
            model_tq = LinearRegression().fit(X_scaled, df_comb['Torque'])
            model_ed = LinearRegression().fit(X_scaled, df_comb['Endurance'])
            
            # 초기값 데이터프레임 기반 업데이트
            init_cd = (float(df_comb['Caulking_Distance'].min()), float(df_comb['Caulking_Distance'].max()))
            init_sc = (float(df_comb['Stud_Center'].min()), float(df_comb['Stud_Center'].max()))
            init_sl = (float(df_comb['Stud_Length'].min()), float(df_comb['Stud_Length'].max()))
            init_tr = (float(df_comb['Torque_Rate'].min()), float(df_comb['Torque_Rate'].max()))
            
            st.session_state.update({
                'model_tq': model_tq, 'model_ed': model_ed, 'scaler': scaler, 'df_caulking': df_comb,
                'optimizer_status': "ENGINE READY",
                'data_bounds': {
                    'Caulking_Distance': init_cd,
                    'Stud_Center': init_sc,
                    'Aging_Status': (0, 1),
                    'Stud_Length': init_sl,
                    'Torque_Rate': init_tr
                },
                'm_cd_min': init_cd[0], 'm_cd_max': init_cd[1],
                'm_sc_min': init_sc[0], 'm_sc_max': init_sc[1],
                'm_sl_min': init_sl[0], 'm_sl_max': init_sl[1],
                'm_tr_min': init_tr[0], 'm_tr_max': init_tr[1],
                'sim_cd': float(round((init_cd[0] + init_cd[1]) / 2, 2)),
                'sim_sc': float(round((init_sc[0] + init_sc[1]) / 2, 2)),
                'sim_sl': float(round((init_sl[0] + init_sl[1]) / 2, 2)),
                'sim_tr': float(round((init_tr[0] + init_tr[1]) / 2, 2))
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

    # ------------------ TAB 1: 품질 타겟 추적 ------------------
    with tab1:
        layout_l, layout_r = st.columns([1.1, 1.4], gap="large")
        
        with layout_l:
            st.markdown("""
                <div class='glass-card'>
                    <div class='glass-card-title'>Boundary Condition Optimizer</div>
            """, unsafe_allow_html=True)
            
            bound_mode = st.radio("Safety Bound Limit Mode", options=["Auto Mode", "Manual Expert Tuning"], index=0, horizontal=True)
            db = st.session_state['data_bounds']
            
            if "Auto Mode" in bound_mode:
                st.markdown(f"""
                    <div style='background:#0f172a; padding:15px; border-radius:6px; border:1px solid #1e293b; font-size:0.85rem;'>
                        <span style='color:#38bdf8; font-weight:600;'>[Auto-Bound Enabled]</span><br>
                        • CD: {db['Caulking_Distance'][0]:.1f}~{db['Caulking_Distance'][1]:.1f} | SC: {db['Stud_Center'][0]:.1f}~{db['Stud_Center'][1]:.1f}<br>
                        • SL: {db['Stud_Length'][0]:.1f}~{db['Stud_Length'][1]:.1f} | TR: {db['Torque_Rate'][0]:.1f}~{db['Torque_Rate'][1]:.1f}
                    </div>
                """, unsafe_allow_html=True)
                chosen_bounds = db
            else:
                # 간단한 수동 조정을 위한 컬럼 구성 (지면 관계상 핵심 2개 + 2개로 배치)
                st.session_state['m_cd_min'], st.session_state['m_cd_max'] = st.slider("CD Range (mm)", 0.0, 15.0, (st.session_state['m_cd_min'], st.session_state['m_cd_max']))
                st.session_state['m_sc_min'], st.session_state['m_sc_max'] = st.slider("SC Range (mm)", 0.0, 10.0, (st.session_state['m_sc_min'], st.session_state['m_sc_max']))
                st.session_state['m_sl_min'], st.session_state['m_sl_max'] = st.slider("SL Range (mm)", 0.0, 100.0, (st.session_state['m_sl_min'], st.session_state['m_sl_max']))
                st.session_state['m_tr_min'], st.session_state['m_tr_max'] = st.slider("TR Range (ratio)", 0.0, 5.0, (st.session_state['m_tr_min'], st.session_state['m_tr_max']))
                chosen_bounds = {
                    'Caulking_Distance': (st.session_state['m_cd_min'], st.session_state['m_cd_max']),
                    'Stud_Center': (st.session_state['m_sc_min'], st.session_state['m_sc_max']),
                    'Aging_Status': (0, 1),
                    'Stud_Length': (st.session_state['m_sl_min'], st.session_state['m_sl_max']),
                    'Torque_Rate': (st.session_state['m_tr_min'], st.session_state['m_tr_max'])
                }
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("""
                <div class='glass-card'>
                    <div class='glass-card-title'>Target Quality KPIs Range</div>
            """, unsafe_allow_html=True)
            
            st.session_state['t_tq_min'], st.session_state['t_tq_max'] = st.slider("Target Torque (Nm)", 20.0, 50.0, (st.session_state['t_tq_min'], st.session_state['t_tq_max']))
            st.session_state['t_ed_min'], st.session_state['t_ed_max'] = st.slider("Target Endurance (Cyc)", 50000, 200000, (int(st.session_state['t_ed_min']), int(st.session_state['t_ed_max'])))
            
            st.session_state['target_tq_range'] = (st.session_state['t_tq_min'], st.session_state['t_tq_max'])
            st.session_state['target_ed_range'] = (float(st.session_state['t_ed_min']), float(st.session_state['t_ed_max']))
            st.markdown("</div>", unsafe_allow_html=True)
            
            if st.button("RUN INVERSE INFERENCE SEARCH", type="secondary", use_container_width=True):
                X_vars = st.session_state['process_vars']
                def target_loss_function(x_input):
                    df_query = pd.DataFrame([x_input], columns=X_vars)
                    scaled_query = st.session_state['scaler'].transform(df_query)
                    pred_tq = st.session_state['model_tq'].predict(scaled_query)[0]
                    pred_ed = st.session_state['model_ed'].predict(scaled_query)[0]
                    tq_min, tq_max = st.session_state['target_tq_range']
                    ed_min, ed_max = st.session_state['target_ed_range']
                    loss = 0.0
                    if pred_tq < tq_min: loss += (tq_min - pred_tq)**2
                    elif pred_tq > tq_max: loss += (pred_tq - tq_max)**2
                    if pred_ed < ed_min: loss += ((ed_min - pred_ed)/1000)**2
                    elif pred_ed > ed_max: loss += ((pred_ed - ed_max)/1000)**2
                    return loss

                best_res = None
                best_score = float('inf')
                # 5개 변수 최적화 수행
                for ag_option in [0, 1]:
                    init_x = [ (chosen_bounds[v][0]+chosen_bounds[v][1])/2 for v in ['Caulking_Distance', 'Stud_Center', 'Aging_Status', 'Stud_Length', 'Torque_Rate'] ]
                    init_x[2] = ag_option
                    bounds = [chosen_bounds[v] for v in ['Caulking_Distance', 'Stud_Center', 'Aging_Status', 'Stud_Length', 'Torque_Rate']]
                    bounds[2] = (ag_option, ag_option)
                    res = minimize(target_loss_function, init_x, method='SLSQP', bounds=bounds)
                    if res.fun < best_score:
                        best_score = res.fun
                        best_res = res
                
                if best_res:
                    st.session_state['opt_result_x'] = best_res.x
                    scaled_opt = st.session_state['scaler'].transform(pd.DataFrame([best_res.x], columns=X_vars))
                    st.session_state['opt_pred_tq'] = st.session_state['model_tq'].predict(scaled_opt)[0]
                    st.session_state['opt_pred_ed'] = st.session_state['model_ed'].predict(scaled_opt)[0]
                    st.session_state['confidence_score'] = 92.5 # 단순화된 지표
                    st.rerun()

        with layout_r:
            if st.session_state['opt_result_x'] is not None:
                opt_x = st.session_state['opt_result_x']
                st.markdown("<div class='glass-card'><div class='glass-card-title' style='color:#3b82f6;'>Optimization Results</div>", unsafe_allow_html=True)
                st.write(f"CD: {opt_x[0]:.2f} | SC: {opt_x[1]:.2f} | AG: {'Aged' if opt_x[2]>0.5 else 'Unaged'} | SL: {opt_x[3]:.2f} | TR: {opt_x[4]:.2f}")
                st.markdown(f"**Predicted Torque**: {st.session_state['opt_pred_tq']:.2f} Nm | **Endurance**: {st.session_state['opt_pred_ed']:,.0f} Cyc", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

    # ------------------ TAB 2: 실시간 시뮬레이터 ------------------
    with tab2:
        sim_l, sim_r = st.columns([1.1, 1.4], gap="large")
        with sim_l:
            st.markdown("<div class='glass-card'><div class='glass-card-title'>Real-time Simulation Input</div>", unsafe_allow_html=True)
            st.session_state['sim_cd'] = st.slider("Caulking Distance (mm)", 0.0, 15.0, st.session_state['sim_cd'])
            st.session_state['sim_sc'] = st.slider("Stud Center (mm)", 0.0, 10.0, st.session_state['sim_sc'])
            st.session_state['sim_sl'] = st.slider("Stud Length (mm)", 0.0, 100.0, st.session_state['sim_sl'])
            st.session_state['sim_tr'] = st.slider("Torque Rate", 0.0, 5.0, st.session_state['sim_tr'])
            sim_ag = st.radio("Aging Status", [0, 1], format_func=lambda x: "Aged" if x==1 else "Unaged")
            
            if st.button("EXECUTE PREDICTIVE SIMULATION"):
                df_sim = pd.DataFrame([[st.session_state['sim_cd'], st.session_state['sim_sc'], sim_ag, st.session_state['sim_sl'], st.session_state['sim_tr']]], columns=st.session_state['process_vars'])
                scaled = st.session_state['scaler'].transform(df_sim)
                st.session_state['sim_pred_tq'] = st.session_state['model_tq'].predict(scaled)[0]
                st.session_state['sim_pred_ed'] = st.session_state['model_ed'].predict(scaled)[0]
                st.session_state['sim_executed_vars'] = [st.session_state['sim_cd'], st.session_state['sim_sc'], sim_ag, st.session_state['sim_sl'], st.session_state['sim_tr']]
                st.session_state['sim_confidence'] = 95.0
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        with sim_r:
            if st.session_state['sim_pred_tq'] is not None:
                st.markdown("<div class='glass-card'><div class='glass-card-title'>Simulation Outputs</div>", unsafe_allow_html=True)
                st.metric("Predicted Torque (Nm)", f"{st.session_state['sim_pred_tq']:.2f}")
                st.metric("Predicted Endurance (Cyc)", f"{st.session_state['sim_pred_ed']:,.0f}")
                st.markdown("</div>", unsafe_allow_html=True)

    # ------------------ TAB 3 ------------------
    with tab3:
        st.dataframe(st.session_state['df_caulking'], use_container_width=True)

else:
    st.info("CORE ENGINE INACTIVE: Please upload data log.")
