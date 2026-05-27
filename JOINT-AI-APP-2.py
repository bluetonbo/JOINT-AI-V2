import streamlit as st
import pandas as pd
import numpy as np
import io
import ezdxf  # DXF 파일 처리를 위해 추가
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import minimize

# 1. 페이지 설정
st.set_page_config(
    layout="wide", 
    page_title="JOINT AI - Process Optimization Suite",
    page_icon="⚡"
)

# [CSS 스타일링은 기존과 동일하므로 생략 - 전체 코드 복사 시 포함됩니다]
st.markdown("""
    <style>
    .stApp { background-color: #090d16 !important; color: #e2e8f0 !important; font-family: 'Inter', sans-serif; }
    [data-testid="stSidebar"] { background-color: #0f1524 !important; border-right: 1px solid #1e293b; min-width: 360px !important; }
    .glass-card { background: #131b2e; border: 1px solid #223154; border-radius: 6px; padding: 12px 16px; margin-bottom: 12px; }
    .glass-card-title { color: #38bdf8; font-size: 0.85rem; font-weight: 700; text-transform: uppercase; margin-bottom: 0px; padding: 2px 0; }
    .stButton>button { width: 100%; height: 2.8rem; border-radius: 4px; background: #10b981 !important; color: #ffffff !important; border: none; }
    </style>
""", unsafe_allow_html=True)

# 3. 시스템 인증 패널
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    _, center, _ = st.columns([1, 1.8, 1])
    with center:
        pwd = st.text_input("Enter Password", type="password")
        if st.button("AUTHENTICATE SYSTEM"):
            if pwd == "admin1234":
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Invalid credentials.")
    st.stop()

# 4. 세션 데이터 구조 초기화
if 'model_tq' not in st.session_state:
    st.session_state.update({'model_tq': None, 'model_ed': None, 'scaler': None, 'df_caulking': pd.DataFrame(), 'optimizer_status': "STANDBY", 'dxf_params': None})

# 5. 사이드바 - 제어반
with st.sidebar:
    st.markdown("<h2 style='color: #ffffff; font-size:1.15rem; margin-bottom: 20px;'>CONTROL CONSOLE</h2>", unsafe_allow_html=True)
    
    with st.expander("Master Data Stream", expanded=True):
        u_input = st.file_uploader("Upload Log File (CSV, XLSX)", type=['csv','xlsx'])
        dxf_input = st.file_uploader("Upload CAD Spec (DXF)", type=['dxf'])

    if st.button("RUN ENGINE INITIALIZATION", type="primary"):
        # DXF 파일에서 S_L, T_R 추출 로직
        extracted_params = {"S_L": 0.0, "T_R": 0.0}
        if dxf_input:
            try:
                doc = ezdxf.read(dxf_input)
                msp = doc.modelspace()
                for entity in msp.query('TEXT MTEXT'):
                    txt = entity.dxf.text if entity.dxftype() == 'TEXT' else entity.text
                    if "S_L" in txt: extracted_params["S_L"] = float(''.join(filter(str.isdigit, txt)))
                    if "T_R" in txt: extracted_params["T_R"] = float(''.join(filter(str.isdigit, txt)))
                st.session_state['dxf_params'] = extracted_params
            except Exception as e:
                st.error(f"DXF Read Error: {e}")

        if u_input:
            df_master = pd.read_csv(u_input) if u_input.name.endswith('csv') else pd.read_excel(u_input)
            # [이하 기존 모델 학습 로직 동일하게 유지]
            X_list = ['Caulking_Distance', 'Stud_Center', 'Aging_Status']
            df_comb = df_master.dropna(subset=['Torque', 'Endurance'])
            scaler = MinMaxScaler().fit(df_comb[X_list])
            model_tq = LinearRegression().fit(scaler.transform(df_comb[X_list]), df_comb['Torque'])
            model_ed = LinearRegression().fit(scaler.transform(df_comb[X_list]), df_comb['Endurance'])
            st.session_state.update({'model_tq': model_tq, 'model_ed': model_ed, 'scaler': scaler, 'df_caulking': df_comb, 'optimizer_status': "ENGINE READY"})
            st.rerun()

# 6. 메인 뷰포트 (결과 화면은 기존 유지)
if st.session_state['model_tq']:
    # 기존 코드의 탭 및 결과 디스플레이 영역 그대로 유지
    # 필요 시 st.write(f"DXF Param S_L: {st.session_state.get('dxf_params', {}).get('S_L')}") 등을 추가하여 값 확인 가능
    st.success(f"System Initialized. CAD Specs Loaded: {st.session_state.get('dxf_params')}")
    # ... (나머지 기존 Tab 구현부와 동일)
else:
    st.info("Upload logs and CAD file to start.")
