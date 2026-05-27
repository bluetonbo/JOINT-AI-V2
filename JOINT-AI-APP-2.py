import streamlit as st
import pandas as pd
import numpy as np
import io
import ezdxf
import re
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import minimize

# 1. 페이지 설정 및 스타일
st.set_page_config(layout="wide", page_title="JOINT AI - Engineering Suite", page_icon="⚡")

# 2. 세션 초기화
if 'dxf_params' not in st.session_state:
    st.session_state.dxf_params = {"S_L": 0.0, "T_R": 0.0}
if 'model_tq' not in st.session_state:
    st.session_state.update({'model_tq': None, 'model_ed': None, 'scaler': None, 'df_caulking': pd.DataFrame()})

# 3. DXF 파싱 함수 (데이터 구조  반영)
def extract_dxf_params(dxf_file):
    params = {"S_L": 0.0, "T_R": 0.0}
    try:
        doc = ezdxf.read(io.BytesIO(dxf_file.getvalue()))
        msp = doc.modelspace()
        texts = [e.dxf.text for e in msp.query('TEXT') if e.dxftype() == 'TEXT']
        
        # 텍스트 리스트에서 S_L과 T_R 값 찾기 
        for i, txt in enumerate(texts):
            if "S_L" in txt and i + 1 < len(texts):
                params["S_L"] = float(texts[i+1])
            if "T_R" in txt and i + 1 < len(texts):
                params["T_R"] = float(texts[i+1])
    except Exception as e:
        st.error(f"DXF Parsing Error: {e}")
    return params

# 4. 사이드바 - 제어반
with st.sidebar:
    st.header("⚙️ CONTROL CONSOLE")
    u_input = st.file_uploader("Upload Log File (CSV, XLSX)", type=['csv','xlsx'])
    dxf_input = st.file_uploader("Upload CAD Spec (DXF)", type=['dxf'])

    if dxf_input:
        # 파일이 업로드되면 즉시 파싱 수행
        st.session_state.dxf_params = extract_dxf_params(dxf_input)

    if st.button("RUN ENGINE INITIALIZATION"):
        if u_input and st.session_state.dxf_params["S_L"] != 0:
            df = pd.read_csv(u_input) if u_input.name.endswith('csv') else pd.read_excel(u_input)
            df['S_L'] = st.session_state.dxf_params['S_L']
            df['T_R'] = st.session_state.dxf_params['T_R']
            df = df.dropna(subset=['Torque', 'Endurance'])
            
            features = ['Caulking_Distance', 'Stud_Center', 'Aging_Status', 'S_L', 'T_R']
            scaler = MinMaxScaler().fit(df[features])
            X = scaler.transform(df[features])
            
            st.session_state.update({
                'model_tq': LinearRegression().fit(X, df['Torque']),
                'model_ed': LinearRegression().fit(X, df['Endurance']),
                'scaler': scaler, 'df_caulking': df
            })
            st.success("Engine Initialized with CAD Specs!")
            st.rerun()

# 5. 메인 뷰포트
if st.session_state['model_tq']:
    dxf = st.session_state.dxf_params
    st.success(f"System Ready. [Design Specs: S_L={dxf['S_L']}, T_R={dxf['T_R']}]")
    
    tab1, tab2, tab3 = st.tabs(["INVERSE TARGETING", "WHAT-IF SIMULATOR", "DATA LOGS"])
    
    with tab1:
        target = st.number_input("Target Torque (Nm)", value=36.0)
        if st.button("Generate Optimal Process"):
            def obj(x):
                data = np.array([[x[0], x[1], x[2], dxf['S_L'], dxf['T_R']]])
                scaled = st.session_state['scaler'].transform(data)
                return (st.session_state['model_tq'].predict(scaled)[0] - target)**2
            res = minimize(obj, x0=[5.0, 2.5, 0], bounds=[(4,7), (1.5,3.5), (0,1)])
            st.write(f"Optimal Result: CD={res.x[0]:.2f}, SC={res.x[1]:.2f}")

    with tab2:
        cd = st.slider("Caulking Distance", 4.0, 7.0, 5.5)
        sc = st.slider("Stud Center", 1.5, 3.5, 2.5)
        if st.button("Predict"):
            data = np.array([[cd, sc, 0, dxf['S_L'], dxf['T_R']]])
            scaled = st.session_state['scaler'].transform(data)
            st.metric("Predicted Torque", f"{st.session_state['model_tq'].predict(scaled)[0]:.2f} Nm")
            
    with tab3:
        st.dataframe(st.session_state['df_caulking'])
else:
    st.info("파일을 업로드하고 초기화를 진행하세요.")
