import streamlit as st
import pandas as pd
import numpy as np
import io
import ezdxf
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import minimize

# 1. 페이지 설정
st.set_page_config(layout="wide", page_title="JOINT AI - Engineering Suite", page_icon="⚡")

# 2. DXF 파싱 함수 (데이터 구조 최적화)
def extract_dxf_params(dxf_file):
    params = {"S_L": 0.0, "T_R": 0.0}
    try:
        doc = ezdxf.read(io.BytesIO(dxf_file.getvalue()))
        msp = doc.modelspace()
        # 텍스트 엔티티 전체 리스트 추출
        texts = [e.dxf.text for e in msp.query('TEXT') if e.dxftype() == 'TEXT']
        
        # 'S_L' 또는 'T_R' 텍스트 뒤에 오는 숫자를 찾는 로직
        for i, txt in enumerate(texts):
            if "S_L" in txt and i + 1 < len(texts):
                params["S_L"] = float(texts[i+1])
            if "T_R" in txt and i + 1 < len(texts):
                params["T_R"] = float(texts[i+1])
    except Exception as e:
        st.error(f"DXF Parsing Error: {e}")
    return params

# 3. 사이드바 - 제어반
with st.sidebar:
    st.header("⚙️ CONTROL CONSOLE")
    u_input = st.file_uploader("Upload Log File (CSV, XLSX)", type=['csv','xlsx'])
    dxf_input = st.file_uploader("Upload CAD Spec (DXF)", type=['dxf'])

    # 파싱된 파라미터 상태 저장
    if dxf_input:
        st.session_state.dxf_params = extract_dxf_params(dxf_input)

    if st.button("RUN ENGINE INITIALIZATION"):
        if u_input and dxf_input:
            # CSV/XLSX 로드
            df = pd.read_csv(u_input) if u_input.name.endswith('csv') else pd.read_excel(u_input)
            
            # [핵심] DXF에서 불러온 변수를 데이터프레임의 새로운 컬럼으로 확장
            df['S_L'] = st.session_state.dxf_params['S_L']
            df['T_R'] = st.session_state.dxf_params['T_R']
            
            df = df.dropna(subset=['Torque', 'Endurance'])
            
            # [핵심] 5개의 모든 변수를 학습 데이터(Features)로 활용
            features = ['Caulking_Distance', 'Stud_Center', 'Aging_Status', 'S_L', 'T_R']
            scaler = MinMaxScaler().fit(df[features])
            X_scaled = scaler.transform(df[features])
            
            # 모델 학습
            st.session_state.update({
                'model_tq': LinearRegression().fit(X_scaled, df['Torque']),
                'model_ed': LinearRegression().fit(X_scaled, df['Endurance']),
                'scaler': scaler,
                'df_caulking': df
            })
            st.success(f"Initialization Success! S_L={df['S_L'].iloc[0]}, T_R={df['T_R'].iloc[0]}")
            st.rerun()

# 4. 메인 뷰포트
if st.session_state.get('model_tq') is not None:
    st.success("모델이 5개의 공정 변수(CD, SC, Aging, S_L, T_R)를 학습했습니다.")
    
    tab1, tab2 = st.tabs(["INVERSE OPTIMIZATION", "REAL-TIME SIMULATOR"])
    
    with tab1:
        target = st.number_input("Target Torque", value=36.0)
        if st.button("Calculate Optimal Process"):
            # S_L, T_R은 DXF 값을 고정 변수로 입력하여 CD, SC, Aging 최적화
            def obj(x):
                # x = [CD, SC, Aging]
                input_data = np.array([[x[0], x[1], x[2], st.session_state.dxf_params['S_L'], st.session_state.dxf_params['T_R']]])
                scaled = st.session_state['scaler'].transform(input_data)
                return (st.session_state['model_tq'].predict(scaled)[0] - target)**2
            
            res = minimize(obj, x0=[5.5, 2.5, 0], bounds=[(4,7), (1.5,3.5), (0,1)])
            st.write(f"추천 파라미터: CD={res.x[0]:.2f}, SC={res.x[1]:.2f}, Aging={res.x[2]:.0f}")

    with tab2:
        cd = st.slider("Caulking Distance", 4.0, 7.0, 5.5)
        sc = st.slider("Stud Center", 1.5, 3.5, 2.5)
        # S_L, T_R은 자동으로 DXF 값 반영
        if st.button("Predict"):
            input_data = np.array([[cd, sc, 0, st.session_state.dxf_params['S_L'], st.session_state.dxf_params['T_R']]])
            scaled = st.session_state['scaler'].transform(input_data)
            st.metric("Predicted Torque", f"{st.session_state['model_tq'].predict(scaled)[0]:.2f}")
else:
    st.info("파일을 업로드하고 초기화를 진행하세요.")
