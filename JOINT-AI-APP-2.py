import streamlit as st
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

# 페이지 기본 설정
st.set_page_config(
    page_title="Weld Line AI Integrated Diagnosis System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================================
# 1. 세션 상태(Session State) 초기화 및 boundary 제어 안전장치
# =========================================================================
# 에러가 발생한 'sim_sl' 및 주요 공정 변수들의 기본값과 입력 범위를 정의합니다.
CONFIG_BOUNDS = {
    'sim_sl': {'min': 0.0, 'max': 20.0, 'default': 5.0},
    'mold_temp': {'min': 20.0, 'max': 120.0, 'default': 60.0},
    'melt_temp': {'min': 180.0, 'max': 300.0, 'default': 230.0},
    'inj_speed': {'min': 10.0, 'max': 150.0, 'default': 50.0},
    'pack_press': {'min': 20.0, 'max': 100.0, 'default': 50.0}
}

# 세션 상태 초기화 및 상한/하한선 강제 클램핑(Clamping)
for var, bounds in CONFIG_BOUNDS.items():
    if var not in st.session_state:
        st.session_state[var] = bounds['default']
    else:
        # 외부 연산이나 리런으로 인해 범위를 벗어나는 현상을 원천 차단
        st.session_state[var] = max(bounds['min'], min(float(st.session_state[var]), bounds['max']))


# =========================================================================
# 2. 가상 AI 모델 로드 / 예시 데이터 정의 (XGBoost & Random Forest)
# =========================================================================
@st.cache_resource
def load_diagnostic_models():
    # 실제 환경에서는 학습된 pkl 파일을 로드하거나, 여기에서 가상 학습을 진행합니다.
    # 예시를 위해 더미 분류기를 생성합니다.
    np.random.seed(42)
    X_dummy = np.random.rand(100, 5) * 100
    y_dummy = np.random.choice([0, 1], size=100)
    
    xgb_model = XGBClassifier(n_estimators=50, max_depth=3, random_state=42)
    xgb_model.fit(X_dummy, y_dummy)
    
    rf_model = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)
    rf_model.fit(X_dummy, y_dummy)
    
    return xgb_model, rf_model

xgb_model, rf_model = load_diagnostic_models()


# =========================================================================
# 3. 사이드바 - 가동 조건 및 전문가 제약 조건 입력 (Expert Constraints)
# =========================================================================
st.sidebar.title("🛠️ Expert Condition Entry")
st.sidebar.markdown("---")

st.sidebar.subheader("공정 제약 범위 설정")
max_sl_constraint = st.sidebar.slider("S_L 허용 최댓값 제약", 10.0, 20.0, 20.0, step=0.5)
min_temp_constraint = st.sidebar.slider("금형온도 하한 제약 (°C)", 20.0, 50.0, 40.0, step=5.0)

st.sidebar.markdown("---")
st.sidebar.info("💡 **AI 모델 안내**\n본 진단 시스템은 현재 수집된 대시보드 데이터와 MoldFlow 해석 데이터, 전문가 지식을 융합하여 웰드라인 불량 리스크를 진단합니다.")


# =========================================================================
# 4. 메인 화면 레이아웃 - 대시보드 및 입력 위젯
# =========================================================================
st.title("📊 Weld Line AI Integrated Diagnosis System")
st.write("실시간 공정 조건과 MoldFlow 해석 결과 기반 불량 위험도 진단 및 최적화 시스템")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("💡 MoldFlow / Simulation Inputs")
    
    st.markdown("**S_L Simulation Input (해석 S_L 값)**")
    # [에러 해결 지점] 렌더링 직전 세션 상태 값 상한선 정렬
    st.session_state['sim_sl'] = max(0.0, min(float(st.session_state['sim_sl']), max_sl_constraint))
    
    # 525번째 라인 문제 해결본
    st.session_state['sim_sl'] = st.number_input(
        "S_L Simulation Input", 
        min_value=0.0, 
        max_value=max_sl_constraint, 
        value=st.session_state['sim_sl'], 
        step=0.01, 
        format="%.2f", 
        label_visibility="collapsed"
    )
    
    st.markdown("**금형 온도 (Mold Temperature, °C)**")
    st.session_state['mold_temp'] = st.number_input(
        "Mold Temperature", min_value=min_temp_constraint, max_value=120.0, 
        value=st.session_state['mold_temp'], step=1.0, label_visibility="collapsed"
    )

with col2:
    st.subheader("⚙️ Injection Process Conditions")
    
    st.markdown("**수지 온도 (Melt Temperature, °C)**")
    st.session_state['melt_temp'] = st.number_input(
        "Melt Temperature", min_value=180.0, max_value=300.0, 
        value=st.session_state['melt_temp'], step=1.0, label_visibility="collapsed"
    )
    
    st.markdown("**사출 속도 (Injection Speed, mm/s)**")
    st.session_state['inj_speed'] = st.number_input(
        "Injection Speed", min_value=10.0, max_value=150.0, 
        value=st.session_state['inj_speed'], step=5.0, label_visibility="collapsed"
    )


# =========================================================================
# 5. 진단 실행 및 최적 공정 도출 알고리즘 (Optimization & Inference)
# =========================================================================
st.markdown("---")
layout_btn1, layout_btn2, _ = st.columns([1, 1, 2])

with layout_btn1:
    run_diagnosis = st.button("🔍 불량 리스크 진단 수행", use_container_width=True)

with layout_btn2:
    derive_optimization = st.button("🚀 최적 공정 도출 (Optimize)", use_container_width=True)

# 입력값 어레이 정렬
input_features = np.array([[
    st.session_state['sim_sl'], 
    st.session_state['mold_temp'], 
    st.session_state['melt_temp'], 
    st.session_state['inj_speed'],
    st.session_state['pack_press'] # 기본값 연동
]])

# 5-1. 리스크 진단 버튼 클릭 시
if run_diagnosis:
    xgb_pred_prob = xgb_model.predict_proba(input_features)[0][1] * 100
    rf_pred_prob = rf_model.predict_proba(input_features)[0][1] * 100
    
    st.markdown("### 📈 불량 리스크 예측 결과")
    res_col1, res_col2 = st.columns(2)
    with res_col1:
        st.metric(label="XGBoost 예측 위험도", value=f"{xgb_pred_prob:.1f}%")
    with res_col2:
        st.metric(label="Random Forest 예측 위험도", value=f"{rf_pred_prob:.1f}%")

# 5-2. 최적 공정 도출 버튼 클릭 시 (Expert Constraint 적용 역산 로직)
if derive_optimization:
    st.markdown("### 🛠️ AI 모델 기반 최적 공정 조건 제안")
    
    # 가상의 역산 최적화 알고리즘 구동 (위험도가 낮아지는 최적의 sim_sl 및 공정 조건 계산)
    # 계산 결과가 위젯 상한선(예: 20.0)을 넘는 상황을 시뮬레이션 합니다.
    raw_optimized_sl = 21.45  # 계산 로직에서 20을 초과하는 아웃라이어가 발생했다고 가정
    
    # [핵심 방어 코드] 세션 상태에 반영하기 전, 위젯 맥스값 및 전문가 제약 한계값으로 Clamping
    st.session_state['sim_sl'] = min(raw_optimized_sl, max_sl_constraint)
    st.session_state['mold_temp'] = max(min_temp_constraint, 85.0)  # 예시 최적화 값
    st.session_state['melt_temp'] = 245.0
    st.session_state['inj_speed'] = 65.0
    
    st.success(f"전문가 가이드라인 범위 내에서 최적 공정 조건을 도출했습니다! (S_L 보정값: {st.session_state['sim_sl']:.2f})")
    
    # 변경된 세션 상태를 대시보드 위젯에 즉시 주입하기 위해 리런
    st.rerun()


# =========================================================================
# 6. 하단 데이터 로그 디스플레이
# =========================================================================
st.markdown("---")
st.subheader("📋 Current Session Data Log")
current_data = {
    "Parameter": ["S_L Simulation", "Mold Temp (°C)", "Melt Temp (°C)", "Injection Speed (mm/s)"],
    "Current Value": [
        st.session_state['sim_sl'], 
        st.session_state['mold_temp'], 
        st.session_state['melt_temp'], 
        st.session_state['inj_speed']
    ]
}
st.table(pd.DataFrame(current_data))
