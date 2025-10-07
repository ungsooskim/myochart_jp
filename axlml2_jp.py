# -*- coding: utf-8 -*-
from __future__ import annotations

from datetime import date
from typing import Optional, List
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st
import pytesseract, shutil, os
import re
from PIL import Image

# 사용자 인증 모듈 import
try:
    from auth import is_logged_in, get_current_user, require_login, get_user_specific_data_path, save_user_data, load_user_data, create_demo_user
except ImportError:
    # auth 모듈이 없는 경우 기본 함수들 정의
    def is_logged_in():
        return 'user' in st.session_state
    
    def get_current_user():
        return st.session_state.get('user')
    
    def require_login():
        pass
    
    def get_user_specific_data_path(filename):
        return Path(f"./axl_data/{filename}")
    
    def save_user_data(data, filename):
        file_path = get_user_specific_data_path(filename)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    
    def load_user_data(filename):
        try:
            file_path = get_user_specific_data_path(filename)
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except:
            pass
        return None
    
    def create_demo_user():
        return {'username': 'demo', 'fullName': '데모 사용자'}

# =========================
#  한글 폰트 (OS 자동 설정)
# =========================
_sys = platform.system().lower()
if "windows" in _sys:
    matplotlib.rcParams["font.family"] = "Malgun Gothic"
elif "darwin" in _sys:
    matplotlib.rcParams["font.family"] = "AppleGothic"
else:
    matplotlib.rcParams["font.family"] = "NanumGothic"
matplotlib.rcParams["axes.unicode_minus"] = False

# =========================
#  저장/불러오기 유틸
# =========================
DATA_ROOT = Path("./axl_data")
DATA_ROOT.mkdir(parents=True, exist_ok=True)

REMARK_OPTIONS = ["0.125% AT", "low-dose AT", "OK-lens", "DIMS", "HAL", "MR", "CR"]

def _safe_id(pid: str) -> str:
    return "".join(c for c in (pid or "").strip() if c.isalnum() or c in ("-", "_"))

def remarks_to_str(remarks):
    return "; ".join(remarks) if isinstance(remarks, list) and remarks else ""

def normalize_remarks(raw: str) -> List[str]:
    if not isinstance(raw, str) or not raw.strip():
        return []
    tokens = [t.strip() for t in raw.replace("/", ",").replace(";", ",").split(",")]
    tokens = [t for t in tokens if t]
    canon = []
    for t in tokens:
        if t in REMARK_OPTIONS:
            canon.append(t); continue
        tl = t.lower()
        if tl in ["mg", "myo", "uard"]:
            canon.append("0.125% AT")
        elif tl in ["at", "low-dose at", "low dose at", "atropine", "ldat"]:
            canon.append("low-dose AT")
        elif tl in ["ok", "ok lens", "ortho-k", "orthok", "ok-lens"]:
            canon.append("OK-lens")
        elif tl == "dims":
            canon.append("DIMS")
        elif tl == "hal":
            canon.append("HAL")
        elif tl in ["mr", "manifest refraction", "manifest"]:
            canon.append("MR")
        elif tl in ["cr", "cycloplegic refraction", "cycloplegic", "auto"]:
            canon.append("CR")
    out = []
    for x in REMARK_OPTIONS:
        if x in canon and x not in out:
            out.append(x)
    return out
def clear_input_defaults():
    """입력창의 기본값들을 초기화하는 함수"""
    # 안축장 기본값 초기화
    st.session_state.default_settings["tab1_default_axl_od"] = 23.0
    st.session_state.default_settings["tab1_default_axl_os"] = 23.0
    
    # 굴절이상 기본값 초기화
    st.session_state.default_settings["tab1_default_re_od"] = -2.0
    st.session_state.default_settings["tab1_default_re_os"] = -2.0
    
    # 각막곡률 기본값 초기화
    st.session_state.default_settings["tab1_default_k_od"] = 43.0
    st.session_state.default_settings["tab1_default_k_os"] = 43.0
    
    # 각막두께 기본값 초기화
    st.session_state.default_settings["tab1_default_ct_od"] = 540
    st.session_state.default_settings["tab1_default_ct_os"] = 540
    
    # 비고 기본값 초기화
    st.session_state.default_settings["tab1_default_remarks"] = []
# 안축장 nomogram 데이터 추가
def get_axial_length_nomogram():
    """아이저노 백분위 데이터를 반환하는 함수 (정확한 데이터)"""
    # 남성 데이터 (실제 아이저노 백분위)
    male_data = {
        'age': [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
        'p3': [21.26, 21.49, 21.71, 21.91, 22.09, 22.27, 22.42, 22.56, 22.68, 22.78, 22.86, 22.91, 22.94, 22.95, 22.92],
        'p5': [21.41, 21.64, 21.86, 22.07, 22.26, 22.44, 22.60, 22.75, 22.88, 22.99, 23.08, 23.15, 23.20, 23.23, 23.22],
        'p10': [21.63, 21.87, 22.10, 22.32, 22.53, 22.72, 22.89, 23.05, 23.20, 23.33, 23.44, 23.53, 23.60, 23.66, 23.69],
        'p25': [21.99, 22.26, 22.51, 22.75, 22.998, 23.19, 23.40, 23.58, 23.76, 23.92, 24.06, 24.19, 24.31, 24.41, 24.50],
        'p50': [22.39, 22.69, 22.97, 23.25, 23.51, 23.76, 23.99, 24.22, 24.43, 24.62, 24.81, 24.98, 25.13, 25.28, 25.41],
        'p75': [22.78, 23.12, 23.45, 23.76, 24.07, 24.36, 24.64, 24.90, 25.15, 25.39, 25.61, 25.82, 26.01, 26.18, 26.35],
        'p90': [23.13, 23.51, 23.88, 24.24, 24.60, 24.93, 25.26, 25.57, 25.86, 26.14, 26.39, 26.63, 26.84, 27.04, 27.21],
        'p95': [23.33, 23.74, 24.15, 24.54, 24.92, 25.30, 25.65, 25.99, 26.31, 26.61, 26.89, 27.14, 27.36, 27.56, 27.74]
    }
    
    # 여성 데이터 (실제 아이저노 백분위)
    female_data = {
        'age': [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
        'p3': [20.74, 20.96, 21.17, 21.37, 21.56, 21.73, 21.89, 22.04, 22.18, 22.29, 22.40, 22.48, 22.55, 22.59, 22.61],
        'p5': [20.87, 21.11, 21.33, 21.53, 21.73, 21.92, 22.09, 22.25, 22.39, 22.53, 22.64, 22.74, 22.82, 22.89, 22.93],
        'p10': [21.08, 21.33, 21.56, 21.79, 22.00, 22.21, 22.40, 22.57, 22.73, 22.88, 23.02, 23.14, 23.24, 23.33, 23.41],
        'p25': [21.41, 21.69, 21.96, 22.22, 22.46, 22.70, 22.92, 23.12, 23.31, 23.49, 23.66, 23.81, 23.95, 24.08, 24.19],
        'p50': [21.78, 22.10, 22.41, 22.70, 22.98, 23.25, 23.51, 23.75, 23.97, 24.19, 24.39, 24.57, 24.75, 24.91, 25.05],
        'p75': [22.14, 22.50, 22.85, 23.19, 23.51, 23.82, 24.11, 24.39, 24.65, 24.90, 25.13, 25.34, 25.54, 25.73, 25.89],
        'p90': [22.46, 22.87, 23.26, 23.63, 23.99, 24.34, 24.67, 24.98, 25.28, 25.55, 25.81, 26.05, 26.27, 26.46, 26.64],
        'p95': [22.66, 23.08, 23.50, 23.90, 24.28, 24.65, 25.01, 25.34, 25.66, 25.95, 26.22, 26.47, 26.70, 26.90, 27.08]
    }
    
    return male_data, female_data
def add_nomogram_background(fig, patient_sex, patient_ages=None):
    """Plotly 차트에 nomogram 백분위 곡선을 배경으로 추가"""
    male_data, female_data = get_axial_length_nomogram()
    
    # 患者の性別に応じてデータ選択
    nomogram_data = male_data if patient_sex == "男" else female_data
    
    # 색상 설정 (연한 색상으로 배경 표시)
    colors = {
        'p3': 'rgba(255, 0, 0, 0.2)',      # 빨강 (연함)
        'p5': 'rgba(255, 100, 0, 0.2)',    # 주황 (연함)
        'p10': 'rgba(255, 200, 0, 0.2)',   # 노랑 (연함)
        'p25': 'rgba(100, 255, 100, 0.2)', # 연두 (연함)
        'p50': 'rgba(0, 0, 255, 0.3)',     # 파랑 (중간)
        'p75': 'rgba(100, 255, 100, 0.2)', # 연두 (연함)
        'p90': 'rgba(255, 200, 0, 0.2)',   # 노랑 (연함)
        'p95': 'rgba(255, 100, 0, 0.2)'    # 주황 (연함)
    }
        # 백분위별로 곡선 추가
    for percentile in ['p3', 'p5', 'p10', 'p25', 'p50', 'p75', 'p90', 'p95']:
        if percentile in nomogram_data:
            fig.add_trace(go.Scatter(
                x=nomogram_data['age'],
                y=nomogram_data[percentile],
                mode='lines',
                name=f'{percentile[1:]}%',
                line=dict(color=colors[percentile], width=1, dash='dot'),
                showlegend=True,
                hoverinfo='x+y+name'
            ))
    
    # 범례 그룹화를 위한 레이아웃 업데이트
    fig.update_layout(
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1
        )
    )
    
    return fig
    
def save_bundle(pid: str):
    pid = _safe_id(pid)
    if not pid:
        return False, "환자 ID가 비어 있습니다."
    # 사용자별 데이터 디렉토리 사용
    if is_logged_in():
        pdir = get_user_specific_data_path(pid)
        pdir.mkdir(parents=True, exist_ok=True)
    else:
        pdir = DATA_ROOT / pid
        pdir.mkdir(parents=True, exist_ok=True)

    # AXL
    df_axl = st.session_state.get("data_axl", pd.DataFrame()).copy()
    if not df_axl.empty:
        df_axl["remarks"] = df_axl["remarks"].apply(remarks_to_str)
        df_axl["date"] = pd.to_datetime(df_axl["date"], errors="coerce")
        # 각막곡률 필드가 없는 경우 기본값으로 채우기
        for col in ["OD_K1", "OD_K2", "OD_meanK", "OS_K1", "OS_K2", "OS_meanK"]:
            if col not in df_axl.columns:
                df_axl[col] = np.nan
        df_axl.sort_values("date").to_csv(pdir / "data.csv", index=False)

    # RE
    df_re = st.session_state.get("data_re", pd.DataFrame()).copy()
    if not df_re.empty:
        df_re["remarks"] = df_re["remarks"].apply(remarks_to_str)
        df_re["date"] = pd.to_datetime(df_re["date"], errors="coerce")
        df_re.sort_values("date").to_csv(pdir / "re_data.csv", index=False)

    # 각막곡률
    df_k = st.session_state.get("data_k", pd.DataFrame()).copy()
    if not df_k.empty:
        df_k["remarks"] = df_k["remarks"].apply(remarks_to_str)
        df_k["date"] = pd.to_datetime(df_k["date"], errors="coerce")
        df_k.sort_values("date").to_csv(pdir / "k_data.csv", index=False)

    # 각막두께
    df_ct = st.session_state.get("data_ct", pd.DataFrame()).copy()
    if not df_ct.empty:
        df_ct["remarks"] = df_ct["remarks"].apply(remarks_to_str)
        df_ct["date"] = pd.to_datetime(df_ct["date"], errors="coerce")
        df_ct.sort_values("date").to_csv(pdir / "ct_data.csv", index=False)

    meta = st.session_state.get("meta", {})
    with open(pdir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, default=str)

    return True, f"저장 완료: {pdir}"

def load_bundle(pid: str):
    pid = _safe_id(pid)
    if not pid:
        return False, "환자 ID가 비어 있습니다."
    
    # 사용자별 데이터 디렉토리 사용
    if is_logged_in():
        pdir = get_user_specific_data_path(pid)
    else:
        pdir = DATA_ROOT / pid
    
    if not pdir.exists():
        return False, f"폴더가 없습니다: {pdir}"

    # 기존 데이터 완전히 초기화
    st.session_state.data_axl = pd.DataFrame(columns=["date","OD_mm","OS_mm","OD_K1","OD_K2","OD_meanK","OS_K1","OS_K2","OS_meanK","remarks"]).astype({"date":"datetime64[ns]","OD_mm":"float64","OS_mm":"float64","OD_K1":"float64","OD_K2":"float64","OD_meanK":"float64","OS_K1":"float64","OS_K2":"float64","OS_meanK":"float64","remarks":"object"})
    st.session_state.data_re = pd.DataFrame(columns=["date","OD_sph","OD_cyl","OD_axis","OS_sph","OS_cyl","OS_axis","OD_SE","OS_SE","remarks"]).astype({"date":"datetime64[ns]","OD_sph":"float64","OD_cyl":"float64","OD_axis":"float64","OS_sph":"float64","OS_cyl":"float64","OS_axis":"float64","OD_SE":"float64","OS_SE":"float64","remarks":"object"})
    st.session_state.data_k = pd.DataFrame(columns=["date","OD_K1","OD_K2","OD_meanK","OS_K1","OS_K2","OS_meanK","remarks"]).astype({"date":"datetime64[ns]","OD_K1":"float64","OD_K2":"float64","OD_meanK":"float64","OS_K1":"float64","OS_K2":"float64","OS_meanK":"float64","remarks":"object"})
    st.session_state.data_ct = pd.DataFrame(columns=["date","OD_ct","OS_ct","remarks"]).astype({"date":"datetime64[ns]","OD_ct":"float64","OS_ct":"float64","remarks":"object"})

    # AXL 데이터 로드
    f_axl = pdir / "data.csv"
    if f_axl.exists() and f_axl.stat().st_size > 0:
        df_axl = pd.read_csv(f_axl, na_filter=False)
        if "date" in df_axl.columns:
            df_axl["date"] = pd.to_datetime(df_axl["date"], errors="coerce")
        if "remarks" in df_axl.columns:
            df_axl["remarks"] = df_axl["remarks"].apply(lambda s: normalize_remarks(str(s)))
        else:
            df_axl["remarks"] = [[] for _ in range(len(df_axl))]
        for c in ["OD_mm", "OS_mm", "OD_K1", "OD_K2", "OD_meanK", "OS_K1", "OS_K2", "OS_meanK"]:
            if c not in df_axl.columns: df_axl[c] = np.nan
        st.session_state.data_axl = df_axl[["date","OD_mm","OS_mm","OD_K1","OD_K2","OD_meanK","OS_K1","OS_K2","OS_meanK","remarks"]].sort_values("date")

    # RE 데이터 로드
    f_re = pdir / "re_data.csv"
    if f_re.exists() and f_re.stat().st_size > 0:
        df_re = pd.read_csv(f_re, na_filter=False)
        if "date" in df_re.columns:
            df_re["date"] = pd.to_datetime(df_re["date"], errors="coerce")
        if "remarks" in df_re.columns:
            df_re["remarks"] = df_re["remarks"].apply(lambda s: normalize_remarks(str(s)))
        else:
            df_re["remarks"] = [[] for _ in range(len(df_re))]
        for c in ["OD_sph","OD_cyl","OD_axis","OS_sph","OS_cyl","OS_axis","OD_SE","OS_SE"]:
            if c not in df_re.columns:
                df_re[c] = np.nan
        st.session_state.data_re = df_re[["date","OD_sph","OD_cyl","OD_axis","OS_sph","OS_cyl","OS_axis","OD_SE","OS_SE","remarks"]].sort_values("date")

    # 각막곡률 데이터 로드
    f_k = pdir / "k_data.csv"
    if f_k.exists() and f_k.stat().st_size > 0:
        df_k = pd.read_csv(f_k, na_filter=False)
        if "date" in df_k.columns:
            df_k["date"] = pd.to_datetime(df_k["date"], errors="coerce")
        if "remarks" in df_k.columns:
            df_k["remarks"] = df_k["remarks"].apply(lambda s: normalize_remarks(str(s)))
        else:
            df_k["remarks"] = [[] for _ in range(len(df_k))]
        for c in ["OD_K1","OD_K2","OD_meanK","OS_K1","OS_K2","OS_meanK"]:
            if c not in df_k.columns: df_k[c] = np.nan
        st.session_state.data_k = df_k[["date","OD_K1","OD_K2","OD_meanK","OS_K1","OS_K2","OS_meanK","remarks"]].sort_values("date")

    # 각막두께 데이터 로드
    f_ct = pdir / "ct_data.csv"
    if f_ct.exists() and f_ct.stat().st_size > 0:
        df_ct = pd.read_csv(f_ct, na_filter=False)
        if "date" in df_ct.columns:
            df_ct["date"] = pd.to_datetime(df_ct["date"], errors="coerce")
        if "remarks" in df_ct.columns:
            df_ct["remarks"] = df_ct["remarks"].apply(lambda s: normalize_remarks(str(s)))
        else:
            df_ct["remarks"] = [[] for _ in range(len(df_ct))]
        for c in ["OD_ct","OS_ct"]:
            if c not in df_ct.columns: df_ct[c] = np.nan
        st.session_state.data_ct = df_ct[["date","OD_ct","OS_ct","remarks"]].sort_values("date")

    # META 데이터 로드 (생년월일 포함)
    f_meta = pdir / "meta.json"
    if f_meta.exists() and f_meta.stat().st_size > 0:
        with open(f_meta, "r", encoding="utf-8") as f:
            meta = json.load(f)
        
        # 생년월일 처리 개선
        dob_value = meta.get("dob")
        if dob_value:
            try:
                # 문자열인 경우 datetime으로 변환 후 date로 변환
                if isinstance(dob_value, str):
                    dob_date = pd.to_datetime(dob_value).date()
                else:
                    dob_date = dob_value
            except:
                dob_date = None
        else:
            dob_date = None
        
        st.session_state.meta = {
            "sex": meta.get("sex"),
            "dob": dob_date,
            "current_age": meta.get("current_age"),
            "name": meta.get("name"),
        }
    else:
        # META 파일이 없는 경우 기본값으로 초기화
        st.session_state.meta = {
            "sex": None,
            "dob": None,
            "current_age": None,
            "name": None,
        }

    # 새로운 환자 데이터를 불러온 후 입력창 기본값 초기화
    clear_input_defaults()
    
    # 환자 정보가 성공적으로 불러와졌는지 확인
    if st.session_state.meta.get("name"):
        return True, f"불러오기 완료: {st.session_state.meta.get('name')} ({pid})"
    else:
        return True, f"불러오기 완료: {pid} (환자 정보 없음)"

def list_patient_ids() -> list:
    # 사용자별 또는 기관별 데이터 디렉토리에서 환자 목록 가져오기
    if is_logged_in():
        user_data_dir = st.session_state.user_data_dir
        if not user_data_dir.exists():
            return []
        return sorted([p.name for p in user_data_dir.iterdir() if p.is_dir()])
    else:
        if not DATA_ROOT.exists():
            return []
        return sorted([p.name for p in DATA_ROOT.iterdir() if p.is_dir()])

# =========================
#  분석/예측 유틸
# =========================
def _years_between(d1: pd.Timestamp, d2: pd.Timestamp) -> float:
    return (d2 - d1).days / 365.25

def _age_at_dates(dates: pd.Series, dob: Optional[date], current_age: Optional[float]) -> Optional[pd.Series]:
    if dob is not None:
        dob_ts = pd.Timestamp(dob)
        return dates.apply(lambda d: _years_between(dob_ts, d))
    if current_age is not None:
        today = pd.Timestamp(date.today())
        return dates.apply(lambda d: float(current_age) - _years_between(d, today))
    return None

def _trend_and_predict(x_age: pd.Series, y: pd.Series, target_age: float = 20.0, mode: str = "linear"):
    res = {"slope": np.nan, "intercept": np.nan, "r2": np.nan,
           "pred_at_20": np.nan, "last_age": np.nan, "last_value": np.nan,
           "delta_to_20": np.nan, "valid": False}
    try:
        x = np.array(x_age, dtype=float)
        yv = np.array(y, dtype=float)

        if mode == "log":
            mask = np.isfinite(x) & np.isfinite(yv) & (x > 0)
            X = np.log(x[mask]); yv = yv[mask]
        else:
            mask = np.isfinite(x) & np.isfinite(yv)
            X = x[mask]; yv = yv[mask]

        if X.size < 2:
            return res

        b, a = np.polyfit(X, yv, 1)  # y = b*X + a
        y_hat = b * X + a
        ss_res = np.sum((yv - y_hat) ** 2)
        ss_tot = np.sum((yv - np.mean(yv)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

        X20 = np.log(target_age) if mode == "log" else target_age
        pred_20 = b * X20 + a

        last_age = float(x[mask][-1]); last_val = float(yv[-1])
        delta = float(pred_20 - last_val) if last_age < target_age else 0.0

        res.update({"slope": float(b), "intercept": float(a), "r2": float(r2),
                    "pred_at_20": float(pred_20), "last_age": last_age,
                    "last_value": last_val, "delta_to_20": delta, "valid": True})
        return res
    except Exception:
        return res

# =========================
#  추천(heuristic) 모델: 최적 회귀 + 치료조정
# =========================
def _treatment_adjustment_factor(remarks: List[str]) -> float:
    """
    치료/관리 옵션에 따른 진행 속도 조정 계수(작을수록 억제 강함).
    여러 옵션이 함께 있을 때는 가장 강한 억제를 적용합니다.
    """
    if not isinstance(remarks, list) or not remarks:
        return 1.0
    factors = {
        "0.125% AT": 0.7,
        "low-dose AT": 0.8,
        "OK-lens": 0.6,
        "DIMS": 0.65,
        "HAL": 0.65,
    }
    applied = [factors.get(r, 1.0) for r in remarks]
    return float(min(applied)) if applied else 1.0

def _recommendation_predict(x_age: pd.Series, y: pd.Series, remarks_series: pd.Series, target_age: float = 20.0):
    """
    - 선형/로그 회귀 중 설명력이 더 높은 모델을 자동 선택
    - 마지막 시점의 치료/관리(remarks)에 따라 진행(delta)을 조정
    """
    res_linear = _trend_and_predict(x_age, y, target_age=target_age, mode="linear")
    res_log = _trend_and_predict(x_age, y, target_age=target_age, mode="log")

    choose_log = False
    if res_log.get("valid") and res_linear.get("valid"):
        # r2 높은 모델 선택 (동률이면 선형 유지)
        if (res_log.get("r2") or float("nan")) > (res_linear.get("r2") or float("nan")):
            choose_log = True
    elif res_log.get("valid") and not res_linear.get("valid"):
        choose_log = True

    chosen = res_log if choose_log else res_linear
    if not chosen.get("valid"):
        return chosen | {"chosen_mode": None, "adjust_factor": 1.0}

    # remarks는 최근 행 기준으로 조정
    last_remarks = None
    try:
        if isinstance(remarks_series, pd.Series) and len(remarks_series) > 0:
            last_remarks = remarks_series.iloc[-1]
    except Exception:
        last_remarks = None
    if not isinstance(last_remarks, list):
        last_remarks = []

    factor = _treatment_adjustment_factor(last_remarks)

    last_age = float(chosen.get("last_age") or float("nan"))
    last_val = float(chosen.get("last_value") or float("nan"))
    delta = float(chosen.get("delta_to_20") or 0.0)

    if last_age < target_age and np.isfinite(last_val):
        pred_adj = float(last_val + delta * factor)
    else:
        pred_adj = float(chosen.get("pred_at_20") or last_val)

    out = dict(chosen)
    out.update({
        "pred_at_20": pred_adj,
        "delta_to_20": float(delta * factor),
        "chosen_mode": "log" if choose_log else "linear",
        "adjust_factor": factor,
    })
    return out

# =========================
#  안축장도 이미지 OCR 함수
# =========================
def _parse_axl_image_ocr(ocr_text: str) -> tuple:
    """
    안축장도 이미지의 OCR 텍스트에서 OD, OS AL 값을 추출합니다.
    
    Args:
        ocr_text: OCR로 추출된 텍스트
    
    Returns:
        tuple: (od_al_mm, os_al_mm, success) - 우안 AL, 좌안 AL, 성공 여부
    """
    try:
        # OCR 텍스트 전처리
        text = ocr_text.replace("\r", "").replace("\t", " ")
        text = text.translate(str.maketrans({"−": "-", "–": "-", "—": "-", "‑": "-"}))
        
        od_al = None
        os_al = None
        
        # 방법 1: 좌우 분할 방식 (이미지가 좌우로 나뉘어 있는 경우)
        lines = text.split('\n')
        
        # 각 라인에서 OD와 OS 영역을 좌우로 구분
        od_candidates = []
        os_candidates = []
        
        for line in lines:
            line = line.strip()
            if not line or len(line) < 10:  # 너무 짧은 라인 제외
                continue
            
            # AL이 포함된 라인만 처리
            if 'AL' in line.upper():
                # 라인을 중간 지점으로 나누어 좌우 구분
                mid_point = len(line) // 2
                left_part = line[:mid_point]
                right_part = line[mid_point:]
                
                # 좌측(OD)에서 AL 값 찾기
                od_matches = re.findall(r'(\d{1,2}\.\d{2})\s*mm', left_part, re.IGNORECASE)
                for match in od_matches:
                    val = float(match)
                    if 15.0 <= val <= 35.0:
                        od_candidates.append(val)
                
                # 우측(OS)에서 AL 값 찾기
                os_matches = re.findall(r'(\d{1,2}\.\d{2})\s*mm', right_part, re.IGNORECASE)
                for match in os_matches:
                    val = float(match)
                    if 15.0 <= val <= 35.0:
                        os_candidates.append(val)
        
        # 가장 적절한 값 선택 (첫 번째 값 우선)
        if od_candidates:
            od_al = od_candidates[0]
        if os_candidates:
            os_al = os_candidates[0]
        
        # 방법 2: 텍스트 블록 기반 분석 (OD, OS 키워드로 구분)
        if od_al is None or os_al is None:
            od_section = []
            os_section = []
            current_section = None
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # OD 섹션 감지 (더 엄격한 패턴)
                if re.search(r'\bOD\b.*right', line, re.IGNORECASE) or re.search(r'^\s*OD\s*$', line, re.IGNORECASE):
                    current_section = 'OD'
                    od_section.append(line)
                # OS 섹션 감지 (더 엄격한 패턴)
                elif re.search(r'\bOS\b.*left', line, re.IGNORECASE) or re.search(r'^\s*OS\s*$', line, re.IGNORECASE):
                    current_section = 'OS'
                    os_section.append(line)
                # 현재 섹션에 라인 추가 (단, AL이 포함된 라인만)
                elif current_section == 'OD' and 'AL' in line.upper():
                    od_section.append(line)
                elif current_section == 'OS' and 'AL' in line.upper():
                    os_section.append(line)
            
            # OD 섹션에서 AL 값 추출
            if od_section and od_al is None:
                for line in od_section:
                    al_matches = re.findall(r'(\d{1,2}\.\d{2})\s*mm', line, re.IGNORECASE)
                    for match in al_matches:
                        val = float(match)
                        if 15.0 <= val <= 35.0:
                            od_al = val
                            break
                    if od_al:
                        break
            
            # OS 섹션에서 AL 값 추출
            if os_section and os_al is None:
                for line in os_section:
                    al_matches = re.findall(r'(\d{1,2}\.\d{2})\s*mm', line, re.IGNORECASE)
                    for match in al_matches:
                        val = float(match)
                        if 15.0 <= val <= 35.0:
                            os_al = val
                            break
                    if os_al:
                        break
        
        # 방법 3: 모든 AL 값을 위치순으로 추출 (위치 기반)
        if od_al is None or os_al is None:
            all_al_positions = []
            
            # AL: XX.XX mm 패턴으로 모든 값과 위치 추출
            for match in re.finditer(r'AL[:\s]*(\d{1,2}\.\d{2})\s*mm', text, re.IGNORECASE):
                val = float(match.group(1))
                if 15.0 <= val <= 35.0:
                    all_al_positions.append((val, match.start()))
            
            # 일반 XX.XX mm 패턴으로 추가 추출 (AL 근처에 있는 것들)
            if len(all_al_positions) < 2:
                for line in lines:
                    if 'AL' in line.upper():
                        for match in re.finditer(r'(\d{1,2}\.\d{2})\s*mm', line, re.IGNORECASE):
                            val = float(match.group(1))
                            if 15.0 <= val <= 35.0:
                                line_pos = text.find(line)
                                if line_pos >= 0:
                                    all_al_positions.append((val, line_pos + match.start()))
            
            # 중복 제거 및 위치순 정렬
            unique_positions = []
            for val, pos in all_al_positions:
                if not any(abs(existing_pos - pos) < 50 and existing_val == val 
                          for existing_val, existing_pos in unique_positions):
                    unique_positions.append((val, pos))
            
            unique_positions.sort(key=lambda x: x[1])
            
            # 첫 번째는 OD, 두 번째는 OS로 할당
            if len(unique_positions) >= 2:
                if od_al is None:
                    od_al = unique_positions[0][0]
                if os_al is None:
                    os_al = unique_positions[1][0]
            elif len(unique_positions) == 1 and od_al is None:
                od_al = unique_positions[0][0]
        
        # 방법 4: 특정 패턴으로 직접 매칭
        if od_al is None or os_al is None:
            # 23.70과 24.09 같은 특정 값들을 직접 찾기
            specific_values = re.findall(r'(\d{2}\.\d{2})\s*mm', text, re.IGNORECASE)
            valid_values = [float(v) for v in specific_values if 15.0 <= float(v) <= 35.0]
            
            # 중복 제거하면서 순서 유지
            unique_values = []
            for val in valid_values:
                if val not in unique_values:
                    unique_values.append(val)
            
            if len(unique_values) >= 2:
                if od_al is None:
                    od_al = unique_values[0]
                if os_al is None:
                    os_al = unique_values[1]
            elif len(unique_values) == 1 and od_al is None:
                od_al = unique_values[0]
        
        success = od_al is not None or os_al is not None
        return (od_al, os_al, success)
        
    except Exception as e:
        return (None, None, False)

# =========================
#  텍스트 파싱 함수
# =========================
def _parse_axl_lines(txt: str) -> pd.DataFrame:
    rows = []
    for ln in txt.splitlines():
        if not ln.strip(): continue
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) < 3:
            raise ValueError(f"형식 오류: `{ln}` (필드는 최소 3개)")
        d_str, od_str, os_str = parts[0], parts[1], parts[2]
        
        # 각막곡률 값들 (기본값은 NaN)
        od_k1 = od_k2 = od_mean_k = os_k1 = os_k2 = os_mean_k = np.nan
        
        # 각막곡률이 제공된 경우 파싱
        if len(parts) >= 9:  # 날짜, OD_mm, OS_mm, OD_K1, OD_K2, OD_meanK, OS_K1, OS_K2, OS_meanK, [remarks]
            try:
                od_k1 = float(parts[3]) if parts[3].strip() else np.nan
                od_k2 = float(parts[4]) if parts[4].strip() else np.nan
                od_mean_k = float(parts[5]) if parts[5].strip() else np.nan
                os_k1 = float(parts[6]) if parts[6].strip() else np.nan
                os_k2 = float(parts[7]) if parts[7].strip() else np.nan
                os_mean_k = float(parts[8]) if parts[8].strip() else np.nan
                raw_remark = ",".join(parts[9:]).strip() if len(parts) > 9 else ""
            except ValueError:
                raw_remark = ",".join(parts[3:]).strip() if len(parts) > 3 else ""
        else:
            raw_remark = ",".join(parts[3:]).strip() if len(parts) > 3 else ""
        
        d = pd.to_datetime(d_str, errors="raise")
        rmk = normalize_remarks(raw_remark)
        rows.append((d, float(od_str), float(os_str), od_k1, od_k2, od_mean_k, os_k1, os_k2, os_mean_k, rmk))
    
    df = pd.DataFrame(rows, columns=["date","OD_mm","OS_mm","OD_K1","OD_K2","OD_meanK","OS_K1","OS_K2","OS_meanK","remarks"])
    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    return df

def _parse_re_lines(txt: str) -> pd.DataFrame:
    rows = []
    for ln in txt.splitlines():
        if not ln.strip(): continue
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) < 4:
            raise ValueError(f"형식 오류: `{ln}` (필드는 최소 4개: 날짜, OD S/C/A)")
        d_str = parts[0]
        od_sph, od_cyl, od_ax = parts[1], parts[2], parts[3]
        os_sph = os_cyl = os_ax = np.nan
        raw_remark = ""
        # OS 값이 추가된 경우
        if len(parts) >= 7:
            os_sph, os_cyl, os_ax = parts[4], parts[5], parts[6]
            raw_remark = ",".join(parts[7:]).strip() if len(parts) > 7 else ""
        else:
            raw_remark = ",".join(parts[4:]).strip() if len(parts) > 4 else ""
        d   = pd.to_datetime(d_str, errors="raise")
        rmk = normalize_remarks(raw_remark)
        od_sph = float(od_sph); od_cyl = float(od_cyl); od_ax = float(od_ax)
        os_sph = float(os_sph) if os_sph==os_sph else np.nan
        os_cyl = float(os_cyl) if os_cyl==os_cyl else np.nan
        os_ax  = float(os_ax)  if os_ax==os_ax  else np.nan
        od_se  = od_sph + od_cyl/2.0
        os_se  = (os_sph + os_cyl/2.0) if (os_sph==os_sph and os_cyl==os_cyl) else np.nan
        rows.append((d, od_sph, od_cyl, od_ax, os_sph, os_cyl, os_ax, od_se, os_se, rmk))
    cols = ["date","OD_sph","OD_cyl","OD_axis","OS_sph","OS_cyl","OS_axis","OD_SE","OS_SE","remarks"]
    df = pd.DataFrame(rows, columns=cols)
    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    return df

def _parse_ct_lines(txt: str) -> pd.DataFrame:
    rows = []
    for ln in txt.splitlines():
        if not ln.strip(): continue
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) < 3:
            raise ValueError(f"형식 오류: `{ln}` (필드는 최소 3개)")
        d_str, od_str, os_str = parts[0], parts[1], parts[2]
        raw_remark = ",".join(parts[3:]).strip() if len(parts) > 3 else ""
        
        d = pd.to_datetime(d_str, errors="raise")
        rmk = normalize_remarks(raw_remark)
        rows.append((d, float(od_str), float(os_str), rmk))
    
    df = pd.DataFrame(rows, columns=["date","OD_ct","OS_ct","remarks"])
    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    return df

# =========================
#  페이지 설정 및 초기화
# =========================
st.set_page_config(
    page_title="안축장/굴절이상 추이 및 20세 예측",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 사용자 인증 체크
if not is_logged_in():
    st.markdown(
        """
        <h1 style='font-size:2.8em; font-weight:bold; line-height:1.2; margin-bottom:0.2em; text-align:center;'>
            📊 안축장·굴절이상 추이 및 20세 예측
        </h1>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("---")
    st.markdown("### 🔐 로그인이 필요합니다")
    st.info("개인 맞춤형 성장 차트를 이용하려면 로그인해주세요.")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("🔑 로그인", use_container_width=True):
            st.session_state.show_login = True
            st.rerun()
    with col2:
        if st.button("📝 회원가입", use_container_width=True):
            st.session_state.show_register = True
            st.rerun()
    with col3:
        if st.button("🔍 데모 체험", use_container_width=True):
            create_demo_user()
            st.rerun()
    
    # 로그인/회원가입 폼 표시
    if st.session_state.get('show_login'):
        st.markdown("---")
        st.markdown("### 🔑 로그인")
        
        # 로그인 정보 저장/불러오기 JavaScript
        st.components.v1.html("""
        <script>
        // 페이지 로드 시 저장된 로그인 정보 불러오기
        window.addEventListener('load', function() {
            const savedUsername = localStorage.getItem('saved_username');
            const savedPassword = localStorage.getItem('saved_password');
            const rememberLogin = localStorage.getItem('remember_login') === 'true';
            
            if (savedUsername && rememberLogin) {
                // 입력 필드에 저장된 값 설정
                setTimeout(function() {
                    const usernameInput = document.querySelector('input[data-testid="textInput"][aria-label*="사용자명"]');
                    const passwordInput = document.querySelector('input[data-testid="textInput"][type="password"]');
                    const checkboxInput = document.querySelector('input[data-testid="stCheckbox"]');
                    
                    if (usernameInput) usernameInput.value = savedUsername;
                    if (passwordInput) passwordInput.value = savedPassword;
                    if (checkboxInput) checkboxInput.checked = rememberLogin;
                }, 1000);
            }
        });
        </script>
        """, height=0)
        
        with st.form("login_form"):
            username = st.text_input("사용자명 또는 이메일", 
                                   placeholder="사용자명 또는 이메일을 입력하세요",
                                   key="login_username")
            password = st.text_input("비밀번호", 
                                   type="password", 
                                   placeholder="비밀번호를 입력하세요",
                                   key="login_password")
            
            # 로그인 정보 저장 옵션
            remember_login = st.checkbox("로그인 정보 저장", 
                                       help="브라우저에 로그인 정보를 저장합니다 (보안상 권장하지 않음)",
                                       key="remember_login")
            
            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                login_submitted = st.form_submit_button("로그인", use_container_width=True)
            with col2:
                demo_submitted = st.form_submit_button("데모 로그인", use_container_width=True)
            with col3:
                clear_saved = st.form_submit_button("🗑️", help="저장된 로그인 정보 삭제", use_container_width=True)
        
        if login_submitted:
            if username and password:
                from auth import authenticate_user, save_user_session, find_user_by_email
                # 이메일로 로그인 시도
                user = authenticate_user(username, password)
                if not user:
                    # 이메일로 사용자 찾기
                    email_user = find_user_by_email(username)
                    if email_user:
                        user = authenticate_user(email_user['username'], password)
                
                if user:
                    save_user_session(user)
                    
                    # 로그인 정보 저장 처리 (JavaScript만 사용)
                    if remember_login:
                        # JavaScript로 브라우저에 저장
                        st.components.v1.html(f"""
                        <script>
                        localStorage.setItem('saved_username', '{username}');
                        localStorage.setItem('saved_password', '{password}');
                        localStorage.setItem('remember_login', 'true');
                        </script>
                        """, height=0)
                        st.success("로그인 성공! 로그인 정보가 저장되었습니다.")
                    else:
                        # JavaScript로 브라우저에서 삭제
                        st.components.v1.html("""
                        <script>
                        localStorage.removeItem('saved_username');
                        localStorage.removeItem('saved_password');
                        localStorage.removeItem('remember_login');
                        </script>
                        """, height=0)
                        st.success("로그인 성공!")
                    
                    st.rerun()
                else:
                    st.error("사용자명/이메일 또는 비밀번호가 올바르지 않습니다.")
            else:
                st.error("모든 필드를 입력해주세요.")
        
        if demo_submitted:
            create_demo_user()
            st.success("데모 계정으로 로그인했습니다!")
            st.rerun()
        
        if clear_saved:
            # JavaScript로 브라우저에서 삭제
            st.components.v1.html("""
            <script>
            localStorage.removeItem('saved_username');
            localStorage.removeItem('saved_password');
            localStorage.removeItem('remember_login');
            </script>
            """, height=0)
            
            st.success("저장된 로그인 정보가 삭제되었습니다.")
            st.rerun()
        
        if st.button("← 돌아가기"):
            st.session_state.show_login = False
            st.rerun()
    
    elif st.session_state.get('show_register'):
        st.markdown("---")
        st.markdown("### 📝 회원가입")
        
        with st.form("register_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 👤 개인 정보")
                username = st.text_input("사용자명 *", placeholder="사용자명을 입력하세요")
                email = st.text_input("이메일 *", placeholder="이메일을 입력하세요")
                password = st.text_input("비밀번호 *", type="password", placeholder="비밀번호를 입력하세요 (최소 6자)")
                confirm_password = st.text_input("비밀번호 확인 *", type="password", placeholder="비밀번호를 다시 입력하세요")
                full_name = st.text_input("실명 *", placeholder="실명을 입력하세요")
                birth_date = st.date_input("생년월일 *", value=date(2010, 1, 1), max_value=date.today())
                gender = st.selectbox("성별 *", ["", "남", "여"])
            
            with col2:
                st.markdown("#### 🏥 기관 정보")
                institution_name = st.text_input("기관명 *", placeholder="병원명 또는 기관명을 입력하세요")
                institution_address = st.text_area("직장주소 *", placeholder="기관의 주소를 입력하세요", height=100)
                license_number = st.text_input("면허번호 *", placeholder="의사면허번호를 입력하세요")
                
                st.markdown("#### 🔒 데이터 공유 설정")
                data_sharing = st.radio(
                    "기관 내 데이터 공유",
                    ["개인 데이터만 사용", "기관 내 공유 데이터 사용"],
                    help="기관 내 공유 데이터를 선택하면 동일 기관 사용자들과 환자 데이터를 공유할 수 있습니다."
                )
            
            submitted = st.form_submit_button("회원가입", use_container_width=True)
        
        if submitted:
            from auth import save_user, load_user, find_user_by_email
            # 유효성 검사
            errors = []
            
            if not username:
                errors.append("사용자명을 입력해주세요.")
            elif len(username) < 3:
                errors.append("사용자명은 최소 3자 이상이어야 합니다.")
            elif load_user(username):
                errors.append("이미 존재하는 사용자명입니다.")
            
            if not email:
                errors.append("이메일을 입력해주세요.")
            elif "@" not in email:
                errors.append("올바른 이메일 형식을 입력해주세요.")
            elif find_user_by_email(email):
                errors.append("이미 등록된 이메일입니다.")
            
            if not password:
                errors.append("비밀번호를 입력해주세요.")
            elif len(password) < 6:
                errors.append("비밀번호는 최소 6자 이상이어야 합니다.")
            
            if password != confirm_password:
                errors.append("비밀번호가 일치하지 않습니다.")
            
            if not full_name:
                errors.append("실명을 입력해주세요.")
            
            if not gender:
                errors.append("성별을 선택해주세요.")
            
            if birth_date >= date.today():
                errors.append("생년월일은 오늘 이전이어야 합니다.")
            
            if not institution_name:
                errors.append("기관명을 입력해주세요.")
            
            if not institution_address:
                errors.append("직장주소를 입력해주세요.")
            
            if not license_number:
                errors.append("면허번호를 입력해주세요.")
            elif len(license_number) < 6:
                errors.append("면허번호는 최소 6자 이상이어야 합니다.")
            
            if errors:
                for error in errors:
                    st.error(error)
            else:
                # 사용자 데이터 생성
                user_data = {
                    'username': username,
                    'email': email,
                    'password': password,
                    'fullName': full_name,
                    'birthDate': birth_date.isoformat(),
                    'gender': gender,
                    'institutionName': institution_name,
                    'institutionAddress': institution_address,
                    'licenseNumber': license_number,
                    'dataSharing': data_sharing == "기관 내 공유 데이터 사용"
                }
                
                # 사용자 저장
                if save_user(user_data):
                    st.success("회원가입이 완료되었습니다! 로그인해주세요.")
                    st.session_state.show_register = False
                    st.session_state.show_login = True
                    st.rerun()
                else:
                    st.error("회원가입 중 오류가 발생했습니다. 다시 시도해주세요.")
        
        if st.button("← 돌아가기"):
            st.session_state.show_register = False
            st.rerun()
    
    else:
        st.markdown("---")
        st.markdown("### 📋 서비스 안내")
        st.markdown("""
        - **개인 데이터 보호**: 본인만의 데이터에 접근 가능
        - **안전한 저장**: 모든 데이터는 암호화되어 저장
        - **의료 목적**: 성장 추이 분석 및 예측 서비스
        - **데모 체험**: 로그인 없이 샘플 데이터로 체험 가능
        """)
    
    st.stop()

# 로그인된 사용자용 메인 페이지
user = get_current_user()
st.markdown(
    f"""
    <h1 style='font-size:2.8em; font-weight:bold; line-height:1.2; margin-bottom:0.2em;'>
        眼軸長・屈折異常推移及び20歳予測
    </h1>
    <p style='font-size:1.2em; color:#666; margin-bottom:1em;'>
        こんにちは、<strong>{user.get('fullName', user.get('username', 'ユーザー'))}</strong>さん！ 👋
    </p>
    """,
    unsafe_allow_html=True
)

# 테이블 좌측 정렬 스타일 추가
st.markdown("""
<style>
    /* 테이블 헤더 좌측 정렬 */
    .stDataFrame th {
        text-align: left !important;
    }
    
    /* 테이블 셀 좌측 정렬 */
    .stDataFrame td {
        text-align: left !important;
    }
</style>
""", unsafe_allow_html=True)

if "data_axl" not in st.session_state:
    st.session_state.data_axl = pd.DataFrame(
        columns=["date","OD_mm","OS_mm","OD_K1","OD_K2","OD_meanK","OS_K1","OS_K2","OS_meanK","remarks"]
    ).astype({"date":"datetime64[ns]","OD_mm":"float64","OS_mm":"float64",
              "OD_K1":"float64","OD_K2":"float64","OD_meanK":"float64",
              "OS_K1":"float64","OS_K2":"float64","OS_meanK":"float64","remarks":"object"})
if "data_re" not in st.session_state:
    st.session_state.data_re = pd.DataFrame(
        columns=["date","OD_sph","OD_cyl","OD_axis","OS_sph","OS_cyl","OS_axis","OD_SE","OS_SE","remarks"]
    ).astype({"date":"datetime64[ns]","OD_sph":"float64","OD_cyl":"float64","OD_axis":"float64",
              "OS_sph":"float64","OS_cyl":"float64","OS_axis":"float64",
              "OD_SE":"float64","OS_SE":"float64","remarks":"object"})
if "data_k" not in st.session_state:
    st.session_state.data_k = pd.DataFrame(
        columns=["date","OD_K1","OD_K2","OD_meanK","OS_K1","OS_K2","OS_meanK","remarks"]
    ).astype({"date":"datetime64[ns]","OD_K1":"float64","OD_K2":"float64","OD_meanK":"float64",
              "OS_K1":"float64","OS_K2":"float64","OS_meanK":"float64","remarks":"object"})
if "data_ct" not in st.session_state:
    st.session_state.data_ct = pd.DataFrame(
        columns=["date","OD_ct","OS_ct","remarks"]
    ).astype({"date":"datetime64[ns]","OD_ct":"float64","OS_ct":"float64","remarks":"object"})
if "meta" not in st.session_state:
    st.session_state.meta = {"sex": None, "dob": None, "current_age": None, "name": None}


# =========================
#  설정값 초기화
# =========================
if "default_settings" not in st.session_state:
    st.session_state.default_settings = {
        # 탭1 기본값
        "tab1_default_data_type": "안축장",
        "tab1_default_input_method": "선택입력",
        "tab1_default_axl_od": 23.0,
        "tab1_default_axl_os": 23.0,
        "tab1_default_re_od": -2.0,
        "tab1_default_re_os": -2.0,
        "tab1_default_k_od": 43.0,
        "tab1_default_k_os": 43.0,
        "tab1_default_ct_od": 540,
        "tab1_default_ct_os": 540,
        "tab1_default_remarks": [],
        
        # 탭2 기본값
        "tab2_default_graph_type": "안축장",
        
        # 탭3 기본값
        "tab3_default_analyze_re": True,
        "tab3_default_model_choice": "회귀(선형/로그)",
        "tab3_default_trend_mode": "선형(Linear)"
    }

# =========================
#  사이드바: 재구성된 레이아웃
# =========================
with st.sidebar:
    # ユーザー情報及びログアウト
    st.markdown("### 👤 ユーザー情報")
    user = get_current_user()
    st.info(f"**{user.get('fullName', user.get('username', 'ユーザー'))}**さん")
    st.caption(f"ID: {user.get('username', 'demo')}")
    
    # 機関情報表示
    if user.get('institutionName'):
        st.markdown("### 🏥 機関情報")
        st.success(f"**{user.get('institutionName')}**")
        st.caption(f"免許番号: {user.get('licenseNumber', 'N/A')}")
        
        # データ共有状態表示
        if user.get('dataSharing', False):
            st.markdown("### 🔄 データ共有")
            st.success("機関内共有データ使用中")
            st.caption("同一機関ユーザーとデータ共有")
            
            # 機関内ユーザーリスト表示
            try:
                from auth import get_institution_users
                institution_users = get_institution_users(user.get('institutionName'))
                if institution_users:
                    with st.expander(f"👥 機関ユーザー ({len(institution_users)}名)"):
                        for inst_user in institution_users:
                            if inst_user['username'] != user.get('username'):
                                st.caption(f"• {inst_user.get('fullName', inst_user.get('username'))}")
            except:
                pass
        else:
            st.markdown("### 🔒 データ保護")
            st.info("個人データのみ使用中")
            st.caption("本人データのみアクセス可能")
    
    if st.button("🚪 ログアウト", use_container_width=True):
        try:
            from auth import clear_user_session
            clear_user_session()
        except:
            # authモジュールがない場合、セッションのみクリア
            for key in list(st.session_state.keys()):
                if key.startswith('user'):
                    del st.session_state[key]
        
        # JavaScriptでブラウザに保存されたログイン情報削除
        st.components.v1.html("""
        <script>
        localStorage.removeItem('saved_username');
        localStorage.removeItem('saved_password');
        localStorage.removeItem('remember_login');
        </script>
        """, height=0)
        
        st.rerun()
    
    st.markdown("---")
    
    st.header("患者情報")
    
    # 사용자 정보에서 기본값 설정
    name_default = user.get('fullName', '') or (st.session_state.meta.get("name") if isinstance(st.session_state.get("meta"), dict) else None) or ""
    
    # 불러온 환자 정보가 있으면 우선 사용
    if st.session_state.meta.get("name") and not name_default:
        name_default = st.session_state.meta.get("name")
    
    name = st.text_input("名前/Initial", value=name_default, key="name")
    
    # 환자 ID 기본값 설정
    patient_id_default = ""
    if st.session_state.meta.get("name"):
        patient_id_default = st.session_state.meta.get("name")
    
    # 세션 상태에서 환자 ID 가져오기
    if st.session_state.get("patient_id"):
        patient_id_default = st.session_state.get("patient_id")
    
    patient_id = st.text_input("患者ID（保存用）", value=patient_id_default, key="patient_id")
    
    # 환자 불러오기 후 환자 정보가 제대로 표시되도록 하는 로직은 폼 외부에서 처리됨
    
    # 性別追加（どちらか一つだけ選択）
    st.markdown("**性別**")
    sex_options = ["男", "女"]
    default_sex = st.session_state.meta.get("sex") if isinstance(st.session_state.get("meta"), dict) else None
    if default_sex not in sex_options:
        default_sex = None
    sex = st.radio("性別を選択してください", sex_options, index=sex_options.index(default_sex) if default_sex else 0, horizontal=True, key="sex_radio")
    
    # 생년월일 기본값 설정 개선
    _dob_default = None
    if isinstance(st.session_state.get("meta"), dict) and st.session_state.meta.get("dob"):
        _dob_default = st.session_state.meta.get("dob")
        # date 객체가 아닌 경우 변환
        if not isinstance(_dob_default, date):
            try:
                _dob_default = pd.to_datetime(_dob_default).date()
            except:
                _dob_default = None
    
    if _dob_default is None:
        _dob_default = date.today()
    
    dob_val = st.date_input("生年月日", value=_dob_default, min_value=date(1900, 1, 1), max_value=date.today(), key="dob")
    
    def _calc_age(dob_: date) -> float:
        t = date.today()
        return float(t.year - dob_.year - ((t.month, t.day) < (dob_.month, dob_.day)))
    
    cur_age = _calc_age(dob_val)
    st.metric("現在の年齢", f"{cur_age:.1f} 歳")
    
    # 両親情報
    st.markdown("---")
    show_parent_info = st.checkbox("両親情報入力")

    # 초기화
    father_myopia = mother_myopia = False
    father_se_od = father_se_os = None
    mother_se_od = mother_se_os = None
    father_unknown = mother_unknown = False
    father_lasik = mother_lasik = False
    siblings_info = ""

    if show_parent_info:
        st.subheader("부모 정보")

        # 아버지 정보
        st.markdown("**아버지**")
        col_f1, col_f2, col_f3 = st.columns([1, 1, 1])
        with col_f1:
            father_myopia = st.checkbox("근시", key="father_myopia")
        with col_f2:
            father_unknown = st.checkbox("잘 모름", key="father_unknown")
        with col_f3:
            father_lasik = st.checkbox("굴절수술 과거력", key="father_lasik")

        if father_myopia and not father_unknown:
            col_fod, col_fos = st.columns(2)
            with col_fod:
                father_se_od = st.number_input("OD SE (D)", key="father_se_od", value=0.0, step=0.25, format="%.2f")
            with col_fos:
                father_se_os = st.number_input("OS SE (D)", key="father_se_os", value=0.0, step=0.25, format="%.2f")

        # 어머니 정보
        st.markdown("**어머니**")
        col_m1, col_m2, col_m3 = st.columns([1, 1, 1])
        with col_m1:
            mother_myopia = st.checkbox("근시", key="mother_myopia")
        with col_m2:
            mother_unknown = st.checkbox("잘 모름", key="mother_unknown")
        with col_m3:
            mother_lasik = st.checkbox("굴절수술 과거력", key="mother_lasik")

        if mother_myopia and not mother_unknown:
            col_mod, col_mos = st.columns(2)
            with col_mod:
                mother_se_od = st.number_input("OD SE (D)", key="mother_se_od", value=0.0, step=0.25, format="%.2f")
            with col_mos:
                mother_se_os = st.number_input("OS SE (D)", key="mother_se_os", value=0.0, step=0.25, format="%.2f")

        # 형제 정보
        st.markdown("**형제**")
        siblings_info = st.text_area("형제 정보", key="siblings_info", height=80, placeholder="형제의 근시 상태나 기타 정보를 입력하세요")
    
    # 생활습관
    st.markdown("---")
    show_lifestyle_info = st.checkbox("생활습관 정보 입력")
    
    near_work_hours = outdoor_hours = 0.0
    bedtime = None
    
    if show_lifestyle_info:
        st.subheader("생활습관")
        
        near_work_hours = st.number_input("근거리 작업 (시간)", min_value=0.0, max_value=24.0, value=0.0, step=0.5, format="%.1f")
        outdoor_hours = st.number_input("야외활동 (시간)", min_value=0.0, max_value=24.0, value=0.0, step=0.5, format="%.1f")
        bedtime = st.time_input("취침시간", value=None)
    
    st.session_state.meta = {
        "name": name or None,
        "sex": sex,
        "dob": dob_val,
        "current_age": cur_age,
        "father_myopia": father_myopia,
        "father_se_od": father_se_od,
        "father_se_os": father_se_os,
        "father_unknown": father_unknown,
        "father_lasik": father_lasik,
        "mother_myopia": mother_myopia,
        "mother_se_od": mother_se_od,
        "mother_se_os": mother_se_os,
        "mother_unknown": mother_unknown,
        "mother_lasik": mother_lasik,
        "siblings_info": siblings_info,
        "near_work_hours": near_work_hours,
        "outdoor_hours": outdoor_hours,
        "bedtime": bedtime,
    }
    
    
    if st.button("저장", use_container_width=True, type="primary"):
        ok, msg = save_bundle(patient_id or name or "")
        st.toast(msg)
    
    # 🔹 하단 블록: 불러오기
    st.markdown("---")
    st.subheader("📂 불러오기")
    saved_ids = ["(선택)"] + list_patient_ids()
    selected_pid = st.selectbox("저장된 환자", saved_ids)
    if st.button("불러오기", use_container_width=True):
        pid = selected_pid if selected_pid != "(선택)" else (patient_id or name or "")
        ok, msg = load_bundle(pid)
        
        if ok:
            st.success(f"✅ 환자 데이터 불러오기 성공: {pid}")
            
            # 환자 정보가 성공적으로 불러와졌는지 확인
            if st.session_state.meta.get("name"):
                st.success(f"🎉 환자 '{st.session_state.meta.get('name')}'의 데이터가 성공적으로 불러와졌습니다!")
            else:
                st.warning("⚠️ 환자 기본 정보가 없습니다. 환자 정보를 입력해주세요.")
            
            # 페이지를 새로고침하여 입력창들이 초기화된 기본값으로 설정되도록 함
            st.rerun()
        else:
            st.error(f"❌ 환자 데이터 불러오기 실패: {msg}")

# 새로운 환자 입력 시에도 기본값 초기화
# 환자 정보가 변경될 때마다 기본값 초기화
if "previous_name" not in st.session_state:
    st.session_state.previous_name = name
    st.session_state.previous_patient_id = patient_id

    # 환자 정보가 변경되었을 때만 기본값 초기화
if (st.session_state.previous_name != name or 
    st.session_state.previous_patient_id != patient_id):
    # 환자 정보가 변경되었을 때 입력창 기본값 초기화
    clear_input_defaults()
    st.session_state.previous_name = name
    st.session_state.previous_patient_id = patient_id

# 환자 불러오기 후 환자 정보가 제대로 표시되도록 하는 로직
# 이 로직은 폼 외부에서 실행되어야 함
# 메인페이지에서 중복 표시를 방지하기 위해 제거됨
# 환자 정보는 불러오기 버튼 클릭 시에만 표시됨

# =========================
#  메인 UI - 탭 기반 구조로 개선
# =========================

# タブで主要機能分離
tab1, tab2, tab3, tab4 = st.tabs(["📝 データ入力", "📊 可視化", "🔮 予測分析", "⚙️ 設定"])

# =========================
#  タブ1: データ入力
# =========================
with tab1:
    st.header("📝 データ入力")
    
    # 🔹 入力選択
    data_type = st.radio("**入力選択**", ["眼軸長", "屈折異常", "角膜曲率", "角膜厚", "なし"], horizontal=True)

if data_type == "なし":
    st.info("データ入力を選択していません。上で希望するデータタイプを選択してください。")
    
elif data_type == "眼軸長":
    # 入力方式選択
    axl_input_method = st.radio("**入力方式**", ["選択入力", "テキスト入力", "画像(OCR)"], horizontal=True)
    
    if axl_input_method == "選択入力":
        st.markdown("##### 眼軸長選択入力")
        
        # 基本眼軸長入力
        col1, col2, col3 = st.columns(3)
        with col1:
            axl_date = st.date_input("検査日")
        with col2:
            od_mm = st.number_input("OD (mm)", min_value=15.0, max_value=35.0, value=23.0, step=0.01)
        with col3:
            os_mm = st.number_input("OS (mm)", min_value=15.0, max_value=35.0, value=23.0, step=0.01)
        
        axl_remarks = st.multiselect("治療/管理", REMARK_OPTIONS, default=[])
        
        if st.button("追加", use_container_width=True):
            new_row = pd.DataFrame([{
                "date": pd.to_datetime(axl_date),
                "OD_mm": float(od_mm),
                "OS_mm": float(os_mm),
                "remarks": axl_remarks
            }])
            df_all = pd.concat([st.session_state.data_axl, new_row], ignore_index=True)
            df_all = df_all.sort_values("date").drop_duplicates(subset=["date"], keep="last")
            st.session_state.data_axl = df_all
            st.success("眼軸長データ追加完了")
            if name: save_bundle(name)
            st.rerun()
    
    elif axl_input_method == "テキスト入力":
        st.markdown("##### 眼軸長テキスト入力")
        st.caption("形式: YYYY-M-D, OD(mm), OS(mm)[, Remarks]")
        st.caption("例: 2025-8-16, 23.25, 23.27, AT; DIMS")
        input_text = st.text_area("カンマ区切り入力", height=120, 
                                   placeholder="2025-8-16, 23.25, 23.27, AT; DIMS")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("テキスト追加", use_container_width=True) and input_text.strip():
                try:
                    df_new = _parse_axl_lines(input_text)
                    df_all = pd.concat([st.session_state.data_axl, df_new], ignore_index=True)
                    df_all = df_all.sort_values("date").drop_duplicates(subset=["date"], keep="last")
                    st.session_state.data_axl = df_all
                    st.success(f"{len(df_new)}個の測定値が追加されました。")
                    if name: save_bundle(name)
                    st.rerun()
                except Exception as e:
                    st.error(f"入力解析失敗: {e}")
        with col2:
            if st.button("すべて削除", type="secondary", use_container_width=True):
                st.session_state.data_axl = st.session_state.data_axl.iloc[0:0]
                st.info("眼軸長データをすべて削除しました。")
                if name: save_bundle(name)
    
    else:  # 画像(OCR)
        st.markdown("##### 眼軸長図画像OCR抽出")
        st.caption("眼軸長図測定結果画像をアップロードすると、OD、OSのAL値を自動で抽出します。")
        axl_img = st.file_uploader("眼軸長図画像", type=["png","jpg","jpeg"], key="axl_img")
        
        if axl_img is not None:
            try:
                img = Image.open(axl_img).convert("L")
                ocr_text = pytesseract.image_to_string(img, lang="eng")
                
                # 眼軸長OCR解析
                od_al, os_al, success = _parse_axl_image_ocr(ocr_text)
                
                if success:
                    st.success("眼軸長データ抽出完了！")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if od_al is not None:
                            st.write(f"**右眼(OD) AL**: {od_al:.2f} mm")
                        else:
                            st.write("**右眼(OD)**: 抽出失敗")
                    with col2:
                        if os_al is not None:
                            st.write(f"**左眼(OS) AL**: {os_al:.2f} mm")
                        else:
                            st.write("**左眼(OS)**: 抽出失敗")
                    
                    # 日付及び追加設定
                    col1, col2 = st.columns(2)
                    with col1:
                        axl_ocr_date = st.date_input("検査日", value=date.today(), key="axl_ocr_date")
                    with col2:
                        axl_ocr_remarks = st.multiselect("治療/管理 (OCR)", REMARK_OPTIONS, default=[], key="axl_ocr_remarks")
                    
                    # 手動修正オプション
                    st.markdown("**手動修正（必要時）**")
                    
                    # 안축장 수정
                    col1, col2 = st.columns(2)
                    with col1:
                        od_manifest = st.number_input("OD (mm) 수정", 
                                                   min_value=15.0, max_value=35.0, 
                                                   value=od_al if od_al is not None else 23.0, 
                                                   step=0.01, key="od_manifest")
                    with col2:
                        os_manifest = st.number_input("OS (mm) 수정", 
                                                   min_value=15.0, max_value=35.0, 
                                                   value=os_al if os_al is not None else 23.0, 
                                                   step=0.01, key="os_manifest")
                    
                    if st.button("안축장 OCR 데이터 추가", use_container_width=True, key="add_axl_ocr"):
                        new_row = pd.DataFrame([{
                            "date": pd.to_datetime(axl_ocr_date),
                            "OD_mm": float(od_manifest),
                            "OS_mm": float(os_manifest),
                            "remarks": axl_ocr_remarks
                        }])
                        df_all = pd.concat([st.session_state.data_axl, new_row], ignore_index=True)
                        df_all = df_all.sort_values("date").drop_duplicates(subset=["date"], keep="last")
                        st.session_state.data_axl = df_all
                        st.success("안축장 OCR 데이터 추가됨")
                        if name: save_bundle(name)
                        st.rerun()
                        
                else:
                    st.warning("이미지에서 안축장 데이터를 추출할 수 없습니다.")
                    st.info("다음을 확인해주세요:\n- 이미지가 선명한지\n- OD, OS 및 AL 값이 명확히 보이는지\n- mm 단위로 표시되어 있는지")
                    
                    # OCR 원본 텍스트 및 분석 결과 표시 (디버깅용)
                    with st.expander("OCR 분석 결과 보기"):
                        st.text("=== OCR 원본 텍스트 ===")
                        st.text(ocr_text)
                        
            except Exception as e:
                st.error(f"안축장 OCR 오류: {e}")
                st.info("이미지 형식이나 품질을 확인해주세요.")

elif data_type == "굴절이상":
    # 🔹 입력 방식 선택
    input_method = st.radio("**입력 방식**", ["선택입력", "텍스트입력", "이미지(OCR)"], horizontal=True)
    
    if input_method == "선택입력":
        st.markdown("##### 굴절이상 선택입력")
        
        # MR/CR 구분 체크박스
        col_mr_cr = st.columns(2)
        with col_mr_cr[0]:
            is_mr = st.checkbox("MR (manifest refraction)", value=True, key="is_mr")
        with col_mr_cr[1]:
            is_cr = st.checkbox("CR (cycloplegic refraction)", value=False, key="is_cr")
        
        # MR과 CR 중 하나만 선택되도록 처리
        if is_mr and is_cr:
            st.warning("MR과 CR 중 하나만 선택해주세요.")
        elif not is_mr and not is_cr:
            st.warning("MR 또는 CR 중 하나를 선택해주세요.")
        
        # 선택된 타입 표시
        refraction_type = ""
        if is_mr and not is_cr:
            refraction_type = "MR"
        elif is_cr and not is_mr:
            refraction_type = "CR"
        
        col1, col2 = st.columns(2)
        with col1:
            re_date = st.date_input("검사일", key="re_date")
            sph_opts = [f"{v:.2f}" for v in np.arange(10.00, -20.25, -0.25)]
            sph_zero_idx = sph_opts.index("0.00") if "0.00" in sph_opts else 0
            cyl_opts = [f"{v:.2f}" for v in np.arange(0.00, -8.25, -0.25)]
            axis_opts = [str(i) for i in range(1,181)]
            
            od_sph = st.selectbox("OD Sph", sph_opts, index=sph_zero_idx)
            od_cyl = st.selectbox("OD Cyl", cyl_opts, index=0)
            od_axis = st.selectbox("OD Axis", axis_opts, index=179)
        
        with col2:
            st.write("")
            os_sph = st.selectbox("OS Sph", sph_opts, index=sph_zero_idx)
            os_cyl = st.selectbox("OS Cyl", cyl_opts, index=0)
            os_axis = st.selectbox("OS Axis", axis_opts, index=179)
        
        re_remarks = st.multiselect("치료/관리", REMARK_OPTIONS, default=[])
        
        if st.button("추가", use_container_width=True, key="re_add"):
            # MR/CR 정보를 remarks에 추가
            final_remarks = re_remarks.copy()
            if refraction_type:
                final_remarks.append(refraction_type)
            
            new_row = pd.DataFrame([{
                "date": pd.to_datetime(re_date),
                "OD_sph": float(od_sph), "OD_cyl": float(od_cyl), "OD_axis": float(od_axis),
                "OS_sph": float(os_sph), "OS_cyl": float(os_cyl), "OS_axis": float(os_axis),
                "OD_SE": float(od_sph) + float(od_cyl)/2.0,
                "OS_SE": float(os_sph) + float(os_cyl)/2.0,
                "remarks": final_remarks
            }])
            df_all = pd.concat([st.session_state.data_re, new_row], ignore_index=True)
            df_all = df_all.sort_values("date").drop_duplicates(subset=["date"], keep="last")
            st.session_state.data_re = df_all
            st.success("굴절이상 데이터 추가됨")
            if name: save_bundle(name)
            st.rerun()
    
    elif input_method == "텍스트입력":
        st.markdown("##### 굴절이상 텍스트입력")
        
        # MR/CR 구분 체크박스 (텍스트 입력용)
        col_mr_cr_text = st.columns(2)
        with col_mr_cr_text[0]:
            is_mr_text = st.checkbox("MR (manifest refraction)", value=True, key="is_mr_text")
        with col_mr_cr_text[1]:
            is_cr_text = st.checkbox("CR (cycloplegic refraction)", value=False, key="is_cr_text")
        
        # MR과 CR 중 하나만 선택되도록 처리
        if is_mr_text and is_cr_text:
            st.warning("MR과 CR 중 하나만 선택해주세요.")
        elif not is_mr_text and not is_cr_text:
            st.warning("MR 또는 CR 중 하나를 선택해주세요.")
        
        # 선택된 타입 표시
        refraction_type_text = ""
        if is_mr_text and not is_cr_text:
            refraction_type_text = "MR"
        elif is_cr_text and not is_mr_text:
            refraction_type_text = "CR"
        
        st.caption("형식: YYYY-M-D, OD(Sph), OD(Cyl), OD(Axis)[, OS(Sph), OS(Cyl), OS(Axis)][, Remarks], 예 : 2025-8-16, -2.50, -1.50, 180, -2.25, -1.25, 175, AT")
        input_text_re = st.text_area("콤마 분리 입력", height=140,
                                     placeholder="2025-8-16, -2.50, -1.50, 180, -2.25, -1.25, 175, AT")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("텍스트 추가", use_container_width=True) and input_text_re.strip():
                try:
                    df_new = _parse_re_lines(input_text_re)
                    
                    # MR/CR 정보를 각 행의 remarks에 추가
                    if refraction_type_text:
                        for idx in df_new.index:
                            if isinstance(df_new.loc[idx, 'remarks'], list):
                                df_new.loc[idx, 'remarks'].append(refraction_type_text)
                            else:
                                df_new.loc[idx, 'remarks'] = [refraction_type_text]
                    
                    df_all = pd.concat([st.session_state.data_re, df_new], ignore_index=True)
                    df_all = df_all.sort_values("date").drop_duplicates(subset=["date"], keep="last")
                    st.session_state.data_re = df_all
                    st.success(f"{len(df_new)}개 측정치가 추가되었습니다.")
                    if name: save_bundle(name)
                    st.rerun()
                except Exception as e:
                    st.error(f"입력 파싱 실패: {e}")
        with col2:
            if st.button("모두 지우기", type="secondary", use_container_width=True):
                st.session_state.data_re = st.session_state.data_re.iloc[0:0]
                st.info("굴절이상 데이터를 모두 비웠습니다.")
                if name: save_bundle(name)
    
    else:  # 이미지()
        st.markdown("##### 이미지 OCR 추출")
        up_img = st.file_uploader("자동굴절계 이미지", type=["png","jpg","jpeg"])
        
        if up_img is not None:
            try:
                img = Image.open(up_img).convert("L")
                ocr_text = pytesseract.image_to_string(img, lang="eng")
                
                # OCR 처리 (간소화된 버전)
                t = ocr_text.replace("\r","").replace("\t"," ")
                t = t.translate(str.maketrans({"−": "-", "–": "-", "—": "-", "‑":"-"}))
                
                # 날짜 추출
                dt_match = re.search(r'(\d{4}[-./]\d{1,2}[-./]\d{1,2})', t)
                dt_date = dt_match.group(1) if dt_match else ""
                
                # REF.DATA 블럭 추출
                ref_block = ""
                m_ref = re.search(r'REF\.?DATA(.*?)(KRT\.?DATA|PD:|$)', t, flags=re.S | re.I)
                if m_ref:
                    ref_block = m_ref.group(1)
                
                # OCR 교정
                ref_block = re.sub(r'<\s*b\b', '<L>', ref_block, flags=re.I)
                
                def parse_measurement_line(line: str):
                    pattern = r'([+-]?\d{3,4})\s+([+-]?\d{3,4})\s+(\d{2,3})'
                    match = re.search(pattern, line)
                    if match:
                        s_str, c_str, a_str = match.groups()
                        try:
                            s_val = float(s_str) / 100.0 if abs(float(s_str)) >= 100 else float(s_str)
                            c_val = float(c_str) / 100.0 if abs(float(c_str)) >= 100 else float(c_str)
                            if c_val > 0: c_val = -c_val
                            a_val = int(a_str)
                            if abs(s_val) <= 30 and abs(c_val) <= 15 and 0 <= a_val <= 180:
                                return (round(s_val, 2), round(c_val, 2), a_val)
                        except:
                            pass
                    return None
                
                # 우안/좌안 데이터 추출
                def extract_eye_data(block: str, eye_marker: str):
                    if eye_marker.upper() == 'R':
                        pattern = r'<R>(.*?)(?=<L>|S\.E\.|PD:|$)'
                    else:
                        pattern = r'<L>(.*?)(?=S\.E\.|PD:|$)'
                    
                    match = re.search(pattern, block, flags=re.S | re.I)
                    if not match: return []
                    
                    candidates = []
                    for line in match.group(1).split('\n'):
                        line = line.strip()
                        if line and not re.search(r'\b(WD|PD|MM|AVE|ic|Ss)\b', line, flags=re.I):
                            measurement = parse_measurement_line(line)
                            if measurement: candidates.append(measurement)
                    return candidates
                
                candR = extract_eye_data(ref_block, 'R')
                candL = extract_eye_data(ref_block, 'L')

                # Fallback: REF.DATA 블럭이 없거나 매칭 실패 시 전체 텍스트에서 3-수치 패턴 스캔
                if not (candR or candL):
                    all_triples = []
                    for line in t.split('\n'):
                        line_s = line.strip()
                        if not line_s:
                            continue
                        m = parse_measurement_line(line_s)
                        if m:
                            all_triples.append(m)
                    if len(all_triples) >= 1 and not candR:
                        candR = [all_triples[-1]]
                    if len(all_triples) >= 2 and not candL:
                        candL = [all_triples[-2]]
                
                if candR or candL:
                    st.success("데이터 추출 완료!")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if candR:
                            final_R = candR[-1]  # 마지막 값
                            st.write(f"**우안**: S {final_R[0]:.2f}, C {final_R[1]:.2f}, A {final_R[2]}°")
                    with col2:
                        if candL:
                            final_L = candL[-1]  # 마지막 값
                            st.write(f"**좌안**: S {final_L[0]:.2f}, C {final_L[1]:.2f}, A {final_L[2]}°")
                    
                    ocr_date = st.date_input("검사일", value=pd.to_datetime(dt_date, errors="coerce").date() if dt_date else date.today())
                    ocr_remarks = st.multiselect("치료/관리 (OCR)", REMARK_OPTIONS, default=[])
                    
                    if st.button("OCR 데이터 추가", use_container_width=True):
                        final_R = candR[-1] if candR else (0.0, 0.0, 180)
                        final_L = candL[-1] if candL else (0.0, 0.0, 180)
                        
                        new_row = pd.DataFrame([{
                            "date": pd.to_datetime(ocr_date),
                            "OD_sph": final_R[0], "OD_cyl": final_R[1], "OD_axis": final_R[2],
                            "OS_sph": final_L[0], "OS_cyl": final_L[1], "OS_axis": final_L[2],
                            "OD_SE": final_R[0] + final_R[1]/2.0,
                            "OS_SE": final_L[0] + final_L[1]/2.0,
                            "remarks": ocr_remarks
                        }])
                        df_all = pd.concat([st.session_state.data_re, new_row], ignore_index=True)
                        df_all = df_all.sort_values("date").drop_duplicates(subset=["date"], keep="last")
                        st.session_state.data_re = df_all
                        st.success("OCR 데이터 추가됨")
                        if name: save_bundle(name)
                        st.rerun()
                else:
                    st.warning("데이터를 추출할 수 없습니다.")
            except Exception as e:
                st.error(f"OCR 오류: {e}")

elif data_type == "각막곡률":
    # 🔹 입력 방식 선택
    k_input_method = st.radio("**입력 방식**", ["선택입력", "텍스트입력"], horizontal=True)
    
    if k_input_method == "선택입력":
        st.markdown("##### 각막곡률 선택입력")
        
        # 검사일 입력
        k_date = st.date_input("검사일", key="k_date")
        
        # 우안(OD) 각막곡률
        st.markdown("**우안(OD) 각막곡률**")
        col1, col2, col3 = st.columns(3)
        with col1:
            od_k1 = st.number_input("OD K1 (D)", min_value=30.0, max_value=50.0, value=43.0, step=0.01, key="od_k1")
        with col2:
            od_k2 = st.number_input("OD K2 (D)", min_value=30.0, max_value=50.0, value=44.0, step=0.01, key="od_k2")
        with col3:
            od_mean_k = st.number_input("OD Mean K (D)", min_value=30.0, max_value=50.0, value=43.5, step=0.01, key="od_mean_k")
        
        # 좌안(OS) 각막곡률
        st.markdown("**좌안(OS) 각막곡률**")
        col1, col2, col3 = st.columns(3)
        with col1:
            os_k1 = st.number_input("OS K1 (D)", min_value=30.0, max_value=50.0, value=43.0, step=0.01, key="os_k1")
        with col2:
            os_k2 = st.number_input("OS K2 (D)", min_value=30.0, max_value=50.0, value=44.0, step=0.01, key="os_k2")
        with col3:
            os_mean_k = st.number_input("OS Mean K (D)", min_value=30.0, max_value=50.0, value=43.5, step=0.01, key="os_mean_k")
        
        # 치료/관리 선택
        k_remarks = st.multiselect("치료/관리", REMARK_OPTIONS, default=[], key="k_remarks")
        
        if st.button("각막곡률 추가", use_container_width=True, key="k_add"):
            new_row = pd.DataFrame([{
                "date": pd.to_datetime(k_date),
                "OD_K1": float(od_k1),
                "OD_K2": float(od_k2),
                "OD_meanK": float(od_mean_k),
                "OS_K1": float(os_k1),
                "OS_K2": float(os_k2),
                "OS_meanK": float(os_mean_k),
                "remarks": k_remarks
            }])
            
            # 각막곡률 데이터프레임이 없으면 생성
            if "data_k" not in st.session_state:
                st.session_state.data_k = pd.DataFrame()
            
            df_all = pd.concat([st.session_state.data_k, new_row], ignore_index=True)
            df_all = df_all.sort_values("date").drop_duplicates(subset=["date"], keep="last")
            st.session_state.data_k = df_all
            st.success("각막곡률 데이터 추가됨")
            if name: save_bundle(name)
            st.rerun()
    
    elif k_input_method == "텍스트입력":
        st.markdown("##### 각막곡률 텍스트입력")
        st.caption("형식: YYYY-M-D, OD_K1, OD_K2, OD_MeanK, OS_K1, OS_K2, OS_MeanK[, Remarks]")
        st.caption("예시: 2025-8-16, 43.25, 44.12, 43.69, 43.18, 44.05, 43.62, AT")
        k_input_text = st.text_area("콤마 분리 입력", height=120, 
                                    placeholder="2025-8-16, 43.25, 44.12, 43.69, 43.18, 44.05, 43.62, AT",
                                    key="k_input_text")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("텍스트 추가", use_container_width=True, key="k_text_add") and k_input_text.strip():
                try:
                    # 각막곡률 텍스트 파싱 함수 호출 (아직 구현되지 않음)
                    # df_new = _parse_k_lines(k_input_text)
                    # 임시로 직접 파싱
                    lines = [line.strip() for line in k_input_text.split('\n') if line.strip()]
                    df_new = pd.DataFrame()
                    
                    for line in lines:
                        parts = [part.strip() for part in line.split(',')]
                        if len(parts) >= 7:
                            try:
                                date_val = pd.to_datetime(parts[0])
                                od_k1 = float(parts[1])
                                od_k2 = float(parts[2])
                                od_mean_k = float(parts[3])
                                os_k1 = float(parts[4])
                                os_k2 = float(parts[5])
                                os_mean_k = float(parts[6])
                                remarks = parts[7] if len(parts) > 7 else []
                                
                                new_row = pd.DataFrame([{
                                    "date": date_val,
                                    "OD_K1": od_k1,
                                    "OD_K2": od_k2,
                                    "OD_meanK": od_mean_k,
                                    "OS_K1": os_k1,
                                    "OS_K2": os_k2,
                                    "OS_meanK": os_mean_k,
                                    "remarks": [remarks] if remarks else []
                                }])
                                df_new = pd.concat([df_new, new_row], ignore_index=True)
                            except Exception as e:
                                st.error(f"라인 파싱 실패: {line} - {e}")
                    
                    if not df_new.empty:
                        # 각막곡률 데이터프레임이 없으면 생성
                        if "data_k" not in st.session_state:
                            st.session_state.data_k = pd.DataFrame()
                        
                        df_all = pd.concat([st.session_state.data_k, df_new], ignore_index=True)
                        df_all = df_all.sort_values("date").drop_duplicates(subset=["date"], keep="last")
                        st.session_state.data_k = df_all
                        st.success(f"{len(df_new)}개 측정치가 추가되었습니다.")
                        if name: save_bundle(name)
                        st.rerun()
                    else:
                        st.error("파싱된 데이터가 없습니다.")
                except Exception as e:
                    st.error(f"입력 파싱 실패: {e}")
        with col2:
            if st.button("모두 지우기", type="secondary", use_container_width=True, key="k_clear"):
                if "data_k" in st.session_state:
                    st.session_state.data_k = st.session_state.data_k.iloc[0:0]
                st.info("각막곡률 데이터를 모두 비웠습니다.")
                if name: save_bundle(name)

elif data_type == "각막두께":
    # 입력 방식 선택
    ct_input_method = st.radio("**입력 방식**", ["선택입력", "텍스트입력"], horizontal=True)
    
    if ct_input_method == "선택입력":
        st.markdown("##### 각막두께 선택입력")
        
        # 기본 각막두께 입력
        col1, col2, col3 = st.columns(3)
        with col1:
            ct_date = st.date_input("검사일")
        with col2:
            od_ct = st.number_input("OD (μm)", min_value=400, max_value=700, value=550, step=1)
        with col3:
            os_ct = st.number_input("OS (μm)", min_value=400, max_value=700, value=550, step=1)
        
        ct_remarks = st.multiselect("치료/관리", REMARK_OPTIONS, default=[])
        
        if st.button("추가", use_container_width=True):
            new_row = pd.DataFrame([{
                "date": pd.to_datetime(ct_date),
                "OD_ct": float(od_ct),
                "OS_ct": float(os_ct),
                "remarks": ct_remarks
            }])
            df_all = pd.concat([st.session_state.data_ct, new_row], ignore_index=True)
            df_all = df_all.sort_values("date").drop_duplicates(subset=["date"], keep="last")
            st.session_state.data_ct = df_all
            st.success("각막두께 데이터 추가됨")
            if name: save_bundle(name)
            st.rerun()
    
    else:  # 텍스트입력
        st.markdown("##### 각막두께 텍스트입력")
        st.caption("형식: YYYY-M-D, OD(μm), OS(μm)[, Remarks]")
        st.caption("예시: 2025-8-16, 550, 545, AT; DIMS")
        input_text = st.text_area("콤마 분리 입력", height=120, 
                                   placeholder="2025-8-16, 550, 545, AT; DIMS")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("텍스트 추가", use_container_width=True) and input_text.strip():
                try:
                    df_new = _parse_ct_lines(input_text)
                    df_all = pd.concat([st.session_state.data_ct, df_new], ignore_index=True)
                    df_all = df_all.sort_values("date").drop_duplicates(subset=["date"], keep="last")
                    st.session_state.data_ct = df_all
                    st.success(f"{len(df_new)}개 측정치가 추가되었습니다.")
                    if name: save_bundle(name)
                    st.rerun()
                except Exception as e:
                    st.error(f"입력 파싱 실패: {e}")
        with col2:
            if st.button("모두 지우기", type="secondary", use_container_width=True):
                st.session_state.data_ct = st.session_state.data_ct.iloc[0:0]
                st.info("각막두께 데이터를 모두 비웠습니다.")
                if name: save_bundle(name)

# 데이터 존재 여부 확인 (전역 변수)
has_axl = not st.session_state.data_axl.empty
has_re = not st.session_state.data_re.empty
has_k = "data_k" in st.session_state and not st.session_state.data_k.empty
has_ct = "data_ct" in st.session_state and not st.session_state.data_ct.empty

# =========================
#  탭 2: 시각화
# =========================
with tab2:
    st.header("📊 시각화")
    
    if has_axl or has_re or has_k or has_ct:
        # 그래프 타입 선택
        graph_options = []
        if has_axl:
            graph_options.append("안축장")
        if has_re:
            graph_options.append("굴절이상")

        if has_axl and has_re:
            graph_options.append("이중축 (안축장 + 굴절이상)")
        
        graph_type = st.radio("그래프 타입 선택", graph_options, horizontal=True, key="graph_type")
        
        # 선택된 그래프 타입에 따른 시각화
        if graph_type == "안축장" and has_axl:
            df = st.session_state.data_axl.copy()
            
            # 환자 정보 가져오기
            patient_sex = st.session_state.meta.get("sex", "남")  # 기본값은 남성
            dob = st.session_state.meta.get("dob")
            
            # 2015년 이후 데이터만 필터링
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df = df[df['date'].dt.year >= 2015]
            
            # Plotly 차트 생성
            fig = go.Figure()
            
            # 환자 데이터 변수 저장 (애니메이션에서 사용)
            patient_od_data = None
            patient_os_data = None
            patient_od_ages = None
            patient_os_ages = None
            
            # 1. 환자 데이터 준비 (나이 기준으로)
            if not df.empty and 'OD_mm' in df.columns and 'OS_mm' in df.columns and dob is not None:
                # 환자 나이 계산
                patient_ages = df["date"].apply(lambda d: _years_between(pd.Timestamp(dob), d))
                
                # 마이너스 나이 필터링 (생년월일 이전 날짜 제거)
                valid_age_mask = patient_ages >= 0
                if not valid_age_mask.all():
                    invalid_count = (~valid_age_mask).sum()
                    st.warning(f"⚠️ {invalid_count}개의 생년월일 이전 날짜 데이터가 제외되었습니다. (검사일이 생년월일보다 이전)")
                
                # 유효한 나이만 사용
                patient_ages = patient_ages[valid_age_mask]
                df_filtered = df[valid_age_mask]
                
                # 환자 데이터 저장 (필터링된 데이터 사용)
                if not df_filtered['OD_mm'].isna().all():
                    patient_od_data = df_filtered[['OD_mm']].dropna()
                    patient_od_ages = patient_ages[df_filtered['OD_mm'].notna()]
                
                if not df_filtered['OS_mm'].isna().all():
                    patient_os_data = df_filtered[['OS_mm']].dropna()
                    patient_os_ages = patient_ages[df_filtered['OS_mm'].notna()]
            elif not df.empty and ('OD_mm' in df.columns or 'OS_mm' in df.columns):
                # 생년월일이 없는 경우 날짜 기준으로 표시
                if not df['OD_mm'].isna().all():
                    patient_od_data = df[['date', 'OD_mm']].dropna()
                
                if not df['OS_mm'].isna().all():
                    patient_os_data = df[['date', 'OS_mm']].dropna()
            
            # 2. 백분위 곡선을 배경에 추가 (그림과 같은 스타일)
            male_data, female_data = get_axial_length_nomogram()
            nomogram_data = male_data if patient_sex == "남" else female_data
            
            # 그림자 영역 추가 (p50 중심으로 p25-p75 영역)
            if 'p25' in nomogram_data and 'p75' in nomogram_data:
                fig.add_trace(go.Scatter(
                    x=nomogram_data['age'] + nomogram_data['age'][::-1],
                    y=nomogram_data['p75'] + nomogram_data['p25'][::-1],
                    fill='toself',
                    fillcolor='rgba(200, 200, 200, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='25-75% 영역',
                    showlegend=False,
                    hoverinfo='skip'
                ))
            
            # 백분위별로 곡선 추가 (8개 백분위만)
            percentiles = ['p3', 'p5', 'p10', 'p25', 'p50', 'p75', 'p90', 'p95']
            
            for percentile in percentiles:
                if percentile in nomogram_data:
                    if percentile == 'p50':
                        # 50% 백분위는 굵은 검은색 실선
                        fig.add_trace(go.Scatter(
                            x=nomogram_data['age'],
                            y=nomogram_data[percentile],
                            mode='lines',
                            name='50% 백분위',
                            line=dict(color='black', width=2, dash='solid'),
                            showlegend=True,
                            hoverinfo='x+y+name'
                        ))
                    else:
                        # 나머지는 얇은 회색 점선
                        fig.add_trace(go.Scatter(
                            x=nomogram_data['age'],
                            y=nomogram_data[percentile],
                            mode='lines',
                            name=f'{percentile[1:]}% 백분위',
                            line=dict(color='rgba(128, 128, 128, 0.8)', width=1, dash='dot'),
                            showlegend=False,  # 범례에 표시하지 않음
                            hoverinfo='skip'
                        ))
            
            # 환자 데이터 추가 (그림과 같은 색상)
            if patient_od_data is not None:
                if dob is not None and patient_od_ages is not None:
                    fig.add_trace(go.Scatter(
                        x=patient_od_ages, 
                        y=patient_od_data["OD_mm"], 
                        mode="lines+markers", 
                        name="우안(OD)",
                        line=dict(color='rgba(255, 100, 255, 0.8)', width=3),
                        marker=dict(size=8, color='rgba(255, 100, 255, 0.8)'),
                        hovertemplate='<b>우안(OD)</b><br>나이: %{x:.1f}세<br>안축장: %{y:.2f}mm<extra></extra>'
                    ))
                else:
                    fig.add_trace(go.Scatter(
                        x=patient_od_data["date"], 
                        y=patient_od_data["OD_mm"], 
                        mode="lines+markers", 
                        name="우안(OD)",
                        line=dict(color='rgba(255, 100, 255, 0.8)', width=3),
                        marker=dict(size=8, color='rgba(255, 100, 255, 0.8)'),
                        hovertemplate='<b>우안(OD)</b><br>날짜: %{x}<br>안축장: %{y:.2f}mm<extra></extra>'
                    ))
            
            if patient_os_data is not None:
                if dob is not None and patient_os_ages is not None:
                    fig.add_trace(go.Scatter(
                        x=patient_os_ages, 
                        y=patient_os_data["OS_mm"], 
                        mode="lines+markers", 
                        name="좌안(OS)",
                        line=dict(color='rgba(100, 200, 255, 0.8)', width=3),
                        marker=dict(size=8, color='rgba(100, 200, 255, 0.8)'),
                        hovertemplate='<b>좌안(OS)</b><br>나이: %{x:.1f}세<br>안축장: %{y:.2f}mm<extra></extra>'
                    ))
                else:
                    fig.add_trace(go.Scatter(
                        x=patient_os_data["date"], 
                        y=patient_os_data["OS_mm"], 
                        mode="lines+markers", 
                        name="좌안(OS)",
                        line=dict(color='rgba(100, 200, 255, 0.8)', width=3),
                        marker=dict(size=8, color='rgba(100, 200, 255, 0.8)'),
                        hovertemplate='<b>좌안(OS)</b><br>날짜: %{x}<br>안축장: %{y:.2f}mm<extra></extra>'
                    ))
            
            # 레이아웃 업데이트 (그림과 같은 스타일)
            if dob is not None:
                x_title = "나이 (연)"
                title_suffix = f"({patient_sex}성 백분위 곡선 포함, 2015년 이후)"
            else:
                x_title = "날짜"
                title_suffix = f"(생년월일 미입력, 2015년 이후)"
            
            # 레이아웃 업데이트를 단계별로 수행하여 오류 방지
            try:
                fig.update_layout(
                    title="안축장 성장 차트",
                    xaxis=dict(
                        title=x_title,
                        titlefont=dict(size=12),
                        tickfont=dict(size=10),
                        showgrid=True,
                        gridcolor='rgba(128,128,128,0.3)',
                        gridwidth=1,
                        zeroline=True,
                        zerolinecolor='rgba(128,128,128,0.5)',
                        zerolinewidth=1,
                        showline=True,
                        linecolor='rgba(128,128,128,0.8)',
                        linewidth=1,
                        showticklabels=True,
                        tickmode='linear',
                        tick0=4,
                        dtick=2
                    ),
                    yaxis=dict(
                        title="안축장 (mm)",
                        titlefont=dict(size=12),
                        tickfont=dict(size=10),
                        showgrid=True,
                        gridcolor='rgba(128,128,128,0.3)',
                        gridwidth=1,
                        zeroline=True,
                        zerolinecolor='rgba(128,128,128,0.5)',
                        zerolinewidth=1,
                        showline=True,
                        linecolor='rgba(128,128,128,0.8)',
                        linewidth=1,
                        showticklabels=True,
                        tickmode='linear',
                        dtick=1,
                        fixedrange=False
                    ),
                    hovermode='closest',
                    plot_bgcolor='rgba(255,255,255,1)',
                    paper_bgcolor='rgba(255,255,255,1)',
                    legend=dict(
                        yanchor="top",
                        y=0.98,
                        xanchor="right",
                        x=0.98,
                        bgcolor="rgba(255,255,255,0.9)",
                        bordercolor="rgba(0,0,0,0.3)",
                        borderwidth=1,
                        font=dict(size=10)
                    ),
                    margin=dict(l=60, r=60, t=80, b=60),
                    height=600
                )
                
                # dragmode는 별도로 설정 (호환성 문제 방지)
                fig.update_layout(dragmode='pan')
                
            except Exception as e:
                st.warning(f"차트 레이아웃 설정 중 오류가 발생했습니다: {str(e)}")
                # 기본 레이아웃만 적용
                fig.update_layout(
                    title="안축장 성장 차트",
                    height=600
                )
            
            # Y축 범위 설정 (3%부터 95%까지 항상 표시)
            # 3%와 95% 백분위의 최소/최대값을 계산
            if 'p3' in nomogram_data and 'p95' in nomogram_data:
                p3_min = min(nomogram_data['p3'])
                p3_max = max(nomogram_data['p3'])
                p95_min = min(nomogram_data['p95'])
                p95_max = max(nomogram_data['p95'])
                
                # 전체 나이 구간에서 3%와 95%의 범위 계산
                y_min = min(p3_min, p95_min) - 0.5  # 3%보다 조금 더 아래
                y_max = max(p3_max, p95_max) + 0.5  # 95%보다 조금 더 위
                
                fig.update_yaxes(range=[y_min, y_max])
            
            # X축 범위 설정 (4-18세 전체 범위 표시)
            fig.update_xaxes(range=[4, 18])
            
            
            # 컨트롤 버튼들과 Y축 슬라이더
            button_col1, button_col2, slider_col, button_spacer = st.columns([1, 1, 2, 6])
            
            with button_col1:
                fitting_clicked = st.button("🎯", help="Fitting: 환자 데이터 기준으로 +3개월 범위 보기")
            
            with button_col2:
                autoscale_clicked = st.button("📏", help="Autoscale: 전체 범위로 되돌리기")
            
            with slider_col:
                # Y축 범위 조절 슬라이더 (가로 슬라이더)
                if 'p3' in nomogram_data and 'p95' in nomogram_data:
                    y_scale = st.slider(
                        "Y축 범위", 
                        min_value=0.5, 
                        max_value=3.0, 
                        value=1.0, 
                        step=0.1,
                        help="Y축 범위 조절 (0.5배 ~ 3배)"
                    )
            
            
            # 세션 상태 초기화 (뷰 모드 관리)
            if 'view_mode' not in st.session_state:
                st.session_state.view_mode = 'autoscale'
            
            # 버튼 클릭 처리
            if fitting_clicked:
                st.session_state.view_mode = 'fitting'
                if not df.empty and ('date' in df.columns):
                    # 환자 데이터의 마지막 날짜 찾기
                    patient_dates = df['date'].dropna()
                    if not patient_dates.empty:
                        last_date = patient_dates.max()
                        # +3개월 후 날짜 계산
                        if dob is not None:
                            # 나이 기준으로 +3개월 (0.25년)
                            patient_ages = df["date"].apply(lambda d: _years_between(pd.Timestamp(dob), d))
                            # 마이너스 나이 필터링
                            valid_ages = patient_ages[patient_ages >= 0]
                            if not valid_ages.empty:
                                max_age = valid_ages.max()
                                target_age = max_age + 0.25  # +3개월
                                
                                # X축 범위를 환자 데이터의 첫 데이터 -3개월 ~ 마지막 데이터 +3개월로 설정
                                min_age = valid_ages.min()
                                fig.update_xaxes(range=[min_age - 0.25, target_age])
                            else:
                                st.warning("유효한 나이 데이터가 없습니다.")
                        else:
                            # 날짜 기준으로 +3개월
                            target_date = last_date + pd.DateOffset(months=3)
                            fig.update_xaxes(range=[last_date - pd.DateOffset(months=6), target_date])
                    else:
                        st.warning("환자 데이터가 없습니다.")
                else:
                    st.warning("환자 데이터가 없습니다.")
            
            if autoscale_clicked:
                st.session_state.view_mode = 'autoscale'
                # 전체 범위로 되돌리기
                fig.update_xaxes(range=[4, 18])
            
            # 현재 뷰 모드에 따라 X축 범위 설정 (슬라이더는 항상 적용)
            if st.session_state.view_mode == 'fitting' and not df.empty and ('date' in df.columns):
                patient_dates = df['date'].dropna()
                if not patient_dates.empty and dob is not None:
                    patient_ages = df["date"].apply(lambda d: _years_between(pd.Timestamp(dob), d))
                    # 마이너스 나이 필터링
                    valid_ages = patient_ages[patient_ages >= 0]
                    if not valid_ages.empty:
                        max_age = valid_ages.max()
                        target_age = max_age + 0.25
                        min_age = valid_ages.min()
                        fig.update_xaxes(range=[min_age - 0.25, target_age])
                    else:
                        fig.update_xaxes(range=[4, 18])
                elif not patient_dates.empty:
                    last_date = patient_dates.max()
                    target_date = last_date + pd.DateOffset(months=3)
                    fig.update_xaxes(range=[last_date - pd.DateOffset(months=6), target_date])
            else:
                fig.update_xaxes(range=[4, 18])
            
            # Y축 범위 계산 및 적용
            if 'p3' in nomogram_data and 'p95' in nomogram_data:
                p3_min = min(nomogram_data['p3'])
                p3_max = max(nomogram_data['p3'])
                p95_min = min(nomogram_data['p95'])
                p95_max = max(nomogram_data['p95'])
                
                base_y_min = min(p3_min, p95_min) - 0.5
                base_y_max = max(p3_max, p95_max) + 0.5
                y_range = base_y_max - base_y_min
                
                # 슬라이더 값에 따라 Y축 범위 계산
                center_y = (base_y_min + base_y_max) / 2
                new_range = y_range * y_scale
                new_y_min = center_y - new_range / 2
                new_y_max = center_y + new_range / 2
                
                # Y축 범위 적용
                fig.update_yaxes(range=[new_y_min, new_y_max])
            
            # 디버깅 정보 표시
            with st.expander("🔍 디버깅 정보", expanded=False):
                st.write(f"환자 성별: {patient_sex}")
                st.write(f"생년월일: {dob}")
                st.write(f"데이터프레임 크기: {df.shape}")
                st.write(f"OD 데이터 존재: {patient_od_data is not None}")
                st.write(f"OS 데이터 존재: {patient_os_data is not None}")
                if not df.empty:
                    st.write("데이터프레임 컬럼:", df.columns.tolist())
                    if 'OD_mm' in df.columns:
                        st.write("OD_mm 데이터:", df['OD_mm'].dropna().tolist())
                    if 'OS_mm' in df.columns:
                        st.write("OS_mm 데이터:", df['OS_mm'].dropna().tolist())
                    
                    # 나이 계산 디버깅
                    if dob is not None and 'date' in df.columns:
                        st.write("=== 나이 계산 디버깅 ===")
                        st.write(f"생년월일: {dob}")
                        sample_dates = df['date'].dropna().head(5)
                        st.write("샘플 날짜들:", sample_dates.tolist())
                        
                        # 나이 계산 결과 확인
                        ages = df["date"].apply(lambda d: _years_between(pd.Timestamp(dob), d))
                        st.write("계산된 나이들:", ages.dropna().tolist())
                        st.write("나이 범위:", f"{ages.min():.2f} ~ {ages.max():.2f}세")
                        
                        # 마이너스 나이가 있는지 확인 (생년월일 이전 날짜)
                        negative_ages = ages[ages < 0]
                        if not negative_ages.empty:
                            st.error(f"⚠️ 생년월일 이전 날짜 발견 (마이너스 나이): {negative_ages.tolist()}")
                            st.write("해당 날짜들:", df.loc[negative_ages.index, 'date'].tolist())
                            st.write("→ 이 날짜들은 생년월일보다 이전이므로 제외됩니다.")
                        
                        # OD/OS 데이터와 나이 매칭 확인
                        if patient_od_ages is not None:
                            st.write("OD 나이들:", patient_od_ages.tolist())
                        if patient_os_ages is not None:
                            st.write("OS 나이들:", patient_os_ages.tolist())
                
                st.write(f"차트 트레이스 수: {len(fig.data)}")
            
            # 차트가 비어있는지 확인
            if len(fig.data) == 0:
                st.warning("⚠️ 차트에 표시할 데이터가 없습니다. 안축장 데이터를 입력해주세요.")
                # 기본 백분위 곡선만 표시
                male_data, female_data = get_axial_length_nomogram()
                nomogram_data = male_data if patient_sex == "남" else female_data
                
                # 50% 백분위만 표시
                fig.add_trace(go.Scatter(
                    x=nomogram_data['age'],
                    y=nomogram_data['p50'],
                    mode='lines',
                    name='50% 백분위 (기준선)',
                    line=dict(color='black', width=2, dash='solid'),
                    showlegend=True
                ))
            
            # 차트 표시 (페이지에 꽉 차게)
            st.plotly_chart(
                fig, 
                use_container_width=True, 
                height=600,
                config={
                    'displayModeBar': True,
                    'displaylogo': False,
                    'modeBarButtonsToAdd': ['drawline', 'eraseshape'],
                    'scrollZoom': True,
                    'doubleClick': 'reset+autosize'
                }
            )
            
            # nomogram 정보 표시
            if dob is not None:
                st.info("📅 Nomogram : British Journal of Ophthalmology 2023;107:167-175. ")
                st.info("🎯 Fitting 버튼(🎯): 환자 데이터 기준으로 +3개월 범위 보기")
                st.info("📏 Autoscale 버튼(📏): 전체 범위로 되돌리기")
                st.info("📊 Y축 범위 슬라이더: Y축 간격을 0.5배~3배로 조절 가능")
            else:
                st.warning("⚠️ 생년월일을 입력하면 나이 기준으로 백분위 곡선과 비교할 수 있습니다.")
                st.info("📅 Nomogram : British Journal of Ophthalmology 2023;107:167-175. ")
                st.info("🔍 마우스 휠로 줌, 드래그로 팬, 더블클릭으로 리셋이 가능합니다.")
                st.info("🎯 Fitting 버튼(🎯): 환자 데이터 기준으로 +3개월 범위 보기")
                st.info("📏 Autoscale 버튼(📏): 전체 범위로 되돌리기")
                st.info("📊 Y축 범위 슬라이더: Y축 간격을 0.5배~3배로 조절 가능")
            
            # 안축장 Raw Data 테이블 추가
            st.markdown("---")
            st.markdown("##### 📊 안축장 Raw Data")
            
            # 표시용 데이터 준비
            display_df_axl = df.copy()
            display_df_axl['date'] = display_df_axl['date'].dt.strftime('%Y-%m-%d')
            
            # 필요한 컬럼만 선택하고 컬럼명 변경 (안축장만)
            display_columns = ['date', 'OD_mm', 'OS_mm']
            display_df_axl = display_df_axl[display_columns]
            
            # 컬럼명을 한글로 변경
            column_mapping = {
                'date': '측정일자',
                'OD_mm': '우안(OD) 안축장(mm)',
                'OS_mm': '좌안(OS) 안축장(mm)'
            }
            
            display_df_axl.columns = [column_mapping.get(col, col) for col in display_df_axl.columns]
            
            # 치료/관리 정보를 같은 테이블에 추가
            if 'remarks' in df.columns:
                # 치료/관리 정보를 문자열로 변환
                treatment_info = df['remarks'].apply(lambda x: '; '.join(x) if isinstance(x, list) and x else '')
                display_df_axl['치료/관리'] = treatment_info
            else:
                display_df_axl['치료/관리'] = ''
            
            # 통합된 테이블 표시 (좌측 정렬)
            st.dataframe(display_df_axl, use_container_width=True, hide_index=True)
            
            # 안축장 데이터 수정 기능 - 테이블 형태로 개선
            st.markdown("##### ✏️ 안축장 데이터 수정")
            
            if not display_df_axl.empty:
                # 각 행별로 수정 버튼과 삭제 버튼을 테이블 형태로 표시
                for idx, row in display_df_axl.iterrows():
                    original_date = row['측정일자']
                    original_idx = df[df['date'].dt.strftime('%Y-%m-%d') == original_date].index[0]
                    
                    with st.expander(f"📅 {original_date} - OD: {row['우안(OD) 안축장(mm)']}mm, OS: {row['좌안(OS) 안축장(mm)']}mm", expanded=False):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            edit_date = st.date_input(
                                "측정일자",
                                value=pd.to_datetime(original_date).date(),
                                key=f"edit_axl_date_{idx}"
                            )
                        with col2:
                            edit_od_mm = st.number_input(
                                "우안(OD) 안축장(mm)",
                                min_value=15.0,
                                max_value=35.0,
                                value=float(row['우안(OD) 안축장(mm)']),
                                step=0.01,
                                key=f"edit_axl_od_{idx}"
                            )
                        with col3:
                            edit_os_mm = st.number_input(
                                "좌안(OS) 안축장(mm)",
                                min_value=15.0,
                                max_value=35.0,
                                value=float(row['좌안(OS) 안축장(mm)']),
                                step=0.01,
                                key=f"edit_axl_os_{idx}"
                            )
                        
                        # 치료/관리 수정
                        current_remarks = df.loc[original_idx, 'remarks'] if isinstance(df.loc[original_idx, 'remarks'], list) else []
                        edit_remarks = st.multiselect(
                            "치료/관리",
                            REMARK_OPTIONS,
                            default=current_remarks,
                            key=f"edit_axl_remarks_{idx}"
                        )
                        
                        col_btn1, col_btn2 = st.columns(2)
                        with col_btn1:
                            if st.button("💾 수정 저장", use_container_width=True, key=f"save_axl_edit_{idx}"):
                                # 데이터 수정
                                st.session_state.data_axl.loc[original_idx, 'date'] = pd.to_datetime(edit_date)
                                st.session_state.data_axl.loc[original_idx, 'OD_mm'] = edit_od_mm
                                st.session_state.data_axl.loc[original_idx, 'OS_mm'] = edit_os_mm
                                st.session_state.data_axl.loc[original_idx, 'remarks'] = edit_remarks
                                
                                # 날짜순 정렬
                                st.session_state.data_axl = st.session_state.data_axl.sort_values("date").reset_index(drop=True)
                                
                                st.success(f"{original_date} 안축장 데이터가 수정되었습니다!")
                                if name: save_bundle(name)
                                st.rerun()
                        
                        with col_btn2:
                            if st.button("🗑️ 삭제", use_container_width=True, type="secondary", key=f"delete_axl_row_{idx}"):
                                # 데이터 삭제
                                st.session_state.data_axl = st.session_state.data_axl.drop(original_idx).reset_index(drop=True)
                                
                                st.success(f"{original_date} 안축장 데이터가 삭제되었습니다!")
                                if name: save_bundle(name)
                                st.rerun()
            else:
                st.info("수정할 안축장 데이터가 없습니다.")
        
        elif graph_type == "굴절이상" and has_re:
            df = st.session_state.data_re.copy()
            
            # 생년월일 정보 가져오기
            dob = st.session_state.meta.get("dob")
            
            # 나이 계산
            if dob is not None:
                patient_ages = df["date"].apply(lambda d: _years_between(pd.Timestamp(dob), d))
                # 마이너스 나이 필터링
                valid_age_mask = patient_ages >= 0
                if not valid_age_mask.all():
                    invalid_count = (~valid_age_mask).sum()
                    st.warning(f"⚠️ 굴절이상 차트: {invalid_count}개의 생년월일 이전 날짜 데이터가 제외되었습니다.")
                
                # 유효한 데이터만 사용
                df_filtered = df[valid_age_mask]
                x_data = patient_ages[valid_age_mask]
                x_title = "나이 (연)"
            else:
                df_filtered = df
                x_data = df["date"]
                x_title = "날짜"
            
            # Plotly 차트
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_data, y=df_filtered["OD_SE"], mode="lines+markers", name="우안 SE"))
            fig.add_trace(go.Scatter(x=x_data, y=df_filtered["OS_SE"], mode="lines+markers", name="좌안 SE"))
            
            # 레이아웃 설정
            fig.update_layout(
                title="굴절이상 추이",
                xaxis=dict(
                    title=x_title,
                    titlefont=dict(size=12),
                    tickfont=dict(size=10),
                    showgrid=True,
                    gridcolor='rgba(128,128,128,0.3)',
                    gridwidth=1,
                    zeroline=True,
                    zerolinecolor='rgba(128,128,128,0.5)',
                    zerolinewidth=1,
                    showline=True,
                    linecolor='rgba(128,128,128,0.8)',
                    linewidth=1,
                    showticklabels=True,
                    tickmode='linear',
                    dtick=1
                ),
                yaxis=dict(
                    title="구면대응 (D)",
                    titlefont=dict(size=12),
                    tickfont=dict(size=10),
                    showgrid=True,
                    gridcolor='rgba(128,128,128,0.3)',
                    gridwidth=1,
                    zeroline=True,
                    zerolinecolor='rgba(128,128,128,0.5)',
                    zerolinewidth=1,
                    showline=True,
                    linecolor='rgba(128,128,128,0.8)',
                    linewidth=1,
                    showticklabels=True,
                    tickmode='linear',
                    dtick=1,
                    autorange="reversed"
                ),
                hovermode='closest',
                plot_bgcolor='rgba(255,255,255,1)',
                paper_bgcolor='rgba(255,255,255,1)',
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                ),
                margin=dict(l=50, r=50, t=50, b=50)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 굴절이상 Raw Data 테이블 추가
            st.markdown("---")
            st.markdown("##### 📊 굴절이상 Raw Data")
            
            # 표시용 데이터 준비
            display_df_re = df.copy()
            display_df_re['date'] = display_df_re['date'].dt.strftime('%Y-%m-%d')
            
            # 필요한 컬럼만 선택하고 컬럼명 변경
            display_columns = ['date', 'OD_sph', 'OD_cyl', 'OD_axis', 'OD_SE', 'OS_sph', 'OS_cyl', 'OS_axis', 'OS_SE']
            display_df_re = display_df_re[display_columns]
            
            # 컬럼명을 한글로 변경
            column_mapping = {
                'date': '측정일자',
                'OD_sph': '우안 구면(D)',
                'OD_cyl': '우안 원주(D)',
                'OD_axis': '우안 축각도(°)',
                'OD_SE': '우안 구면대응(D)',
                'OS_sph': '좌안 구면(D)',
                'OS_cyl': '좌안 원주(D)',
                'OS_axis': '좌안 축각도(°)',
                'OS_SE': '좌안 구면대응(D)'
            }
            
            display_df_re.columns = [column_mapping.get(col, col) for col in display_df_re.columns]
            
            # 치료/관리 정보를 같은 테이블에 추가
            if 'remarks' in df.columns:
                # 치료/관리 정보를 문자열로 변환
                treatment_info = df['remarks'].apply(lambda x: '; '.join(x) if isinstance(x, list) and x else '')
                display_df_re['치료/관리'] = treatment_info
            else:
                display_df_re['치료/관리'] = ''
            
            # 통합된 테이블 표시 (좌측 정렬)
            st.dataframe(display_df_re, use_container_width=True, hide_index=True)
            
            # 굴절이상 데이터 수정 기능 - 테이블 형태로 개선
            st.markdown("##### ✏️ 굴절이상 데이터 수정")
            
            if not display_df_re.empty:
                # 각 행별로 수정 버튼과 삭제 버튼을 테이블 형태로 표시
                for idx, row in display_df_re.iterrows():
                    original_date_re = row['측정일자']
                    original_idx_re = df[df['date'].dt.strftime('%Y-%m-%d') == original_date_re].index[0]
                    
                    with st.expander(f"📅 {original_date_re} - OD: {row['우안 구면(D)']}D, OS: {row['좌안 구면(D)']}D", expanded=False):
                        col1, col2 = st.columns(2)
                        with col1:
                            edit_date_re = st.date_input(
                                "측정일자",
                                value=pd.to_datetime(original_date_re).date(),
                                key=f"edit_re_date_{idx}"
                            )
                            
                            st.markdown("**우안(OD)**")
                            edit_od_sph = st.number_input(
                                "구면(D)",
                                min_value=-20.0,
                                max_value=10.0,
                                value=float(row['우안 구면(D)']),
                                step=0.25,
                                key=f"edit_re_od_sph_{idx}"
                            )
                            edit_od_cyl = st.number_input(
                                "원주(D)",
                                min_value=-8.0,
                                max_value=0.0,
                                value=float(row['우안 원주(D)']),
                                step=0.25,
                                key=f"edit_re_od_cyl_{idx}"
                            )
                            edit_od_axis = st.number_input(
                                "축각도(°)",
                                min_value=1,
                                max_value=180,
                                value=int(row['우안 축각도(°)']),
                                key=f"edit_re_od_axis_{idx}"
                            )
                        
                        with col2:
                            st.write("")  # 공간 맞추기
                            st.markdown("**좌안(OS)**")
                            edit_os_sph = st.number_input(
                                "구면(D)",
                                min_value=-20.0,
                                max_value=10.0,
                                value=float(row['좌안 구면(D)']),
                                step=0.25,
                                key=f"edit_re_os_sph_{idx}"
                            )
                            edit_os_cyl = st.number_input(
                                "원주(D)",
                                min_value=-8.0,
                                max_value=0.0,
                                value=float(row['좌안 원주(D)']),
                                step=0.25,
                                key=f"edit_re_os_cyl_{idx}"
                            )
                            edit_os_axis = st.number_input(
                                "축각도(°)",
                                min_value=1,
                                max_value=180,
                                value=int(row['좌안 축각도(°)']),
                                key=f"edit_re_os_axis_{idx}"
                            )
                        
                        # 치료/관리 수정
                        current_remarks_re = df.loc[original_idx_re, 'remarks'] if isinstance(df.loc[original_idx_re, 'remarks'], list) else []
                        edit_remarks_re = st.multiselect(
                            "치료/관리",
                            REMARK_OPTIONS,
                            default=current_remarks_re,
                            key=f"edit_re_remarks_{idx}"
                        )
                        
                        col_btn1, col_btn2 = st.columns(2)
                        with col_btn1:
                            if st.button("💾 수정 저장", use_container_width=True, key=f"save_re_edit_{idx}"):
                                # 구면대응 계산
                                edit_od_se = edit_od_sph + edit_od_cyl / 2.0
                                edit_os_se = edit_os_sph + edit_os_cyl / 2.0
                                
                                # 데이터 수정
                                st.session_state.data_re.loc[original_idx_re, 'date'] = pd.to_datetime(edit_date_re)
                                st.session_state.data_re.loc[original_idx_re, 'OD_sph'] = edit_od_sph
                                st.session_state.data_re.loc[original_idx_re, 'OD_cyl'] = edit_od_cyl
                                st.session_state.data_re.loc[original_idx_re, 'OD_axis'] = edit_od_axis
                                st.session_state.data_re.loc[original_idx_re, 'OS_sph'] = edit_os_sph
                                st.session_state.data_re.loc[original_idx_re, 'OS_cyl'] = edit_os_cyl
                                st.session_state.data_re.loc[original_idx_re, 'OS_axis'] = edit_os_axis
                                st.session_state.data_re.loc[original_idx_re, 'OD_SE'] = edit_od_se
                                st.session_state.data_re.loc[original_idx_re, 'OS_SE'] = edit_os_se
                                st.session_state.data_re.loc[original_idx_re, 'remarks'] = edit_remarks_re
                                
                                # 날짜순 정렬
                                st.session_state.data_re = st.session_state.data_re.sort_values("date").reset_index(drop=True)
                                
                                st.success(f"{original_date_re} 굴절이상 데이터가 수정되었습니다!")
                                if name: save_bundle(name)
                                st.rerun()
                        
                        with col_btn2:
                            if st.button("🗑️ 삭제", use_container_width=True, type="secondary", key=f"delete_re_row_{idx}"):
                                # 데이터 삭제
                                st.session_state.data_re = st.session_state.data_re.drop(original_idx_re).reset_index(drop=True)
                                
                                st.success(f"{original_date_re} 굴절이상 데이터가 삭제되었습니다!")
                                if name: save_bundle(name)
                                st.rerun()
            else:
                st.info("수정할 굴절이상 데이터가 없습니다.")
        
        elif graph_type == "이중축 (안축장 + 굴절이상)" and has_axl and has_re:
            # 이중축 그래프 구현
            st.markdown("##### 이중축 그래프 (안축장 + 굴절이상)")
            
            # 안축장 데이터
            df_axl = st.session_state.data_axl.copy()
            # 굴절이상 데이터 (SE 사용)
            df_re = st.session_state.data_re.copy()
            
            # 생년월일 정보 가져오기
            dob = st.session_state.meta.get("dob")
            
            # 나이 계산
            if dob is not None:
                axl_ages = df_axl["date"].apply(lambda d: _years_between(pd.Timestamp(dob), d))
                re_ages = df_re["date"].apply(lambda d: _years_between(pd.Timestamp(dob), d))
                
                # 마이너스 나이 필터링
                axl_valid_mask = axl_ages >= 0
                re_valid_mask = re_ages >= 0
                
                if not axl_valid_mask.all():
                    invalid_count = (~axl_valid_mask).sum()
                    st.warning(f"⚠️ 이중축 차트(안축장): {invalid_count}개의 생년월일 이전 날짜 데이터가 제외되었습니다.")
                
                if not re_valid_mask.all():
                    invalid_count = (~re_valid_mask).sum()
                    st.warning(f"⚠️ 이중축 차트(굴절이상): {invalid_count}개의 생년월일 이전 날짜 데이터가 제외되었습니다.")
                
                # 유효한 데이터만 사용
                df_axl_filtered = df_axl[axl_valid_mask]
                df_re_filtered = df_re[re_valid_mask]
                axl_x_data = axl_ages[axl_valid_mask]
                re_x_data = re_ages[re_valid_mask]
                x_title = "나이 (연)"
            else:
                df_axl_filtered = df_axl
                df_re_filtered = df_re
                axl_x_data = df_axl["date"]
                re_x_data = df_re["date"]
                x_title = "날짜"
            
            # 공통 날짜 찾기 (필터링된 데이터 사용)
            axl_dates = set(df_axl_filtered["date"].dt.date.astype(str))
            re_dates = set(df_re_filtered["date"].dt.date.astype(str))
            common_dates = axl_dates.intersection(re_dates)
            if len(common_dates) > 0:
                # 이중축 그래프
                fig = go.Figure()
                
                # 안축장 (왼쪽 Y축)
                fig.add_trace(go.Scatter(
                    x=axl_x_data, y=df_axl_filtered["OD_mm"], 
                    mode="lines+markers", name="OD 안축장 (mm)",
                    yaxis="y"
                ))
                fig.add_trace(go.Scatter(
                    x=axl_x_data, y=df_axl_filtered["OS_mm"], 
                    mode="lines+markers", name="OS 안축장 (mm)",
                    yaxis="y"
                ))
                
                # 굴절이상 (오른쪽 Y축) - 절대값으로 표시
                fig.add_trace(go.Scatter(
                    x=re_x_data, y=abs(df_re_filtered["OD_SE"]), 
                    mode="lines+markers", name="OD SE (D) 절대값",
                    yaxis="y2"
                ))
                fig.add_trace(go.Scatter(
                    x=re_x_data, y=abs(df_re_filtered["OS_SE"]), 
                    mode="lines+markers", name="OS SE (D) 절대값",
                    yaxis="y2"
                ))
                
                # 레이아웃 설정
                fig.update_layout(
                    title="안축장 + 굴절이상 이중축 그래프",
                    xaxis=dict(
                        title=x_title,
                        titlefont=dict(size=12),
                        tickfont=dict(size=10),
                        showgrid=True,
                        gridcolor='rgba(128,128,128,0.3)',
                        gridwidth=1,
                        zeroline=True,
                        zerolinecolor='rgba(128,128,128,0.5)',
                        zerolinewidth=1,
                        showline=True,
                        linecolor='rgba(128,128,128,0.8)',
                        linewidth=1,
                        showticklabels=True,
                        tickmode='linear',
                        dtick=1
                    ),
                    yaxis=dict(
                        title="안축장 (mm)", 
                        side="left",
                        titlefont=dict(size=12),
                        tickfont=dict(size=10),
                        showgrid=True,
                        gridcolor='rgba(128,128,128,0.3)',
                        gridwidth=1,
                        zeroline=True,
                        zerolinecolor='rgba(128,128,128,0.5)',
                        zerolinewidth=1,
                        showline=True,
                        linecolor='rgba(128,128,128,0.8)',
                        linewidth=1,
                        showticklabels=True,
                        tickmode='linear',
                        dtick=1
                    ),
                    yaxis2=dict(
                        title="굴절이상 절대값 (D)", 
                        side="right", 
                        overlaying="y",
                        titlefont=dict(size=12),
                        tickfont=dict(size=10),
                        showgrid=False,
                        zeroline=True,
                        zerolinecolor='rgba(128,128,128,0.5)',
                        zerolinewidth=1,
                        showline=True,
                        linecolor='rgba(128,128,128,0.8)',
                        linewidth=1,
                        showticklabels=True,
                        tickmode='linear',
                        dtick=1
                    ),
                    hovermode="x unified",
                    plot_bgcolor='rgba(255,255,255,1)',
                    paper_bgcolor='rgba(255,255,255,1)',
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01
                    ),
                    margin=dict(l=50, r=50, t=50, b=50)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 이중축 Raw Data 테이블 추가
                st.markdown("---")
                st.markdown("##### 📊 이중축 Raw Data")
                
                # 공통 날짜에 대한 데이터만 추출하여 테이블 생성
                common_dates_list = sorted(list(common_dates))
                
                # 안축장과 굴절이상 데이터를 날짜별로 병합
                merged_data = []
                for date_str in common_dates_list:
                    date_obj = pd.to_datetime(date_str)
                    
                    # 안축장 데이터 찾기
                    axl_row = df_axl[df_axl['date'].dt.date.astype(str) == date_str]
                    od_mm = axl_row['OD_mm'].iloc[0] if not axl_row.empty else np.nan
                    os_mm = axl_row['OS_mm'].iloc[0] if not axl_row.empty else np.nan
                    
                    # 굴절이상 데이터 찾기
                    re_row = df_re[df_re['date'].dt.date.astype(str) == date_str]
                    od_se = re_row['OD_SE'].iloc[0] if not re_row.empty else np.nan
                    os_se = re_row['OS_SE'].iloc[0] if not re_row.empty else np.nan
                    
                    merged_data.append({
                        '측정일자': date_str,
                        '우안 안축장(mm)': f"{od_mm:.2f}" if np.isfinite(od_mm) else "",
                        '좌안 안축장(mm)': f"{os_mm:.2f}" if np.isfinite(os_mm) else "",
                        '우안 구면대응(D)': f"{od_se:.2f}" if np.isfinite(od_se) else "",
                        '좌안 구면대응(D)': f"{os_se:.2f}" if np.isfinite(os_se) else ""
                    })
                
                # 병합된 데이터를 데이터프레임으로 변환
                merged_df = pd.DataFrame(merged_data)
                st.dataframe(merged_df, use_container_width=True, hide_index=True)
                
            else:
                st.warning("안축장과 굴절이상 데이터의 공통 날짜가 없습니다.")
        
        # 각막곡률과 각막두께 데이터 항상 표시
        if has_k:
            st.markdown("---")
            st.markdown("##### 📊 각막곡률 데이터")
            df_k = st.session_state.data_k.copy()
            
            if not df_k.empty:
                # 표시용 데이터 준비 (K1, K2, meanK 모두 포함)
                display_df_k = df_k.copy()
                display_df_k['date'] = display_df_k['date'].dt.strftime('%Y-%m-%d')
                
                # 필요한 컬럼만 선택하고 컬럼명 변경
                display_df_k = display_df_k[['date', 'OD_K1', 'OD_K2', 'OD_meanK', 'OS_K1', 'OS_K2', 'OS_meanK']]
                display_df_k.columns = ['측정일자', 'OD_K1', 'OD_K2', 'OD_meanK', 'OS_K1', 'OS_K2', 'OS_meanK']
                
                # 좌측 정렬로 표시하고 너비 제한
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.dataframe(display_df_k, use_container_width=False, width=600)
            else:
                st.info("각막곡률 데이터가 없습니다.")
        
        if has_ct:
            st.markdown("---")
            st.markdown("##### 📊 각막두께 데이터")
            df_ct = st.session_state.data_ct.copy()
            
            if not df_ct.empty:
                # 표시용 데이터 준비
                display_df_ct = df_ct.copy()
                display_df_ct['date'] = display_df_ct['date'].dt.strftime('%Y-%m-%d')
                display_df_ct = display_df_ct[['date', 'OD_ct', 'OS_ct']]
                display_df_ct.columns = ['측정일자', '우안(OD)', '좌안(OS)']
                
                # 좌측 정렬로 표시하고 너비 제한
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.dataframe(display_df_ct, use_container_width=False, width=400)
            else:
                st.info("각막두께 데이터가 없습니다.")
        
        # 치료/관리 현황 표시
        st.markdown("---")
        st.subheader("💊 치료/관리 현황")
        
        # 치료/관리 옵션과 사용 일자 수집
        treatment_history = []
        
        if has_axl:
            for _, row in st.session_state.data_axl.iterrows():
                if isinstance(row['remarks'], list) and row['remarks']:
                    for remark in row['remarks']:
                        treatment_history.append({
                            'date': row['date'].strftime('%Y-%m-%d'),
                            'treatment': remark,
                            'type': '안축장'
                        })
        
        if has_re:
            for _, row in st.session_state.data_re.iterrows():
                if isinstance(row['remarks'], list) and row['remarks']:
                    for remark in row['remarks']:
                        treatment_history.append({
                            'date': row['date'].strftime('%Y-%m-%d'),
                            'treatment': remark,
                            'type': '굴절이상'
                        })
        
        if has_k:
            for _, row in st.session_state.data_k.iterrows():
                if isinstance(row['remarks'], list) and row['remarks']:
                    for remark in row['remarks']:
                        treatment_history.append({
                            'date': row['date'].strftime('%Y-%m-%d'),
                            'treatment': remark,
                            'type': '각막곡률'
                        })
        
        if has_ct:
            for _, row in st.session_state.data_ct.iterrows():
                if isinstance(row['remarks'], list) and row['remarks']:
                    for remark in row['remarks']:
                        treatment_history.append({
                            'date': row['date'].strftime('%Y-%m-%d'),
                            'treatment': remark,
                            'type': '각막두께'
                        })
        
        if treatment_history:
            st.write("**치료/관리 현황:**")
            
            # 치료/관리 옵션별로 그룹화
            from collections import defaultdict
            treatment_by_option = defaultdict(list)
            
            for item in treatment_history:
                treatment_by_option[item['treatment']].append({
                    'date': item['date'],
                    'type': item['type']
                })
            
            # 표 데이터 생성 (시작일, 종료일, 총치료기간만)
            table_data = []
            for treatment, history in treatment_by_option.items():
                # 각 치료/관리 옵션별로 시작일과 종료일 계산
                dates_used = [item['date'] for item in history]
                dates_used.sort()
                
                if len(dates_used) >= 2:
                    start_date = dates_used[0]
                    end_date = dates_used[-1]
                    # 총치료기간 계산 (일수)
                    from datetime import datetime
                    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
                    total_days = (end_dt - start_dt).days
                    total_period = f"{total_days}일"
                else:
                    start_date = dates_used[0] if dates_used else ""
                    end_date = dates_used[0] if dates_used else ""
                    total_period = "1일"
                
                # 시작일과 종료일에 나이와 개월수 추가
                start_age_info = ""
                end_age_info = ""
                
                if start_date and st.session_state.meta.get("dob"):
                    try:
                        # 단일 날짜에 대한 나이 계산
                        from datetime import datetime
                        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                        dob = st.session_state.meta.get("dob")
                        if dob:
                            dob_dt = datetime.combine(dob, datetime.min.time())
                            age_days = (start_dt - dob_dt).days
                            age_years = age_days // 365
                            age_months = (age_days % 365) // 30
                            start_age_info = f"{start_date}\n({age_years}세 {age_months}개월)"
                        else:
                            start_age_info = start_date
                    except:
                        start_age_info = start_date
                else:
                    start_age_info = start_date
                
                if end_date and st.session_state.meta.get("dob"):
                    try:
                        # 단일 날짜에 대한 나이 계산
                        from datetime import datetime
                        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
                        dob = st.session_state.meta.get("dob")
                        if dob:
                            dob_dt = datetime.combine(dob, datetime.min.time())
                            age_days = (end_dt - dob_dt).days
                            age_years = age_days // 365
                            age_months = (age_days % 365) // 30
                            end_age_info = f"{end_date}\n({age_years}세 {age_months}개월)"
                        else:
                            end_age_info = end_date
                    except:
                        end_age_info = end_date
                else:
                    end_age_info = end_date
                
                row = [treatment, start_age_info, end_age_info, total_period]
                table_data.append(row)
            
            # 표 헤더 생성
            headers = ["치료/관리 옵션", "시작일 (나이)", "종료일 (나이)", "총치료기간"]
            
            # Streamlit 표로 표시
            import pandas as pd
            df_treatment = pd.DataFrame(table_data, columns=headers)
            st.dataframe(df_treatment, use_container_width=True)
            
            # 추가 설명
            st.caption("💡 표에서 시작일과 종료일에 나이와 개월수가 함께 표시됩니다.")
        else:
            st.info("아직 치료/관리 정보가 없습니다.")
    else:
        st.info("시각화할 데이터가 없습니다. 데이터를 먼저 입력해주세요.")

# =========================
#  탭 3: 예측 분석
# =========================
with tab3:
    
    if has_axl or has_re or has_k or has_ct:

        # 분석할 데이터 선택 (체크박스)
        st.markdown("**분석할 데이터 선택:**")
        col1, col2 = st.columns(2)
        with col1:
            analyze_axl = st.checkbox("안축장", value=has_axl, disabled=not has_axl, key="analyze_axl_checkbox")
        with col2:
            analyze_re = st.checkbox("굴절이상", value=has_re, disabled=not has_re, key="analyze_re_checkbox")
        
        # 선택된 데이터에 대한 예측 분석 표시
        if analyze_axl and has_axl:
            df_axl = st.session_state.data_axl.copy()
            ages_axl = _age_at_dates(df_axl["date"], st.session_state.meta.get("dob"), st.session_state.meta.get("current_age"))
            
            if ages_axl is not None:
                st.markdown("#### 안축장 20세 예측")
                
                # 예측 모델 선택
                model_choice_axl = st.radio("예측 모델", ["회귀(선형/로그)", "추천(자동/치료조정)"], horizontal=True, key="axl_model_tab3")
                
                if model_choice_axl.startswith("회귀"):
                    trend_mode_axl = st.radio("추세선 모드", ["선형(Linear)", "로그(Log)"], horizontal=True, key="axl_trend_tab3")
                    mode_key_axl = "linear" if trend_mode_axl.startswith("선형") else "log"
                    res_od_axl = _trend_and_predict(ages_axl, df_axl["OD_mm"], mode=mode_key_axl)
                    res_os_axl = _trend_and_predict(ages_axl, df_axl["OS_mm"], mode=mode_key_axl)
                else:
                    res_od_axl = _recommendation_predict(ages_axl, df_axl["OD_mm"], df_axl.get("remarks"))
                    res_os_axl = _recommendation_predict(ages_axl, df_axl["OS_mm"], df_axl.get("remarks"))
                
                if res_od_axl["valid"] or res_os_axl["valid"]:
                    # 간단한 예측 결과
                    col1, col2 = st.columns(2)
                    with col1:
                        if res_od_axl["valid"]:
                            st.success(f"**OD**: {res_od_axl['last_value']:.2f}mm → {res_od_axl['pred_at_20']:.2f}mm")
                    with col2:
                        if res_os_axl["valid"]:
                            st.success(f"**OS**: {res_os_axl['last_value']:.2f}mm → {res_os_axl['pred_at_20']:.2f}mm")
                    
                    # 상세 예측 그래프 (Matplotlib)
                    if st.checkbox("상세 예측 그래프 보기", key="show_axl_detail_tab3"):
                        fig2, ax2 = plt.subplots(figsize=(10, 5), dpi=120)
                        x_age = np.array(ages_axl, dtype=float)
                        finite_mask = np.isfinite(x_age)
                        log_mask = finite_mask & (x_age > 0)
                        
                        y_od = np.array(df_axl["OD_mm"], dtype=float)
                        y_os = np.array(df_axl["OS_mm"], dtype=float)
                        
                        if model_choice_axl.startswith("회귀"):
                            current_mode = "log" if ('mode_key_axl' in locals() and mode_key_axl == "log") else "linear"
                            mask_use = log_mask if current_mode == "log" else finite_mask
                            x_plot = x_age[mask_use]
                            y_od_plot = y_od[mask_use]
                            y_os_plot = y_os[mask_use]
                            
                            if x_plot.size > 0:
                                x_min = float(np.nanmin(x_plot))
                                x_max = float(np.nanmax(x_plot))
                                x_from = max(0.1, x_min) if current_mode == "log" else x_min
                                x_to = max(20.0, x_max)
                                ax2.scatter(x_plot, y_od_plot, label="OD 데이터", alpha=0.7)
                                ax2.scatter(x_plot, y_os_plot, label="OS 데이터", marker="s", alpha=0.7)
                                
                                x_line = np.linspace(x_from, x_to, 200)
                                if res_od_axl["valid"]:
                                    y_line_od = (res_od_axl["slope"] * np.log(x_line) + res_od_axl["intercept"]) if current_mode == "log" else (res_od_axl["slope"] * x_line + res_od_axl["intercept"]) 
                                    ax2.plot(x_line, y_line_od, label=f"OD 추세({current_mode})", linestyle="--")
                                if res_os_axl["valid"]:
                                    y_line_os = (res_os_axl["slope"] * np.log(x_line) + res_os_axl["intercept"]) if current_mode == "log" else (res_os_axl["slope"] * x_line + res_os_axl["intercept"]) 
                                    ax2.plot(x_line, y_line_os, label=f"OS 추세({current_mode})", linestyle="--")
                        else:
                            mode_od = res_od_axl.get("chosen_mode") or "linear"
                            mode_os = res_os_axl.get("chosen_mode") or "linear"
                            mask_od = log_mask if mode_od == "log" else finite_mask
                            mask_os = log_mask if mode_os == "log" else finite_mask
                            x_plot_od = x_age[mask_od]
                            x_plot_os = x_age[mask_os]
                            y_od_plot = y_od[mask_od]
                            y_os_plot = y_os[mask_od]
                            
                            if x_plot_od.size > 0:
                                x_min = float(np.nanmin(x_plot_od))
                                x_max = float(np.nanmax(x_plot_od))
                                x_from = max(0.1, x_min) if mode_od == "log" else x_min
                                x_to = max(20.0, x_max)
                                ax2.scatter(x_plot_od, y_od_plot, label="OD 데이터", alpha=0.7)
                                x_line = np.linspace(x_from, x_to, 200)
                                if res_od_axl["valid"]:
                                    y_line_od = (res_od_axl["slope"] * np.log(x_line) + res_od_axl["intercept"]) if mode_od == "log" else (res_od_axl["slope"] * x_line + res_od_axl["intercept"]) 
                                    ax2.plot(x_line, y_line_od, label=f"OD 추세({mode_od})", linestyle="--")
                            if x_plot_os.size > 0:
                                x_min = float(np.nanmin(x_plot_os))
                                x_max = float(np.nanmax(x_plot_os))
                                x_from = max(0.1, x_min) if mode_os == "log" else x_min
                                x_to = max(20.0, x_max)
                                ax2.scatter(x_plot_os, y_os_plot, label="OS SE 절대값", marker="s", alpha=0.7)
                                x_line = np.linspace(x_from, x_to, 200)
                                if res_os_axl["valid"]:
                                    y_line_os = (res_os_axl["slope"] * np.log(x_line) + res_os_axl["intercept"]) if mode_os == "log" else (res_os_axl["slope"] * x_line + res_os_axl["intercept"]) 
                                    ax2.plot(x_line, y_line_os, label=f"OS 추세({mode_os})", linestyle="--")
                        
                        # 20세 예측점
                        ax2.axvline(20.0, color='red', linestyle=":", alpha=0.6, label="20세")
                        if res_od_axl["valid"] and np.isfinite(res_od_axl["pred_at_20"]):
                            ax2.scatter([20.0], [res_od_axl["pred_at_20"]], marker="*", s=150, color="blue", label=f"OD 20세: {res_od_axl['pred_at_20']:.2f}mm")
                        if res_os_axl["valid"] and np.isfinite(res_os_axl["pred_at_20"]):
                            ax2.scatter([20.0], [res_os_axl['pred_at_20']], marker="*", s=150, color="orange", label=f"OS 20세: {res_os_axl['pred_at_20']:.2f}mm")
                        
                        ax2.set_xlabel("연령 (년)")
                        ax2.set_ylabel("안축장 (mm)")
                        ax2.set_title("안축장 추이 및 20세 예측")
                        ax2.grid(True, alpha=0.3)
                        ax2.legend()
                        ax2.set_xlim(left=x_from, right=x_to)
                        
                        st.pyplot(fig2, use_container_width=True)
            else:
                st.info("20세 예측을 위해 생년월일을 입력하세요.")
        
        if analyze_re and has_re:
            df_re = st.session_state.data_re.copy()
            ages_re = _age_at_dates(df_re["date"], st.session_state.meta.get("dob"), st.session_state.meta.get("current_age"))
            
            if ages_re is not None:
                st.markdown("#### 굴절이상 20세 예측")
                
                # 예측 모델 선택
                model_choice_re = st.radio("예측 모델", ["회귀(선형/로그)", "추천(자동/치료조정)"], horizontal=True, key="re_model_tab3")
                
                if model_choice_re.startswith("회귀"):
                    trend_mode_re = st.radio("추세선 모드", ["선형(Linear)", "로그(Log)"], horizontal=True, key="re_trend_tab3")
                    mode_key_re = "linear" if trend_mode_re.startswith("선형") else "log"
                    res_od_re = _trend_and_predict(ages_re, df_re["OD_SE"], mode=mode_key_re)
                    res_os_re = _trend_and_predict(ages_re, df_re["OS_SE"], mode=mode_key_re)
                else:
                    res_od_re = _recommendation_predict(ages_re, df_re["OD_SE"], df_re.get("remarks"))
                    res_os_re = _recommendation_predict(ages_re, df_re["OS_SE"], df_re.get("remarks"))
                
                if res_od_re["valid"] or res_os_re["valid"]:
                    # 간단한 예측 결과
                    unit = "D"
                    col1, col2 = st.columns(2)
                    with col1:
                        if res_od_re["valid"]:
                            st.success(f"**OD**: {res_od_re['last_value']:.2f}{unit} → {res_od_re['pred_at_20']:.2f}{unit}")
                    with col2:
                        if res_os_re["valid"]:
                            st.success(f"**OS**: {res_os_re['last_value']:.2f}{unit} → {res_os_re['pred_at_20']:.2f}{unit}")
                    
                    # 상세 예측 그래프 (Matplotlib)
                    if st.checkbox("상세 예측 그래프 보기", key="show_re_detail_tab3"):
                        fig2, ax2 = plt.subplots(figsize=(10, 5), dpi=120)
                        x_age = np.array(ages_re, dtype=float)
                        finite_mask = np.isfinite(x_age)
                        log_mask = finite_mask & (x_age > 0)
                        
                        # 굴절이상은 절대값으로 변환하여 표시 (위쪽이 큰 값)
                        y_od = np.abs(np.array(df_re["OD_SE"], dtype=float))
                        y_os = np.abs(np.array(df_re["OS_SE"], dtype=float))
                        
                        # x_from, x_to 변수 초기화
                        x_from = 0.1
                        x_to = 20.0
                        
                        if model_choice_re.startswith("회귀"):
                            # 회귀 모드: trend_mode_re에 따라 모드 선택
                            if 'trend_mode_re' in locals():
                                current_mode = "log" if trend_mode_re.startswith("로그") else "linear"
                            else:
                                current_mode = "linear"  # 기본값
                            
                            mask_use = log_mask if current_mode == "log" else finite_mask
                            x_plot = x_age[mask_use]
                            y_od_plot = y_od[mask_use]
                            y_os_plot = y_os[mask_use]
                            
                            if x_plot.size > 0:
                                x_min = float(np.nanmin(x_plot))
                                x_max = float(np.nanmax(x_plot))
                                x_from = max(0.1, x_min) if current_mode == "log" else x_min
                                x_to = max(20.0, x_max)
                                ax2.scatter(x_plot, y_od_plot, label="OD SE 절대값", alpha=0.7)
                                ax2.scatter(x_plot, y_os_plot, label="OS SE 절대값", marker="s", alpha=0.7)
                                x_line = np.linspace(x_from, x_to, 200)
                                if res_od_re["valid"]:
                                    y_line_od = (res_od_re["slope"] * np.log(x_line) + res_od_re["intercept"]) if current_mode == "log" else (res_od_re["slope"] * x_line + res_od_re["intercept"])
                                    ax2.plot(x_line, y_line_od, label=f"OD 추세({current_mode})", linestyle="--")
                                if res_os_re["valid"]:
                                    y_line_os = (res_os_re["slope"] * np.log(x_line) + res_os_re["intercept"]) if current_mode == "log" else (res_os_re["slope"] * x_line + res_os_re["intercept"])
                                    ax2.plot(x_line, y_line_os, label=f"OS 추세({current_mode})", linestyle="--")
                        else:
                            # 추천 모드: chosen_mode에 따라 모드 선택
                            mode_od = res_od_re.get("chosen_mode") or "linear"
                            mode_os = res_os_re.get("chosen_mode") or "linear"
                            mask_od = log_mask if mode_od == "log" else finite_mask
                            mask_os = log_mask if mode_os == "log" else finite_mask
                            x_plot_od = x_age[mask_od]
                            x_plot_os = x_age[mask_os]
                            y_od_plot = y_od[mask_od]
                            y_os_plot = y_os[mask_od]
                            
                            # x_from, x_to 계산
                            if x_plot_od.size > 0:
                                x_min_od = float(np.nanmin(x_plot_od))
                                x_max_od = float(np.nanmax(x_plot_od))
                                x_from_od = max(0.1, x_min_od) if mode_od == "log" else x_min_od
                                x_to_od = max(20.0, x_max_od)
                                x_from = min(x_from, x_from_od)
                                x_to = max(x_to, x_to_od)
                                
                                ax2.scatter(x_plot_od, y_od_plot, label="OD SE 절대값", alpha=0.7)
                                x_line = np.linspace(x_from_od, x_to_od, 200)
                                if res_od_re["valid"]:
                                    y_line_od = (res_od_re["slope"] * np.log(x_line) + res_od_re["intercept"]) if mode_od == "log" else (res_od_re["slope"] * x_line + res_od_re["intercept"])
                                    ax2.plot(x_line, y_line_od, label=f"OD 추세({mode_od})", linestyle="--")
                            
                            if x_plot_os.size > 0:
                                x_min_os = float(np.nanmin(x_plot_os))
                                x_max_os = float(np.nanmax(x_plot_os))
                                x_from_os = max(0.1, x_min_os) if mode_os == "log" else x_min_os
                                x_to_os = max(20.0, x_max_os)
                                x_from = min(x_from, x_from_os)
                                x_to = max(x_to, x_to_os)
                                
                                ax2.scatter(x_plot_os, y_os_plot, label="OS SE 절대값", marker="s", alpha=0.7)
                                x_line = np.linspace(x_from_os, x_to_os, 200)
                                if res_os_re["valid"]:
                                    y_line_os = (res_os_re["slope"] * np.log(x_line) + res_os_re["intercept"]) if mode_os == "log" else (res_os_re["slope"] * x_line + res_os_re["intercept"])
                                    ax2.plot(x_line, y_line_os, label=f"OS 추세({mode_os})", linestyle="--")

                        
                        # 20세 예측점
                        ax2.axvline(20.0, color='red', linestyle=":", alpha=0.6, label="20세")
                        if res_od_re["valid"] and np.isfinite(res_od_re["pred_at_20"]):
                            ax2.scatter([20.0], [abs(res_od_re["pred_at_20"])], marker="*", s=150, color="blue", label=f"OD 20세: {abs(res_od_re['pred_at_20']):.2f}{unit}")
                        if res_os_re["valid"] and np.isfinite(res_os_re["pred_at_20"]):
                            ax2.scatter([20.0], [abs(res_os_re['pred_at_20'])], marker="*", s=150, color="orange", label=f"OS 20세: {abs(res_os_re['pred_at_20']):.2f}{unit}")
                        
                        ax2.set_xlabel("연령 (년)")
                        ax2.set_ylabel("구면대응 절대값 (D)")
                        ax2.set_title("굴절이상 추이 및 20세 예측 (절대값)")
                        ax2.grid(True, alpha=0.3)
                        ax2.legend()
                        ax2.set_xlim(left=x_from, right=x_to)
                        
                        st.pyplot(fig2, use_container_width=True)
            else:
                st.info("20세 예측을 위해 생년월일을 입력하세요.")
        
        # 2. 치료/관리 현황
        st.subheader("💊 치료/관리 현황")
        
        # 치료/관리 옵션과 사용 일자 수집
        treatment_history = []
        
        if has_axl:
            for _, row in st.session_state.data_axl.iterrows():
                if isinstance(row['remarks'], list) and row['remarks']:
                    for remark in row['remarks']:
                        treatment_history.append({
                            'date': row['date'].strftime('%Y-%m-%d'),
                            'treatment': remark,
                            'type': '안축장'
                        })
        
        if has_re:
            for _, row in st.session_state.data_re.iterrows():
                if isinstance(row['remarks'], list) and row['remarks']:
                    for remark in row['remarks']:
                        treatment_history.append({
                            'date': row['date'].strftime('%Y-%m-%d'),
                            'treatment': remark,
                            'type': '굴절이상'
                        })
        
        if has_k:
            for _, row in st.session_state.data_k.iterrows():
                if isinstance(row['remarks'], list) and row['remarks']:
                    for remark in row['remarks']:
                        treatment_history.append({
                            'date': row['date'].strftime('%Y-%m-%d'),
                            'treatment': remark,
                            'type': '각막곡률'
                        })
        
        if has_ct:
            for _, row in st.session_state.data_ct.iterrows():
                if isinstance(row['remarks'], list) and row['remarks']:
                    for remark in row['remarks']:
                        treatment_history.append({
                            'date': row['date'].strftime('%Y-%m-%d'),
                            'treatment': remark,
                            'type': '각막두께'
                        })
        
        if treatment_history:
            st.write("**치료/관리 현황:**")
            
            # 치료/관리 옵션별로 그룹화
            from collections import defaultdict
            treatment_by_option = defaultdict(list)
            
            for item in treatment_history:
                treatment_by_option[item['treatment']].append({
                    'date': item['date'],
                    'type': item['type']
                })
            
            # 모든 고유 날짜 수집 및 정렬
            all_dates = set()
            for history in treatment_by_option.values():
                for item in history:
                    all_dates.add(item['date'])
            all_dates = sorted(list(all_dates))
            
            # 표 데이터 생성 (이미지 형태와 유사하게)
            table_data = []
            for treatment, history in treatment_by_option.items():
                # 각 치료/관리 옵션별로 시작일과 종료일 계산
                dates_used = [item['date'] for item in history]
                dates_used.sort()
                
                if len(dates_used) >= 2:
                    start_date = dates_used[0]
                    end_date = dates_used[-1]
                    # 총치료기간 계산 (일수)
                    from datetime import datetime
                    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
                    total_days = (end_dt - start_dt).days
                    total_period = f"{total_days}일"
                else:
                    start_date = dates_used[0] if dates_used else ""
                    end_date = dates_used[0] if dates_used else ""
                    total_period = "1일"
                
                # 모든 날짜에 대해 사용 여부 표시
                row = [treatment, start_date, end_date, total_period]
                for date in all_dates:
                    used_on_date = [item for item in history if item['date'] == date]
                    if used_on_date:
                        types = [item['type'] for item in used_on_date]
                        cell_content = ", ".join(types)
                        row.append(cell_content)
                    else:
                        row.append("")
                table_data.append(row)
            
            # 표 헤더 생성
            headers = ["치료/관리 옵션", "시작일", "종료일", "총치료기간"] + all_dates
            
            # Streamlit 표로 표시
            import pandas as pd
            df_treatment = pd.DataFrame(table_data, columns=headers)
            st.dataframe(df_treatment, use_container_width=True)
            
            # 추가 설명
            st.caption("💡 표에서 각 셀은 해당 날짜에 사용된 데이터 타입을 나타내며, 시작일/종료일/총치료기간을 포함합니다.")
        else:
            st.info("아직 치료/관리 정보가 없습니다.")
    
    else:
        st.info("데이터가 없습니다. 데이터 입력 탭에서 데이터를 추가해주세요.")

# =========================
#  탭 4: 설정
# =========================
with tab4:
    st.header("⚙️ 설정")
    st.markdown("각 탭에서 사용할 기본값들을 설정할 수 있습니다.")
    
    # 설정을 섹션별로 구분
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📝 탭1: 데이터 입력 기본값")
        
        # 입력 선택 기본값
        tab1_data_type = st.selectbox(
            "**입력 선택 기본값**",
            ["안축장", "굴절이상", "각막곡률", "각막두께"],
            index=["안축장", "굴절이상", "각막곡률", "각막두께"].index(
                st.session_state.default_settings["tab1_default_data_type"]
            ),
            key="tab1_data_type_default"
        )
        st.session_state.default_settings["tab1_default_data_type"] = tab1_data_type
        
        # 입력 방식 기본값
        tab1_input_method = st.selectbox(
            "**입력 방식 기본값**",
            ["선택입력", "텍스트입력", "이미지(OCR)"],
            index=["선택입력", "텍스트입력", "이미지(OCR)"].index(
                st.session_state.default_settings["tab1_default_input_method"]
            ),
            key="tab1_input_method_default"
        )
        st.session_state.default_settings["tab1_default_input_method"] = tab1_input_method
        

        
        # 기본 치료/관리 옵션
        st.markdown("**기본 치료/관리 옵션**")
        st.session_state.default_settings["tab1_default_remarks"] = st.multiselect(
            "자주 사용하는 옵션들을 미리 선택",
            REMARK_OPTIONS,
            default=st.session_state.default_settings["tab1_default_remarks"],
            key="remarks_default"
        )
    
    with col2:
        st.markdown("### 📊 탭2: 시각화 기본값")
        
        # 그래프 타입 기본값
        tab2_graph_type = st.selectbox(
            "**그래프 타입 기본값**",
            ["안축장", "굴절이상", "이중축 (안축장 + 굴절이상)"],
            index=["안축장", "굴절이상", "이중축 (안축장 + 굴절이상)"].index(
                st.session_state.default_settings["tab2_default_graph_type"]
            ),
            key="tab2_graph_type_default"
        )
        st.session_state.default_settings["tab2_default_graph_type"] = tab2_graph_type
        
        st.markdown("### 🔮 탭3: 예측 분석 기본값")
        
        # 분석할 데이터 기본값
        tab3_analyze_re = st.checkbox(
            "**굴절이상 분석 기본값**",
            value=st.session_state.default_settings["tab3_default_analyze_re"],
            key="tab3_analyze_re_default"
        )
        st.session_state.default_settings["tab3_default_analyze_re"] = tab3_analyze_re
        
        # 예측 모델 기본값
        tab3_model_choice = st.selectbox(
            "**예측 모델 기본값**",
            ["회귀(선형/로그)", "추천(자동/치료조정)"],
            index=["회귀(선형/로그)", "추천(자동/치료조정)"].index(
                st.session_state.default_settings["tab3_default_model_choice"]
            ),
            key="tab3_model_choice_default"
        )
        st.session_state.default_settings["tab3_default_model_choice"] = tab3_model_choice
        
        # 추세선 모드 기본값
        tab3_trend_mode = st.selectbox(
            "**추세선 모드 기본값**",
            ["선형(Linear)", "로그(Log)"],
            index=["선형(Linear)", "로그(Log)"].index(
                st.session_state.default_settings["tab3_default_trend_mode"]
            ),
            key="tab3_trend_mode_default"
        )
        st.session_state.default_settings["tab3_default_trend_mode"] = tab3_trend_mode
        
        # 설정 저장/불러오기
        st.markdown("---")
        st.markdown("### 💾 설정 관리")
        
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            if st.button("설정 저장", use_container_width=True, type="primary"):
                # 설정을 JSON 파일로 저장
                import json
                settings_file = DATA_ROOT / "default_settings.json"
                try:
                    with open(settings_file, 'w', encoding='utf-8') as f:
                        json.dump(st.session_state.default_settings, f, ensure_ascii=False, indent=2)
                    st.success("설정이 저장되었습니다!")
                except Exception as e:
                    st.error(f"설정 저장 실패: {e}")
        
        with col_s2:
            if st.button("설정 불러오기", use_container_width=True, type="secondary"):
                # 설정을 JSON 파일에서 불러오기
                import json
                settings_file = DATA_ROOT / "default_settings.json"
                try:
                    if settings_file.exists():
                        with open(settings_file, 'r', encoding='utf-8') as f:
                            loaded_settings = json.load(f)
                        st.session_state.default_settings.update(loaded_settings)
                        st.success("설정을 불러왔습니다!")
                        st.rerun()
                    else:
                        st.warning("저장된 설정 파일이 없습니다.")
                except Exception as e:
                    st.error(f"설정 불러오기 실패: {e}")
        
        # 현재 설정값 표시
        st.markdown("### 📋 현재 설정값")
        st.json(st.session_state.default_settings)

# フッターメモ
st.markdown(
    """
    **メモ**  
    - 眼軸長は常に屈折異常と一致するとは限りません。
    - 角膜曲率(K1, K2, Mean K)は右眼(OD)と左眼(OS)で区別して保存されます。
    - 角膜厚はマイクロメートル(μm)単位で右眼(OD)と左眼(OS)で区別して保存されます。
    - 詳細な内容については、直接お問い合わせください。 
    - 眼軸長`data.csv`、屈折異常`re_data.csv`、角膜曲率`k_data.csv`、角膜厚`ct_data.csv`で別々に保存されます。  
    - 屈折異常は**球面等価(SE = S + C/2)**をトレンド/予測に使用します。軸(Axis)は保存されますが予測には使用しません。  
    """,
    unsafe_allow_html=True
)
