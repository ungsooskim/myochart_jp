#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
일본어 버전 테스트 스크립트
"""

def test_imports():
    """필요한 패키지들이 모두 import되는지 테스트"""
    try:
        import matplotlib
        print(f"✅ matplotlib {matplotlib.__version__} - OK")
        
        import pandas
        print(f"✅ pandas {pandas.__version__} - OK")
        
        import numpy
        print(f"✅ numpy {numpy.__version__} - OK")
        
        import streamlit
        print(f"✅ streamlit {streamlit.__version__} - OK")
        
        import plotly
        print(f"✅ plotly {plotly.__version__} - OK")
        
        import pytesseract
        print(f"✅ pytesseract - OK")
        
        from PIL import Image
        print(f"✅ Pillow - OK")
        
        print("\n🎌 모든 패키지가 정상적으로 설치되어 있습니다!")
        print("일본어 버전을 실행할 수 있습니다.")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import 오류: {e}")
        print("필요한 패키지를 설치해주세요:")
        print("pip install -r requirements.txt")
        return False

def test_japanese_fonts():
    """일본어 폰트 설정 테스트"""
    try:
        import matplotlib.pyplot as plt
        import platform
        
        # 일본어 폰트 설정
        system = platform.system().lower()
        if "windows" in system:
            font_family = "Yu Gothic"
        elif "darwin" in system:
            font_family = "Hiragino Sans"
        else:
            font_family = "Noto Sans CJK JP"
        
        plt.rcParams["font.family"] = font_family
        print(f"✅ 일본어 폰트 설정 완료: {font_family}")
        
        # 간단한 일본어 텍스트 테스트
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "眼軸長・屈折異常推移及び20歳予測", 
                ha='center', va='center', fontsize=16)
        ax.set_title("일본어 폰트 테스트")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        # 테스트 이미지 저장
        plt.savefig("japanese_font_test.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print("✅ 일본어 폰트 테스트 이미지 생성: japanese_font_test.png")
        return True
        
    except Exception as e:
        print(f"❌ 폰트 테스트 오류: {e}")
        return False

if __name__ == "__main__":
    print("🎌 일본어 버전 테스트 시작...\n")
    
    # 패키지 import 테스트
    print("1. 패키지 import 테스트:")
    import_success = test_imports()
    
    if import_success:
        print("\n2. 일본어 폰트 테스트:")
        font_success = test_japanese_fonts()
        
        if font_success:
            print("\n🎉 모든 테스트 통과! 일본어 버전을 실행할 수 있습니다.")
            print("실행 명령어: streamlit run axlml2_jp.py")
        else:
            print("\n⚠️ 폰트 테스트 실패. 기본 폰트로 실행됩니다.")
    else:
        print("\n❌ 패키지 설치가 필요합니다.")
