import torch
import os
import argparse
from pathlib import Path

def clean_ema_weights(ckpt_path, output_path, ema_key='ema'):
    """
    PyTorch Lightning/Custom YOLO .ckpt 파일에서 EMA 가중치만 추출하여 저장합니다.
    """
    ckpt_path = Path(ckpt_path)
    output_path = Path(output_path)
    
    if not ckpt_path.exists():
        print(f"🚨 Error: 체크포인트 파일이 존재하지 않습니다: {ckpt_path}")
        return

    print(f"🔍 체크포인트 파일 로드 중: {ckpt_path.name}")
    try:
        # CKPT 파일 로드
        data = torch.load(ckpt_path, map_location='cpu')
    except Exception as e:
        print(f"🚨 Error: 체크포인트 로드 실패. 파일이 손상되었거나 형식이 잘못되었습니다. ({e})")
        return

    state_dict = None
    
    # 1. 'state_dict' 키에서 모델 상태 추출 (Lightning 기본)
    if isinstance(data, dict) and 'state_dict' in data:
        state_dict = data['state_dict']
        print(f"✅ 'state_dict' 키에서 가중치 딕셔너리 발견.")
    elif isinstance(data, dict):
        # 2. 파일 자체가 이미 state_dict일 경우 (일부 pt 파일 형식)
        state_dict = data
        print(f"✅ 파일에서 직접 가중치 딕셔너리 발견.")
    else:
        print("🚨 Error: 로드된 파일에서 'state_dict' 키나 유효한 딕셔너리를 찾을 수 없습니다.")
        return

    # 3. EMA 가중치 추출
    # EMA 모델 가중치는 일반적으로 'ema.ema_model.' 또는 'ema.' 접두사를 가집니다.
    ema_state_dict = {}
    
    # EMA 접두사 확인 및 필터링
    ema_prefix_full = f'{ema_key}.ema_model.' 
    ema_prefix_short = f'{ema_key}.'

    found_ema = False
    
    for k, v in state_dict.items():
        # Case 1: 'ema.ema_model.' 접두사가 붙은 경우 (가장 흔함)
        if k.startswith(ema_prefix_full):
            # 'ema.ema_model.' 접두사 제거
            new_key = k[len(ema_prefix_full):]
            ema_state_dict[new_key] = v
            found_ema = True
        # Case 2: 'ema.' 접두사가 붙은 경우 (때때로 사용됨)
        elif not found_ema and k.startswith(ema_prefix_short) and 'model.' in k:
            # 'ema.' 접두사 제거 (모델 관련 키에만 적용)
            new_key = k[len(ema_prefix_short):]
            ema_state_dict[new_key] = v
            found_ema = True
        # Case 3: 가중치가 최상위 레벨에 직접 있을 경우 (순수 pt 파일 형태)
        # 이 경우, cleaning이 필요 없음. 일단 통과.

    if not found_ema and not ema_state_dict and len(state_dict) > 0:
        # EMA 키를 찾지 못했고, state_dict는 있지만 필터링되지 않았다면
        # 이미 cleaned 상태이거나, 다른 키 구조일 수 있으므로 전체 state_dict를 사용합니다.
        ema_state_dict = state_dict
        print("⚠️ Warning: 표준 EMA 접두사(ema.ema_model.)를 찾지 못했습니다. 전체 state_dict를 사용합니다.")
    
    if not ema_state_dict:
        print("🚨 Error: 유효한 EMA 가중치 또는 모델 가중치를 추출하지 못했습니다.")
        return

    # 4. 저장
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ema_state_dict, output_path)
    print(f"\n✅ EMA Cleaning 완료. 추출된 가중치 저장: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean EMA weights from a PyTorch Lightning CKPT file.")
    parser.add_argument("--ckpt_path", 
                        required=True, 
                        type=str,
                        help="학습 결과로 나온 .ckpt 파일의 경로입니다.")
    parser.add_argument("--output_path", 
                        default="./ema_cleaned.pt",
                        type=str,
                        help="EMA 가중치를 저장할 .pt 파일의 경로입니다.")
    
    args = parser.parse_args()
    
    # 예시 경로를 기준으로 추론 코드에 사용할 경로를 기본으로 설정합니다.
    default_ckpt_path = "/shared/home/kdd/HZ/capstone/YOLO/runs/train/v9-dev/YOLO/kv7qxeq9/checkpoints/epoch=3-step=216.ckpt"
    default_output_path = "./YOLO/weights/ema_cleaned_class_2.pt"

    print("--- EMA Weight Cleaning Utility ---")
    
    # 명령줄에서 경로를 주지 않았다면 기본 경로 사용 (사용자 편의)
    if args.ckpt_path == default_ckpt_path:
        print(f"Info: 기본 CKPT 경로를 사용합니다. (경로를 변경하려면 --ckpt_path 옵션을 사용하세요.)")
    
    if args.output_path == "./ema_cleaned.pt":
        # 사용자 편의를 위해 inference.py에 설정된 경로를 기본 출력 경로로 제안합니다.
        args.output_path = default_output_path
        print(f"Info: 추론 코드에 맞는 기본 출력 경로를 사용합니다: {args.output_path}")


    clean_ema_weights(args.ckpt_path, args.output_path)