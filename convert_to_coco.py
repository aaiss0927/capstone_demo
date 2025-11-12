import os
import json
import shutil
from glob import glob
from sklearn.model_selection import train_test_split

# --- 경로 설정 ---
SOURCE_IMAGE_DIR = 'Sample/01.원천데이터'
SOURCE_LABEL_DIR = 'Sample/02.라벨링데이터'
OUTPUT_BASE_DIR = 'YOLO/data/custom'

# --- 클래스 매핑 (원본 JSON의 categories 배열을 그대로 사용) ---
# 'none' (categories_id: 3) 클래스를 제외했습니다.
CATEGORIES = [
    {"category_index": 1, "category_name": "fl"},
    {"category_index": 2, "category_name": "sm"},
]
# 허용된 카테고리 ID 목록 (1과 2만 허용)
ALLOWED_CATEGORY_IDS = [c['category_index'] for c in CATEGORIES]

# --- 분할 비율 ---
TRAIN_RATIO = 0.8  # 80%
VAL_RATIO = 0.1    # 10%
TEST_RATIO = 0.1   # 10%
RANDOM_SEED = 42

def create_output_dirs():
    """필요한 출력 디렉토리를 생성합니다."""
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(OUTPUT_BASE_DIR, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_BASE_DIR, 'labels'), exist_ok=True)
    print(f"출력 폴더 구조 생성 완료: {OUTPUT_BASE_DIR}/{{images, labels}}/{{train, val, test}}")

def process_and_split_data():
    """데이터를 처리하고 분할하여 출력 폴더에 복사 및 통합 JSON을 생성합니다."""
    
    # 1. 모든 JSON 파일 경로 찾기
    all_json_paths = []
    for root, _, files in os.walk(SOURCE_LABEL_DIR):
        for file in files:
            if file.endswith('.json'):
                all_json_paths.append(os.path.join(root, file))
    
    if not all_json_paths:
        print("Error: '02.라벨링데이터'에서 JSON 파일을 찾을 수 없습니다.")
        return

    # 2. Train/Val/Test 분할 (이 부분은 변경 없음)
    test_val_size = VAL_RATIO + TEST_RATIO
    train_paths, test_val_paths = train_test_split(
        all_json_paths, 
        test_size=test_val_size, 
        random_state=RANDOM_SEED
    )
    
    val_paths, test_paths = train_test_split(
        test_val_paths, 
        test_size=TEST_RATIO / test_val_size,
        random_state=RANDOM_SEED
    )
    
    print(f"\n데이터 분할 결과:")
    print(f"Train: {len(train_paths)}개")
    print(f"Validation: {len(val_paths)}개")
    print(f"Test: {len(test_paths)}개")
    
    data_splits = {
        'train': train_paths,
        'val': val_paths,
        'test': test_paths
    }

    # 3. 파일 처리, 복사 및 통합 JSON 생성
    for split_name, json_list in data_splits.items():
        print(f"\n--- {split_name.upper()} 데이터셋 처리 중 ---")
        
        integrated_data = [] 
        
        for json_path in json_list:
            
            # 3-1. JSON 파일 로드
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 3-2. 이미지 복사 (변경 없음)
            base_filename_no_ext = os.path.splitext(os.path.basename(json_path))[0]
            img_source_path_glob = glob(os.path.join(SOURCE_IMAGE_DIR, '**', base_filename_no_ext + '.jpg'), recursive=True)
            
            if not img_source_path_glob:
                print(f"Warning: 이미지 파일 '{data['image']['filename']}'를 찾을 수 없어 건너뜁니다.")
                continue

            img_source_path = img_source_path_glob[0]
            
            img_dest_dir = os.path.join(OUTPUT_BASE_DIR, 'images', split_name)
            shutil.copy(img_source_path, img_dest_dir)

            # 3-3. 데이터를 통합 리스트에 추가 (None 클래스 필터링 로직 추가)
            
            # 원본 데이터의 categories와 annotations를 복사
            new_data = data.copy()
            
            # categories 배열을 필터링된 버전으로 교체
            new_data['categories'] = CATEGORIES
            
            # annotations 배열에서 categories_id가 3인 항목(none)을 제외하고 필터링
            if 'annotations' in data and data['annotations']:
                filtered_annotations = [
                    ann for ann in data['annotations'] 
                    if ann.get('categories_id') in ALLOWED_CATEGORY_IDS
                ]
                new_data['annotations'] = filtered_annotations
            else:
                 # 어노테이션이 아예 없는 경우 빈 리스트 유지
                new_data['annotations'] = []
            
            # 유효한 어노테이션이 하나라도 있거나, 바운딩 박스 정보 외의 메타 정보가 필요한 경우에만 추가
            # 만약 필터링 후 어노테이션이 0개가 되어도, 이미지 메타 정보는 필요할 수 있으므로 리스트에 추가합니다.
            integrated_data.append(new_data)
            
        # 3-4. 통합 JSON 파일 저장 (변경 없음)
        output_json_path = os.path.join(OUTPUT_BASE_DIR, 'labels', f'annotations_{split_name}.json')
        
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(integrated_data, f, indent=4, ensure_ascii=False)
            
        print(f"통합 JSON 파일 저장 완료: {output_json_path}")

    print("\n✅ 모든 데이터 변환 및 필터링 완료.")


if __name__ == "__main__":
    try:
        from sklearn.model_selection import train_test_split
    except ImportError:
        print("🚨 'scikit-learn' 라이브러리가 설치되어 있지 않습니다. 설치해주세요.")
        print("pip install scikit-learn")
        exit()
        
    create_output_dirs()
    process_and_split_data()