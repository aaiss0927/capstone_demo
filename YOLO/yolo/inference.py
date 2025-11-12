# inference.pymport sys
import json
from pathlib import Path

import torch
from PIL import Image
from omegaconf import OmegaConf
from hydra import compose, initialize
import argparse

# --- YOLO repo 내부 모듈 ---
# 프로젝트 루트 추가
project_root = Path().resolve()
sys.path.append(str(project_root))

from yolo import (  # noqa: E402
    AugmentationComposer,
    Config,
    PostProcess,
    create_converter,
    create_model,
    draw_bboxes,
)

parser = argparse.ArgumentParser()
parser.add_argument("--image_path", required=True, help="이미지 경로")
args = parser.parse_args()

# =========================
# 사용자 설정
# =========================
CONFIG_PATH = "./config"  # "./yolo/config"
CONFIG_NAME = "config"
MODEL = "v9-m"  # config의 모델 키 (예: v9-n, v9-s, v9-m, v9-l ...)

# 디바이스
DEVICE = "cuda:0"  # 없으면 "cpu"
device = torch.device(DEVICE)

# 입력 이미지
IMAGE_PATH = args.image_path

# 커스텀 가중치 (학습한 .pt/.ckpt)
CUSTOM_MODEL_PATH = "/shared/home/kdd/HZ/capstone/YOLO/weights/ema_cleaned_class_2.pt"

# 클래스 정보
# - 네가 쓰는 클래스 이름으로 바꿔라. (예: ["figure"])
# - cfg에 coco 80클래스가 있더라도 visualization/JSON에는 이 리스트를 쓴다.
CLASS_NAMES = ["fl", "sm"]
CLASS_NUM = len(CLASS_NAMES)

# NMS/신뢰도 임계값
MIN_CONF = 0.5
MIN_IOU = 0.5
MAX_BOX = 300

# 출력 파일
OUT_IMAGE = "./output.jpg"
OUT_JSON = "./output.json"


def _resolve_image_size(sz_cfg):
    """
    Hydra/OmegaConf의 ListConfig/str/scalar 모두 안전하게 (H, W) 튜플로 변환.
    """
    sz = OmegaConf.to_container(sz_cfg, resolve=True)
    if isinstance(sz, (list, tuple)):
        if len(sz) == 2:
            h, w = int(sz[0]), int(sz[1])
            return h, w
        elif len(sz) == 1:
            h = w = int(sz[0])
            return h, w
        else:
            raise ValueError(f"image_size length must be 1 or 2, got {len(sz)}")
    # scalar인 경우
    return int(sz), int(sz)


def main():
    print(project_root)

    # ========= Hydra 구성 로드 =========
    with initialize(
        config_path=CONFIG_PATH, version_base=None, job_name="inference_job"
    ):
        cfg: Config = compose(
            config_name=CONFIG_NAME,
            overrides=[
                "task=inference",
                f"task.data.source={IMAGE_PATH}",
                f"model={MODEL}",
            ],
        )

    # NMS 세팅 덮어쓰기
    cfg.task.nms.min_confidence = float(MIN_CONF)
    cfg.task.nms.min_iou = float(MIN_IOU)
    cfg.task.nms.max_bbox = int(MAX_BOX)

    print("cfg is")
    print(OmegaConf.to_container(cfg, resolve=True))

    # ========= 이미지 크기 안정 처리 =========
    H, W = _resolve_image_size(cfg.image_size)

    # ========= 모델 구성/로딩 =========
    model = create_model(cfg.model, class_num=CLASS_NUM).to(device)
    model.eval()

    # 가중치 로딩 (strict=False로 유연하게)
    if CUSTOM_MODEL_PATH and Path(CUSTOM_MODEL_PATH).exists():
        sd = torch.load(CUSTOM_MODEL_PATH, map_location=device)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]

        # ema 키/프리픽스 등 잡다한 키가 섞여 있어도 로드되도록
        try:
            missing, unexpected = model.load_state_dict(sd, strict=False)
            if missing:
                print(f"[load_state_dict] missing keys: {len(missing)} (표시는 생략)")
            if unexpected:
                print(
                    f"[load_state_dict] unexpected keys: {len(unexpected)} (표시는 생략)"
                )
        except Exception as e:
            # 실패 시 cpu에서 다시 만들어 재시도
            print(f"load_state_dict(strict=False) 실패: {e}\n-> CPU로 재시도")
            model_cpu = create_model(cfg.model, class_num=CLASS_NUM).cpu()
            model_cpu.load_state_dict(sd, strict=False)
            model = model_cpu.to(device).eval()

    # ========= 더미 포워드로 stride/anchor 초기화 =========
    with torch.no_grad():
        _ = model(torch.zeros(1, 3, H, W, device=device, dtype=torch.float32))

    # ========= 변환/후처리 구성 =========
    transform = AugmentationComposer([], (H, W))
    converter = create_converter(
        cfg.model.name, model, cfg.model.anchor, (H, W), device
    )
    post_process = PostProcess(converter, cfg.task.nms)

    # ========= 이미지 로드 & 변환 =========
    pil_image = Image.open(IMAGE_PATH).convert("RGB")
    image_tensor, _, rev_tensor = transform(pil_image)

    image_tensor = image_tensor.to(device, dtype=torch.float32)[None]  # [1,3,H,W]
    rev_tensor = rev_tensor.to(device)[None]

    # ========= 추론 =========
    with torch.no_grad():
        raw_pred = model(image_tensor)
        pred_bbox = post_process(
            raw_pred, rev_tensor
        )  # 보통 [tensor(N,6)] or tensor(N,6)

    # ========= 그리기/저장 =========
    output_image = draw_bboxes(pil_image, pred_bbox, idx2label=CLASS_NAMES)
    output_image.save(OUT_IMAGE)
    print(f"✅ Output image saved at: {OUT_IMAGE}")

    # ========= JSON 저장 =========
    bboxes = pred_bbox

    if isinstance(bboxes, list) or bboxes.ndim == 3:
        bboxes = bboxes[0]

    detections = []
    for bbox in bboxes:
        class_id, x_min, y_min, x_max, y_max, *conf = [float(val) for val in bbox]
        x1, x2 = min(x_min, x_max), max(x_min, x_max)
        y1, y2 = min(y_min, y_max), max(y_min, y_max)
        w = x2 - x1
        h = y2 - y1

        detections.append(
            {
                "bbox": [float(x1), float(y1), float(w), float(h)],  # COCO xywh
                "score": float(*conf),
                "class_id": class_id,
            }
        )

    payload = {
        "meta": {
            "image_path": str(IMAGE_PATH),
            "image_size": {"width": pil_image.width, "height": pil_image.height},
        },
        "detections": detections,
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"✅ Detections JSON saved at: {OUT_JSON}")


if __name__ == "__main__":
    main()
