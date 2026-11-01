"""웨딩사진 스마트 셀렉 · 포트폴리오용 V2 (YuNet + 설명 가능한 추천/검증)."""
import csv
import hashlib
import io
import time
import hashlib as _model_hashlib
import tempfile
import urllib.request
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageOps

MODEL_FILE = Path(__file__).with_name("face_detection_yunet_2023mar.onnx")
MODEL_URL = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
MODEL_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"


def ensure_model():
    path = MODEL_FILE if MODEL_FILE.is_file() else (Path(tempfile.gettempdir()) / "wedding_yunet_2023mar.onnx")
    if path.is_file() and path.stat().st_size > 100_000:
        if _model_hashlib.sha256(path.read_bytes()).hexdigest() == MODEL_SHA256:
            return path
    try:
        request = urllib.request.Request(MODEL_URL, headers={"User-Agent": "Wedding-Smart-Select/1.0"})
        with urllib.request.urlopen(request, timeout=45) as response:
            data = response.read(2_000_000)
        if _model_hashlib.sha256(data).hexdigest() != MODEL_SHA256:
            raise RuntimeError("공식 모델 파일의 무결성 확인 실패")
        path.write_bytes(data)
        return path
    except Exception as exc:
        raise RuntimeError("YuNet 모델을 내려받지 못했습니다. 잠시 후 재시도해주세요.") from exc


def get_detector():
    if not hasattr(cv2, "FaceDetectorYN"):
        raise RuntimeError("OpenCV에서 FaceDetectorYN을 찾을 수 없습니다.")
    if "_yunet_detector" not in st.session_state:
        path = ensure_model()
        st.session_state["_yunet_detector"] = cv2.FaceDetectorYN.create(str(path), "", (320, 320), 0.8, 0.3, 5000)
    return st.session_state["_yunet_detector"]


def bounded(x):
    return float(np.clip(x, 0, 1))


def score_photo(focus, brightness, contrast, face_boxes, image_shape, mode):
    """손으로 정한 탐색용 가중치. 학습된 미학 평가가 아님."""
    sharp = bounded(np.log1p(focus) / np.log1p(800))
    exposure = bounded(1 - abs(brightness - 130) / 125)
    contrast_norm = bounded(contrast / 75)
    h, w = image_shape[:2]
    largest_face = max((float(b[2] * b[3]) for b in face_boxes), default=0.0)
    relative_face = bounded(largest_face / max(1, h * w * 0.06))

    if mode == "인물 중심":
        parts = {"선명도": 30 * sharp, "노출": 25 * exposure,
                 "대비": 15 * contrast_norm, "얼굴 검출": 20 * bool(len(face_boxes)),
                 "얼굴 크기": 10 * relative_face}
    else:
        parts = {"선명도": 40 * sharp, "노출": 35 * exposure, "대비": 25 * contrast_norm}
    score = round(min(100, sum(parts.values())), 1)
    return score, parts


def make_hash(image):
    # 기존 버전과 동일한 8×8 평균 해시. 유사도는 참고값이며 동일 장면 판정은 아님.
    small = np.asarray(image.convert("L").resize((8, 8)), dtype=np.float32)
    bits = (small > small.mean()).flatten()
    return sum(int(v) << i for i, v in enumerate(bits))


def inspect_image(name, raw):
    with Image.open(io.BytesIO(raw)) as source:
        im = ImageOps.exif_transpose(source).convert("RGB")
        im.thumbnail((1000, 1000))
    rgb = np.array(im)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    h, w = rgb.shape[:2]
    face_model = get_detector()
    face_model.setInputSize((w, h))
    _, detected = face_model.detect(bgr)
    boxes = [] if detected is None else [tuple(map(float, row)) for row in detected]
    return {
        "filename": name,
        "focus": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 1),
        "brightness": round(float(gray.mean()), 1),
        "contrast": round(float(gray.std()), 1),
        "face_count": len(boxes),
        "face_confidence": round(max((box[14] for box in boxes), default=0.0), 3),
        "boxes": boxes,
        "image_shape": (h, w),
        "hash": make_hash(im),
    }


def reasons(item):
    messages = []
    if item["focus"] < 60:
        messages.append("흐림 의심")
    if item["brightness"] > 205:
        messages.append("과다 노출 의심")
    if item["brightness"] < 75:
        messages.append("저조도 의심")
    if item["face_count"] == 0:
        messages.append("얼굴 미검출(수동 확인)")
    return ", ".join(messages) if messages else "뚜렷한 기술적 결함 없음"


def overlay(raw, boxes):
    with Image.open(io.BytesIO(raw)) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        image.thumbnail((1000, 1000))
    draw = ImageDraw.Draw(image)
    for box in boxes:
        x, y, w, h = (int(z) for z in box[:4])
        draw.rectangle((x, y, x + w, y + h), outline="#18cf8b", width=4)
        draw.text((max(0, x), max(0, y - 17)), f"{box[14]:.2f}", fill="#18cf8b")
    return image


def select_photos(records, count, suppress_duplicates, threshold):
    ranked = sorted(records, key=lambda item: item["score"], reverse=True)
    chosen = []
    for item in ranked:
        similar = next((old for old in chosen if
                        (item["hash"] ^ old["hash"]).bit_count() <= threshold), None)
        if suppress_duplicates and similar:
            item["result"] = "유사 후보(검토)"
        elif len(chosen) < count:
            item["result"] = "추천"
            chosen.append(item)
        else:
            item["result"] = "추천 수량 초과"
    return ranked, chosen


def csv_bytes(rows):
    output = io.StringIO()
    fields = ["filename", "score", "face_count", "face_confidence",
              "focus", "brightness", "contrast", "reason", "result", "human_label"]
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for item in rows:
        writer.writerow({key: item.get(key, "") for key in fields})
    return output.getvalue().encode("utf-8-sig")


def main():
    st.set_page_config(page_title="웨딩사진 스마트 셀렉 | Portfolio V2", page_icon="💍", layout="wide")
    st.title("💍 웨딩사진 스마트 셀렉 · 웹 데모")
    st.caption("프로젝트 데모 · 테스트 버전 / 다량의 웨딩사진 일괄 처리를 지원하는 상용 서비스가 아닙니다.")
    st.caption("YuNet 얼굴 위치 시각화 · 목적별 규칙 점수 · 추천 근거 · 사용자 평가")
    st.info("사전학습 YuNet은 얼굴의 위치를 감지합니다. 눈 감음, 감정, 인물의 관계 또는 사진의 예술성은 판단하지 않습니다.")
    left, middle, right = st.columns(3)
    with left:
        mode = st.selectbox("선별 목적", ["인물 중심", "분위기 중심"])
    with middle:
        number = st.number_input("추천 사진 수", min_value=1, max_value=150, value=20)
    with right:
        suppress = st.checkbox("유사 사진 자동 제외(실험용)", value=False,
                               help="평균 해시 오판으로 좋은 사진이 빠질 수 있어 기본값은 끔")
    threshold = st.slider("유사도 임계값(중복 제외 선택 시 적용)", 0, 16, 4)
    st.caption("웹 데모: 사진은 분석을 위해 Streamlit 서버로 전송되며, 별도 데이터베이스나 디스크에 저장하는 기능은 없습니다. 실제 인물 사진은 당사자 동의를 받은 경우에만 업로드하세요.")
    files = st.file_uploader(
        "사진 업로드(JPG/PNG/WEBP) — 한 번에 50장까지",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
    )
    if not files:
        st.info("직접 사진을 업로드하면 분석을 시작합니다. 개인 소유·사용 동의를 받은 사진만 사용해주세요.")
        return
    if len(files) > 50 or len({f.name for f in files}) != len(files):
        st.error("클라우드 데모는 한 번에 50장 이하, 파일명 중복 없이 업로드해주세요.")
        return
    if sum(f.size for f in files) > 100 * 1024 * 1024:
        st.error("클라우드 데모는 총 100MB 이하의 파일만 지원합니다.")
        return
    raw = {f.name: f.getvalue() for f in files}
    records, problems = [], []
    t0 = time.perf_counter()
    local_cache = st.session_state.setdefault("_session_metadata", {})
    # Analysis metadata is cached only inside this visitor's own browser session.
    if len(local_cache) > 100:
        local_cache.clear()
    for filename, data in raw.items():
        try:
            cache_key = (filename, hashlib.sha256(data).hexdigest())
            if cache_key not in local_cache:
                local_cache[cache_key] = inspect_image(filename, data)
            item = dict(local_cache[cache_key])
            item["score"], _ = score_photo(item["focus"], item["brightness"], item["contrast"],
                                             item["boxes"], item["image_shape"], mode)
            item["reason"] = reasons(item)
            records.append(item)
        except Exception as error:
            problems.append(f"{filename}: {type(error).__name__}: {error}")
    elapsed = time.perf_counter() - t0
    if problems:
        st.error("분석 실패: " + " | ".join(problems[:3]))
    if not records:
        return
    ranked, chosen = select_photos(records, int(number), suppress, threshold)
    a, b, c, d = st.columns(4)
    a.metric("분석 성공", f"{len(records)}장")
    b.metric("추천", f"{len(chosen)}장")
    c.metric("얼굴 미검출", f"{sum(r['face_count'] == 0 for r in records)}장")
    d.metric("이번 실행 경과", f"{elapsed:.1f}초", help="캐시 이용 여부에 따라 달라지는 화면 처리시간")

    st.subheader("추천 사진 · 점수와 판단 근거")
    tiles = st.columns(4)
    for idx, item in enumerate(chosen):
        with tiles[idx % 4]:
            st.image(raw[item["filename"]], use_container_width=True)
            st.caption(f"{item['filename']} | {item['score']}/100점 | 감지 {item['face_count']}명")
            st.caption(item["reason"])

    with st.expander("🔎 YuNet 얼굴 위치/신뢰도 확인", expanded=True):
        target = st.selectbox("검토할 사진", [row["filename"] for row in ranked])
        item = next(row for row in ranked if row["filename"] == target)
        st.image(overlay(raw[target], item["boxes"]), use_container_width=False,
                 caption=f"표시된 초록색 상자가 얼굴 검출 위치입니다. 검출 {item['face_count']}명, 최대 신뢰도 {item['face_confidence']:.3f}")
        st.warning("검출된 사람 수가 아니라 '검출된 얼굴 영역 수'입니다. 오탐과 미검출은 직접 확인해야 합니다.")

    st.subheader("전수 분석 결과")
    display = pd.DataFrame([{key: row[key] for key in
                             ["filename", "score", "face_count", "face_confidence", "focus",
                              "brightness", "contrast", "reason", "result"]} for row in ranked])
    st.dataframe(display, use_container_width=True, hide_index=True)

    with st.expander("🧪 사람의 선택으로 추천 품질 평가 (포트폴리오용)"):
        st.write("각 사진을 직접 보고 '우수' 또는 '제외'로 평가하세요. 모든 사진을 평가하면 Precision@K와 Recall을 계산합니다.")
        labels_table = pd.DataFrame({"filename": [r["filename"] for r in ranked],
                                     "AI 결과": [r["result"] for r in ranked],
                                     "human_label": ["미평가"] * len(ranked)})
        labeled = st.data_editor(
            labels_table, hide_index=True, use_container_width=True, key="human_labels_editor",
            disabled=["filename", "AI 결과"],
            column_config={"human_label": st.column_config.SelectboxColumn(
                "사람의 판단", options=["미평가", "우수", "제외"], required=True)},
        )
        labels = dict(zip(labeled["filename"], labeled["human_label"]))
        for row in ranked:
            row["human_label"] = labels.get(row["filename"], "미평가")
        reviewed = sum(label != "미평가" for label in labels.values())
        st.write(f"평가 완료: {reviewed}/{len(records)}장")
        if reviewed == len(records):
            positive = {name for name, label in labels.items() if label == "우수"}
            selected = {r["filename"] for r in chosen}
            precision = len(positive & selected) / len(selected) if selected else 0.0
            recall = len(positive & selected) / len(positive) if positive else 0.0
            st.metric("Precision@K (추천 중 우수 비율)", f"{precision:.1%}")
            st.metric("Recall (우수 사진 중 추천 비율)", f"{recall:.1%}")
            st.caption("우수 판단은 사용자 주관적 기준이며, 합성 사진 평가 결과를 실제 고객 정확도로 주장해서는 안 됩니다.")
        else:
            st.info("전체 사진에 사람이 직접 라벨을 부여하면 정확도 지표를 표시합니다.")

    with st.expander("📊 기존 결과 CSV와 개선 버전 비교 (선택)"):
        previous = st.file_uploader("기존 app_yunet.py의 분석 결과 CSV", type=["csv"], key="old_report")
        if previous is not None:
            try:
                prior = pd.read_csv(previous, encoding="utf-8-sig")
                if not {"filename", "result"}.issubset(set(prior.columns)):
                    st.error("기존 CSV에 filename, result 열이 있어야 합니다.")
                else:
                    prior_selected = set(prior.loc[prior["result"] == "추천", "filename"].astype(str))
                    new_selected = {row["filename"] for row in chosen}
                    overlap = len(prior_selected & new_selected)
                    st.write(f"기존 추천 {len(prior_selected)}장 / V2 추천 {len(new_selected)}장 / 공통 {overlap}장")
                    if reviewed == len(records):
                        good = {name for name, label in labels.items() if label == "우수"}
                        old_known = prior_selected & set(labels)
                        old_prec = (len(old_known & good) / len(old_known)) if old_known else 0.0
                        new_prec = (len(new_selected & good) / len(new_selected)) if new_selected else 0.0
                        st.write(f"기존 Precision@K: {old_prec:.1%} → V2 Precision@K: {new_prec:.1%}")
                        st.caption("같은 이미지 집합·추천 장수·사용자 라벨로 평가했는지 반드시 확인하세요.")
            except Exception as error:
                st.error(f"기존 CSV를 읽지 못했습니다: {error}")

    report = csv_bytes(ranked)
    st.download_button("📄 분석·평가 CSV 다운로드", report, "portfolio_photo_report.csv", "text/csv")
    archive = io.BytesIO()
    with ZipFile(archive, "w", ZIP_DEFLATED) as z:
        for position, row in enumerate(chosen, 1):
            z.writestr(f"selected/{position:03d}_{row['filename']}", raw[row["filename"]])
        z.writestr("portfolio_photo_report.csv", report)
    st.download_button("🗂️ 추천 사진 ZIP 다운로드", archive.getvalue(),
                       "portfolio_selected_photos.zip", "application/zip")
    st.caption("포트폴리오 데모입니다. 사전학습 AI 모델을 활용하며, 모델 학습·감정/관계 인식을 수행하지 않습니다. 다수 사용자·대용량 실서비스를 위한 검증은 수행하지 않았습니다.")


if __name__ == "__main__":
    main()