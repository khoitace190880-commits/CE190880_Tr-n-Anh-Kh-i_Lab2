"""Lab 2 realtime facial emotion recognition app."""
import argparse
import time
import sys
from pathlib import Path
import cv2
import torch
from PIL import Image
import yaml
from detector import FaceDetector
from predict import load_predictor, predict_pil_batch

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

def parse_source(value):
    return int(value) if value.isdigit() else value

def read_config(path):
    p = Path(path)
    if not p.exists():
        return {}
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        print(f"Warning: could not read config {p}: {exc}", file=sys.stderr)
        return {}

def parse_args():
    ap = argparse.ArgumentParser(description="FER-2013 facial emotion recognition")
    ap.add_argument("--source", default=None, help="IP camera URL, webcam index, image path or video path")
    ap.add_argument("--model", choices=["cnn", "transfer"], default=None)
    ap.add_argument("--weights", default=None, help="Optional checkpoint path")
    ap.add_argument("--config", default="config.yaml", help="YAML defaults; command-line values take precedence")
    ap.add_argument("--confidence", type=float, default=None, help="Minimum confidence to show label (0..1)")
    ap.add_argument("--face-detector", choices=["haar"], default="haar")
    ap.add_argument("--no-display", action="store_true", help="Process frames without opening a GUI window")
    ap.add_argument("--max-reconnect-attempts", type=int, default=5, help="Maximum consecutive IP-stream reconnect attempts")
    return ap.parse_args()

def main():
    args = parse_args()
    cfg = read_config(args.config)
    source_text = args.source if args.source is not None else cfg.get("source")
    model_name = args.model if args.model is not None else cfg.get("model", "cnn")
    confidence = args.confidence if args.confidence is not None else float(cfg.get("confidence_threshold", 0.0))
    if not source_text:
        raise SystemExit("Missing --source. Example: python src/app.py --source 0 --model cnn")
    if not 0 <= confidence <= 1:
        raise SystemExit("--confidence must be between 0 and 1")
    weight_cfg = (cfg.get("weights") or {}).get(model_name)
    weights_path = args.weights or weight_cfg or f"weights/{model_name}_best.pth"
    try:
        model, classes, transform, device = load_predictor(model_name, weights_path)
        detector = FaceDetector(args.face_detector)
    except Exception as exc:
        raise SystemExit(f"Could not initialize model/detector: {exc}")
    print(f"Model={model_name} | device={device} | weights={weights_path} | source={source_text}")
    source = parse_source(str(source_text))
    is_image = isinstance(source, str) and Path(source).suffix.lower() in IMAGE_EXTS
    cap = None
    if is_image:
        frame = cv2.imread(source)
        if frame is None:
            raise SystemExit(f"Cannot read image: {source}")
        frames = [frame]
    else:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            if cap is not None: cap.release()
            raise SystemExit(f"Cannot open source: {source_text}. Check the camera URL, network and permissions.")
        # Reduce latency for supported stream backends; ignored if unsupported.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        frames = None
    previous = time.perf_counter()
    fps = 0.0
    reconnects = 0
    try:
        while True:
            if frames is not None:
                if not frames: break
                frame = frames.pop(0)
            else:
                ok, frame = cap.read()
                if not ok or frame is None:
                    if isinstance(source, str) and source.lower().startswith(("http://", "https://", "rtsp://")):
                        reconnects += 1
                        print(f"Camera stream lost; reconnect attempt {reconnects}/{args.max_reconnect_attempts}...", file=sys.stderr)
                        cap.release()
                        if reconnects > max(0, args.max_reconnect_attempts):
                            print("Reconnect limit reached; exiting cleanly.", file=sys.stderr)
                            break
                        time.sleep(min(0.5 * reconnects, 2.0))
                        cap = cv2.VideoCapture(source)
                        if cap.isOpened():
                            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                            reconnects = 0
                        continue
                    print("End of video or frame could not be read.", file=sys.stderr)
                    break
                reconnects = 0
            boxes = detector.detect(frame)
            crops, valid_boxes = [], []
            fh, fw = frame.shape[:2]
            for x, y, w, h in boxes:
                x0, y0 = max(0, x), max(0, y)
                x1, y1 = min(fw, x + w), min(fh, y + h)
                if x1 <= x0 or y1 <= y0: continue
                crop = frame[y0:y1, x0:x1]
                if crop.size == 0: continue
                crops.append(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
                valid_boxes.append((x0, y0, x1-x0, y1-y0))
            predictions = predict_pil_batch(model, crops, transform, classes, device)
            for (x, y, w, h), (label, confidence_value) in zip(valid_boxes, predictions):
                shown = label if confidence_value >= confidence else "uncertain"
                color = (50, 210, 50) if confidence_value >= confidence else (0, 165, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, f"{shown} {confidence_value:.2f}", (x, max(20, y-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
            now = time.perf_counter()
            dt = now - previous
            previous = now
            if dt > 0:
                instant = 1.0 / dt
                fps = instant if fps == 0 else 0.85 * fps + 0.15 * instant
            cv2.putText(frame, f"FPS: {fps:.1f} | Model: {model_name} | Faces: {len(valid_boxes)}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2, cv2.LINE_AA)
            if not args.no_display:
                cv2.imshow("Lab 2 - Emotion Recognition (q to quit)", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"): break
            if is_image:
                output = Path("results") / f"demo_{model_name}.jpg"
                output.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(output), frame)
                print(f"Processed image; annotated output saved to {output}")
                if not args.no_display:
                    cv2.waitKey(0)
                break
    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        if cap is not None: cap.release()
        if not args.no_display: cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
