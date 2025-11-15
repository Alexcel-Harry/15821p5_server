#!/usr/bin/env python3
"""
EchoEngine + ByteTrack integration for single mobile (Gabriel pipeline compatible).

Features:
- Uses ultralytics YOLO model for detection
- Sends detections to ByteTrack (pip `bytetracker` or ultralytics' built-in byte tracker)
- Returns text payloads with normalized xywh,class,conf,track_id;
- Includes --local-test <video.mp4> mode to run locally and measure times (bypass Gabriel)
"""
import argparse
import logging
import io
import cv2
import numpy as np
import torch
import time
import threading

from ultralytics import YOLO
from turbojpeg import TurboJPEG

# Gabriel related imports kept for compatibility with your server pipeline
from gabriel_server import cognitive_engine
from gabriel_server import local_engine
from gabriel_protocol import gabriel_pb2

# Try to import common ByteTrack implementations
BYTETracker = None
try:
    from bytetracker import BYTETracker as BYTETracker  # pip package common name
except Exception:
    BYTETracker = None

# fallback: ultralytics internal tracker (if installed)
if BYTETracker is None:
    try:
        from ultralytics.trackers.byte_tracker import BYTETracker as UBYTE
        BYTETracker = UBYTE
    except Exception:
        BYTETracker = None


SOURCE_NAME = "echo"
INPUT_QUEUE_MAXSIZE = 60
DEFAULT_PORT = 9099
NUM_TOKENS = 3


def _choose_status(*candidates):
    """
    Try to pick the first existing enum status name from gabriel_pb2.ResultWrapper.Status.
    Returns a valid enum value, or the first attribute on Status as fallback.
    """
    for name in candidates:
        val = getattr(gabriel_pb2.ResultWrapper.Status, name, None)
        if val is not None:
            return val
    # fallback: pick any existing attribute (avoid crashing)
    fallback_names = ["WRONG_INPUT_FORMAT", "SUCCESS"]
    for name in fallback_names:
        val = getattr(gabriel_pb2.ResultWrapper.Status, name, None)
        if val is not None:
            return val
    # As last resort, try to get first attribute via dir lookup
    for attr in dir(gabriel_pb2.ResultWrapper.Status):
        if attr.isupper():
            return getattr(gabriel_pb2.ResultWrapper.Status, attr)
    # if everything fails, raise (very unlikely)
    raise RuntimeError("Could not determine a fallback Status enum value.")


class EchoEngine(cognitive_engine.Engine):
    def __init__(self, device=None, use_half=False, tracker_frame_rate=30, track_config=None):
        # Device selection
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            if ("cuda" in device) and (not torch.cuda.is_available()):
                logging.warning("Requested CUDA device but torch.cuda.is_available() is False. Falling back to CPU.")
                self.device = "cpu"
            else:
                self.device = device

        logging.info(f"Using device: {self.device}")
        if "cuda" in self.device:
            torch.backends.cudnn.benchmark = True

        # Load YOLO model
        model_path = "yolo11n.pt" 
        logging.info(f"Loading YOLO model from {model_path} ...")
        self.model = YOLO(model_path)
        self.use_half = use_half and ("cuda" in self.device)

        try:
            # move model to device and set precision
            self.model.to(self.device)
            self.model.eval()
            if self.use_half:
                logging.info("Switching model to half precision (fp16).")
                try:
                    self.model.model.half()
                except Exception:
                    try:
                        self.model.half()
                    except Exception:
                        logging.warning("Couldn't switch to half; continuing with full precision.")
                        self.use_half = False
        except Exception:
            logging.exception("Failed to move model to device; continuing on CPU.")
            self.device = "cpu"
            self.use_half = False

        # warmup
        try:
            dummy = np.zeros((640, 480, 3), dtype=np.uint8)
            with torch.no_grad():
                try:
                    self.model(dummy, device=self.device)
                except Exception:
                    self.model(dummy)
        except Exception:
            pass

        # TurboJPEG for (optional) decode acceleration
        try:
            self.jpeg = TurboJPEG()
        except Exception:
            self.jpeg = None

        # Tracker and concurrency
        self.frame_id = 0
        self.lock = threading.Lock()

        if track_config is None:
            track_config = {
                "track_thresh": 0.4,
                "match_thresh": 0.8,
            }
        self.tracker = None
        if BYTETracker is not None:
            try:
                try:
                    self.tracker = BYTETracker(frame_rate=tracker_frame_rate, **track_config)
                except TypeError:
                    class DummyArgs: pass
                    args = DummyArgs()
                    for k, v in track_config.items():
                        setattr(args, k, v)
                    try:
                        self.tracker = BYTETracker(args, frame_rate=tracker_frame_rate)
                    except Exception:
                        self.tracker = None
            except Exception as e:
                logging.warning(f"bytetracker import succeeded but construction failed: {e}")
                self.tracker = None

        if self.tracker is None:
            logging.warning("ByteTrack tracker not initialized. Tracking disabled.")
        else:
            logging.info("ByteTrack tracker initialized.")

        logging.info("EchoEngine initialized.")

    def process_frame_bytes(self, img_bytes):
        """
        Process a single frame: decode, detect, track, and format results.
        """
        # decode
        np_data = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image bytes.")
        h, w = img.shape[:2]

        t0 = time.time()

        # inference
        try:
            with torch.no_grad():
                try:
                    results = self.model(img, device=self.device)
                except TypeError:
                    results = self.model.predict(img, device=self.device, verbose=False)
        except Exception:
            logging.exception("Inference failed")
            raise

        # 初始默认值
        dets_np = np.zeros((0,6), dtype=np.float32)
        classes = np.array([], dtype=int)
        num_dets = 0

        # 解析 detections 为 Nx6 array: [x1,y1,x2,y2,score,class]
        try:
            boxes = results[0].boxes
            try:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                cls = boxes.cls.cpu().numpy() if hasattr(boxes, "cls") else np.zeros((xyxy.shape[0],))
                if xyxy.size > 0:
                    dets_np = np.hstack((xyxy, confs.reshape(-1,1), cls.reshape(-1,1))).astype(np.float32)
                    classes = cls.astype(int)
                    num_dets = dets_np.shape[0]
                else:
                    dets_np = np.zeros((0,6), dtype=np.float32)
                    classes = np.array([], dtype=int)
                    num_dets = 0
            except Exception:
                det_list = []
                cls_list = []
                for box in boxes:
                    try:
                        xy = box.xyxy[0].tolist()
                        cf = float(box.conf[0])
                        cl = int(box.cls[0]) if hasattr(box, "cls") else 0
                        det_list.append([xy[0], xy[1], xy[2], xy[3], cf, cl])
                        cls_list.append(cl)
                    except Exception:
                        continue
                if det_list:
                    dets_np = np.array(det_list, dtype=np.float32)
                    classes = np.array(cls_list, dtype=int)
                    num_dets = dets_np.shape[0]
                else:
                    dets_np = np.zeros((0,6), dtype=np.float32)
                    classes = np.array([], dtype=int)
                    num_dets = 0
        except Exception:
            logging.exception("Parsing results failed")
            dets_np = np.zeros((0,6), dtype=np.float32)
            classes = np.array([], dtype=int)
            num_dets = 0

        # ---- call tracker.update ----
        online_targets = []
        resultTextString = ""
        num_tracks = 0
        if self.tracker is not None:
            self.frame_id += 1

            if dets_np.shape[0] > 0 and dets_np.shape[1] >= 6:
                dets_np[:, 5] = dets_np[:, 5].astype(np.int32)

            try:
                with self.lock:
                    # Try multiple variants, stop when one succeeds
                    tried = []
                    success = False

                    # Variant A: tensor (float)
                    try:
                        dets_variant = torch.from_numpy(dets_np).float() if dets_np.size else torch.zeros((0,6), dtype=torch.float32)
                        online_targets = self.tracker.update(dets_variant, [h, w])
                        # NOTE: Removed debug print statement from here
                        tried.append(('tensor-float', True))
                        success = True
                    except Exception as e_a:
                        tried.append(('tensor-float', False, str(e_a)))

                    if not success:
                        # Variant B: tensor with frame_id
                        try:
                            if dets_np.size:
                                dets_variant = torch.from_numpy(dets_np).float()
                            else:
                                dets_variant = torch.zeros((0,6), dtype=torch.float32)
                            online_targets = self.tracker.update(dets_variant, self.frame_id)
                            tried.append(('tensor-frameid', True))
                            success = True
                        except Exception as e_b:
                            tried.append(('tensor-frameid', False, str(e_b)))

                    if not success:
                        # Variant C: numpy with full 6 columns
                        try:
                            dets_np_c = dets_np.copy()
                            online_targets = self.tracker.update(dets_np_c, [h, w])
                            tried.append(('numpy-6col', True))
                            success = True
                        except Exception as e_c:
                            tried.append(('numpy-6col', False, str(e_c)))

                    if not success:
                        # Variant D: numpy 5-col (strip class)
                        try:
                            if dets_np.shape[1] >= 6:
                                dets5 = dets_np[:, :5].copy()
                            else:
                                dets5 = dets_np.copy()
                            online_targets = self.tracker.update(dets5, [h, w])
                            tried.append(('numpy-5col', True))
                            success = True
                        except Exception as e_d:
                            tried.append(('numpy-5col', False, str(e_d)))
                    
                    if not success and tried: # Log only if all failed
                         logging.warning(f"All tracker update variants failed: {tried}")

            except Exception:
                logging.exception("Tracker update error (outer)")
                online_targets = []


            # ---- parse online_targets to produce output ----
            resultTextString = ''
            num_tracks = 0
            try:
                is_empty = False
                if online_targets is None:
                    is_empty = True
                else:
                    try:
                        is_empty = (len(online_targets) == 0)
                    except Exception:
                        is_empty = (getattr(online_targets, "size", 0) == 0)

                if is_empty:
                    # no tracked objects: send back detections (track_id = -1)
                    for i in range(num_dets):
                        x1,y1,x2,y2,conf,clsid = dets_np[i].tolist()
                        cx = (x1 + x2) / 2.0 / float(w)
                        cy = (y1 + y2) / 2.0 / float(h)
                        bw = (x2 - x1) / float(w)
                        bh = (y2 - y1) / float(h)
                        resultTextString += f'{cx:.6f},{cy:.6f},{bw:.6f},{bh:.6f},{int(clsid)},{conf:.4f},{-1};'
                    num_tracks = 0
                else:
                    # tracked objects: send back tracks (7 fields)
                    parsed_tracks = []
                    for t in online_targets:
                        x1,y1,x2,y2,score,track_id,cls = 0.0, 0.0, 0.0, 0.0, 0.0, -1, -1
                        if isinstance(t, (list, tuple, np.ndarray)):
                            arr = np.array(t).astype(float).flatten()
                            if arr.size == 0: continue
                            
                            if arr.size >= 7:
                                # [x1, y1, x2, y2, track_id, class, score]
                                x1, y1, x2, y2, track_id, cls, score = arr[0], arr[1], arr[2], arr[3], int(arr[4]), int(arr[5]), float(arr[6])
                            elif arr.size == 6:
                                x1,y1,x2,y2 = arr[0], arr[1], arr[2], arr[3]
                                last, second_last = arr[-1], arr[-2]
                                if float(last).is_integer() and 0 <= int(last) <= 200 and (not float(second_last).is_integer() or int(second_last) > 200):
                                    score, cls, track_id = float(second_last), int(last), -1
                                else:
                                    track_id, cls, score = int(second_last), int(last), 0.5 
                            elif arr.size == 5:
                                x1,y1,x2,y2,track_id, score, cls = arr[0], arr[1], arr[2], arr[3], int(arr[4]), 0.5, -1
                            else:
                                continue
                        else:
                            try:
                                track_id = int(getattr(t, "track_id", getattr(t, "id", -1)))
                            except Exception: track_id = -1
                            score = float(getattr(t, "score", 0.0))
                            try:
                                cls = int(getattr(t, "class_id", getattr(t, "cls", -1)))
                            except Exception: cls = -1
                            
                            bbox = None
                            if hasattr(t, "tlbr"): bbox = t.tlbr
                            elif hasattr(t, "tlwh"):
                                x_tl,y_tl,w_box,h_box = t.tlwh
                                bbox = [x_tl, y_tl, x_tl+w_box, y_tl+h_box]
                            if bbox is None: continue
                            x1,y1,x2,y2 = [float(v) for v in bbox[:4]]

                        parsed_tracks.append((float(x1), float(y1), float(x2), float(y2), float(score), int(track_id), int(cls)))

                    # build output string: cx,cy,bw,bh,class,conf,track_id
                    for (x1,y1,x2,y2,score,track_id,cls) in parsed_tracks:
                        cx = (x1 + x2) / 2.0 / float(w)
                        cy = (y1 + y2) / 2.0 / float(h)
                        bw = (x2 - x1) / float(w)
                        bh = (y2 - y1) / float(h)
                        resultTextString += f'{cx:.6f},{cy:.6f},{bw:.6f},{bh:.6f},{cls},{score:.4f},{track_id};'
                    num_tracks = len(parsed_tracks)

            except Exception:
                logging.exception("Parsing online_targets failed")
                resultTextString = ''
                num_tracks = 0

        t1 = time.time()
        elapsed = t1 - t0
        return resultTextString, elapsed, num_dets, num_tracks

    # Gabriel's expected handler
    def handle(self, input_frame):
        status = _choose_status("SUCCESS")
        result_wrapper = cognitive_engine.create_result_wrapper(status)

        if len(input_frame.payloads) == 0:
            logging.debug("Empty payloads.")
            return result_wrapper
        if input_frame.payload_type != gabriel_pb2.PayloadType.IMAGE:
            status = _choose_status("WRONG_INPUT_FORMAT")
            return cognitive_engine.create_result_wrapper(status)

        img_bytes = input_frame.payloads[0]
        try:
            resultTextString, elapsed, num_dets, num_tracks = self.process_frame_bytes(img_bytes)
        except Exception:
            logging.exception("process_frame_bytes failed")
            status = _choose_status("GENERIC_ERROR", "ERROR", "INTERNAL_ERROR")
            return cognitive_engine.create_result_wrapper(status)

        # build Gabriel result
        result = gabriel_pb2.ResultWrapper.Result()
        result.payload_type = gabriel_pb2.PayloadType.TEXT
        result.payload = resultTextString.encode('utf-8')
        result_wrapper.results.append(result)

        return result_wrapper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", "-p", type=int, default=DEFAULT_PORT)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--device", type=str, default=None,
                        help='device to use, e.g. "cuda", "cuda:0", or "cpu". If omitted, auto-detect.')
    parser.add_argument("--half", action="store_true", help="use fp16 half precision on CUDA device")
    parser.add_argument("--local-test", type=str, default=None,
                        help="If provided, path to a local video file to run local test mode (bypasses Gabriel).")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="path to yolo model")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    # Local test mode
    if args.local_test:
        engine = EchoEngine(device=args.device, use_half=args.half)
        cap = cv2.VideoCapture(args.local_test)
        if not cap.isOpened():
            print("Failed to open video:", args.local_test)
            return
        frame_count = 0
        t_total = 0.0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                ok, buf = cv2.imencode('.jpg', frame)
                if not ok:
                    continue
                data = buf.tobytes()
                try:
                    res_str, elapsed, ndet, ntrk = engine.process_frame_bytes(data)
                except Exception as e:
                    print(f"process error on frame {frame_count}: {e}")
                    continue
                frame_count += 1
                t_total += elapsed
                print(f"frame {frame_count}: time {elapsed*1000:.1f} ms, dets {ndet}, tracks {ntrk}")
                print(res_str[:200]) # print first 200 chars of result
            if frame_count > 0:
                print(f"Avg time per frame: {t_total/frame_count*1000:.1f} ms, FPS ~ {frame_count/(t_total+1e-9):.2f}")
        finally:
            cap.release()
        return

    # Normal Gabriel server mode
    def engine_factory():
        return EchoEngine(device=args.device, use_half=args.half)

    local_engine.run(engine_factory, SOURCE_NAME, INPUT_QUEUE_MAXSIZE, args.port, NUM_TOKENS)


if __name__ == "__main__":
    main()

