#!/usr/bin/env python3
"""
EchoEngine + ByteTrack integration for single mobile (Gabriel pipeline compatible).

MODIFIED FOR BENCHMARKING:
- Receives --max-frames, --fps, and --output arguments.
- Stops processing after max-frames is reached.
- Writes tracking results to --output in MOTChallenge format.
- [NEW] Optionally saves a visualization video with --visualize-video.
"""
import argparse
import logging
import io
import cv2
import numpy as np
import torch
import time
import threading
import sys

from ultralytics import YOLO
from turbojpeg import TurboJPEG

# Gabriel related imports
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

if BYTETracker is None:
    logging.critical("Failed to import any BYTETracker implementation. Please install 'bytetracker' or 'ultralytics'.")
    sys.exit(1)


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
    # === 【MOD】 签名已更改，以接受来自 main 的新参数 ===
    def __init__(self, device=None, use_half=False, tracker_frame_rate=30,
                 model_path="yolo11n.pt", max_frames=900, output_filename="mot_results.txt",
                 track_thresh=0.45, match_thresh=0.8, track_buffer=30,
                 visualize_video_path=None): # <-- 【NEW】 新的可视化参数

        # === Benchmarking parameters ===
        self.max_frames = max_frames
        self.output_filename = output_filename
        try:
            self.output_file = open(self.output_filename, 'w')
        except IOError as e:
            logging.critical(f"Failed to open output file: {self.output_filename}. Error: {e}")
            raise
        self.frame_counter = 0
        self.file_lock = threading.Lock() # Protects self.frame_counter, self.output_file, and self.video_writer

        # === 【NEW】 可视化视频写入器 ===
        self.visualize_video_path = visualize_video_path
        self.video_writer = None
        self.vis_colors = {} # 用于为 track_id 缓存颜色
        if self.visualize_video_path:
            logging.info(f"Visualization enabled. Video will be saved to: {self.visualize_video_path}")
        # 【NEW】 存储帧率以供 VideoWriter 使用
        self.tracker_frame_rate = tracker_frame_rate 

        # === Device selection ===
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

        # === Load YOLO model ===
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

        # === TurboJPEG for (optional) decode acceleration ===
        try:
            self.jpeg = TurboJPEG()
        except Exception:
            self.jpeg = None

        # === Tracker and concurrency ===
        self.tracker_frame_id = 0 # Bytetrack's internal counter
        self.tracker_lock = threading.Lock() # Protects tracker.update and tracker_frame_id

        # === 【MOD】 从构造函数参数动态构建 track_config ===
        track_config = {
            "track_thresh": track_thresh,
            "match_thresh": match_thresh,
            "track_buffer": track_buffer
        }
        logging.info(f"Using ByteTrack config: {track_config}")
        # (基于你找到的 __init__, 'det_thresh' 将被自动设为 track_thresh + 0.1)
        if track_thresh < 0.1:
             logging.warning(f"track_thresh ({track_thresh}) is very low. "
                             f"This will set high_thresh (det_thresh) to {track_thresh + 0.1}, "
                             f"which may cause high False Positives.")

        self.tracker = None
        if BYTETracker is not None:
            logging.info(f"Initializing BYTETracker with frame_rate={self.tracker_frame_rate}")
            try:
                try:
                    self.tracker = BYTETracker(frame_rate=self.tracker_frame_rate, **track_config)
                except TypeError:
                    class DummyArgs: pass
                    args = DummyArgs()
                    for k, v in track_config.items():
                        setattr(args, k, v)
                    try:
                        self.tracker = BYTETracker(args, frame_rate=self.tracker_frame_rate)
                    except Exception:
                        self.tracker = None
            except Exception as e:
                logging.warning(f"bytetracker import succeeded but construction failed: {e}")
                self.tracker = None

        if self.tracker is None:
            logging.critical("ByteTrack tracker not initialized. Tracking disabled.")
        else:
            logging.info("ByteTrack tracker initialized.")

        logging.info("EchoEngine initialized.")

    # === 【NEW】 绘制跟踪框的辅助函数 ===
    def _draw_tracks(self, frame, tracks):
        """
        Draws bounding boxes and track IDs on the frame.
        """
        for (x1, y1, x2, y2, score, track_id, cls) in tracks:
            track_id = int(track_id)
            
            # 为 track_id 生成一个唯一的、一致的颜色
            if track_id not in self.vis_colors:
                self.vis_colors[track_id] = (
                    (track_id * 37 + 100) % 256, 
                    (track_id * 97 + 50) % 256, 
                    (track_id * 53 + 150) % 256
                )
            color = self.vis_colors[track_id]

            pt1 = (int(x1), int(y1))
            pt2 = (int(x2), int(y2))
            cv2.rectangle(frame, pt1, pt2, color, 2)
            
            label = f"ID:{track_id}"
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (pt1[0], pt1[1] - text_h - 5), (pt1[0] + text_w, pt1[1]), color, -1)
            cv2.putText(frame, label, (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame


    def process_frame_bytes(self, img_bytes):
        """
        Process a single frame: decode, detect, track, and format results.
        """
        # decode
        np_data = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(np_data, cv2.IMREAD_COLOR) # <-- 【NEW】 我们需要这个 'img'
        if img is None:
            raise ValueError("Failed to decode image bytes.")
        h, w = img.shape[:2]

        t0 = time.time()

        # inference
        try:
            with torch.no_grad():
                try:
                    # [MOD] We pass conf=0.01 to YOLO to let ByteTrack do the filtering
                    results = self.model(img, device=self.device, conf=0.01, verbose=False)
                except TypeError:
                    results = self.model.predict(img, device=self.device, conf=0.01, verbose=False)
        except Exception:
            logging.exception("Inference failed")
            raise

        # 默认值
        dets_np = np.zeros((0,6), dtype=np.float32)
        classes = np.array([], dtype=int)
        num_dets = 0

        # ... (YOLO 结果解析逻辑保持不变) ...
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

        # ---- [MOD] 只保留“人” (class_id = 0) ----
        if dets_np.shape[0] > 0:
            person_class_id = 0
            keep_mask = (dets_np[:, 5] == person_class_id)
            dets_np = dets_np[keep_mask]

            if dets_np.shape[0] > 0:
                classes = dets_np[:, 5].astype(int)
            else:
                classes = np.array([], dtype=int)
            num_dets = dets_np.shape[0]

        # ---- call tracker.update ----
        # ... (tracker.update 逻辑保持不变) ...
        online_targets = []
        resultTextString = ""
        num_tracks = 0
        parsed_tracks = [] 

        if self.tracker is not None:
            current_tracker_frame_id = 0
            if dets_np.shape[0] > 0 and dets_np.shape[1] >= 6:
                dets_np[:, 5] = dets_np[:, 5].astype(np.int32)
            try:
                with self.tracker_lock:
                    self.tracker_frame_id += 1
                    current_tracker_frame_id = self.tracker_frame_id
                    tried = []
                    success = False
                    # Variant A: tensor (float)
                    try:
                        dets_variant = torch.from_numpy(dets_np).float() if dets_np.size else torch.zeros((0,6), dtype=torch.float32)
                        online_targets = self.tracker.update(dets_variant, [h, w])
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
                            online_targets = self.tracker.update(dets_variant, current_tracker_frame_id)
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
            # ... (解析 online_targets 的逻辑保持不变, 确保 parsed_tracks 被填充) ...
            resultTextString = ''
            num_tracks = 0
            parsed_tracks = []
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
                    for i in range(num_dets):
                        x1,y1,x2,y2,conf,clsid = dets_np[i].tolist()
                        cx = (x1 + x2) / 2.0 / float(w)
                        cy = (y1 + y2) / 2.0 / float(h)
                        bw = (x2 - x1) / float(w)
                        bh = (y2 - y1) / float(h)
                        resultTextString += f'{cx:.6f},{cy:.6f},{bw:.6f},{bh:.6f},{int(clsid)},{conf:.4f},{-1};'
                    num_tracks = 0
                else:
                    for t in online_targets:
                        x1,y1,x2,y2,score,track_id,cls = 0.0, 0.0, 0.0, 0.0, 0.0, -1, -1
                        if isinstance(t, (list, tuple, np.ndarray)):
                            arr = np.array(t).astype(float).flatten()
                            if arr.size == 0: continue
                            if arr.size >= 7:
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

        # === 【MOD】 返回原始的 'img' 以供可视化 ===
        return resultTextString, elapsed, num_dets, num_tracks, parsed_tracks, w, h, img

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
            # === 【MOD】 获取包括 'img' 在内的所有返回值 ===
            resultTextString, elapsed, num_dets, num_tracks, parsed_tracks, w, h, img = self.process_frame_bytes(img_bytes)

        except Exception:
            logging.exception("process_frame_bytes failed")
            status = _choose_status("GENERIC_ERROR", "ERROR", "INTERNAL_ERROR")
            return cognitive_engine.create_result_wrapper(status)

        # ==================================================================
        # 【MOD】 BENCHMARK FILE WRITING LOGIC
        # ==================================================================
        with self.file_lock:
            # Check if we've already finished the benchmark run
            if self.frame_counter >= self.max_frames:
                # Return an empty result, do not process or write anymore
                return result_wrapper

            # This is a valid frame to process
            self.frame_counter += 1
            current_frame_id = self.frame_counter

            # --- 1. 写入 MOT .txt 结果 (无变化) ---
            if parsed_tracks:
                for (x1, y1, x2, y2, score, track_id, cls) in parsed_tracks:
                    bb_left = x1
                    bb_top = y1
                    bb_width = x2 - x1
                    bb_height = y2 - y1
                    line = f"{current_frame_id},{track_id},{bb_left:.2f},{bb_top:.2f},{bb_width:.2f},{bb_height:.2f},{score:.2f},-1,-1,-1\n"
                    self.output_file.write(line)
            
            # --- 2. 【NEW】 写入可视化视频帧 ---
            if self.visualize_video_path:
                # 懒加载 VideoWriter (在第一帧)
                if self.video_writer is None:
                    try:
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # for .mp4
                        fps_video = float(self.tracker_frame_rate)
                        self.video_writer = cv2.VideoWriter(self.visualize_video_path, fourcc, fps_video, (w, h))
                        logging.info(f"Initialized video writer for visualization at {self.visualize_video_path} with size {(w, h)} @ {fps_video} FPS")
                    except Exception as e:
                        logging.error(f"Failed to initialize VideoWriter: {e}")
                        self.visualize_video_path = None # 禁用它以避免重复错误

                # 如果初始化成功，绘制并写入帧
                if self.video_writer:
                    # 我们在 'img' (原始帧的副本) 上绘制
                    vis_frame = self._draw_tracks(img, parsed_tracks)
                    self.video_writer.write(vis_frame)

            # --- 3. 检查是否完成 (合并了 .txt 和 .mp4 的关闭逻辑) ---
            if current_frame_id == self.max_frames:
                logging.info(f"Processed {self.max_frames} frames. Closing output file.")
                logging.info("Benchmark run complete. Server will idle but stop processing.")
                self.output_file.close()
                
                # 【NEW】 关闭视频文件
                if self.video_writer:
                    self.video_writer.release()
                    self.video_writer = None
                    logging.info(f"Visualization video saved to {self.visualize_video_path}")

            # As a safety, flush every 100 frames
            elif (current_frame_id % 100) == 0:
                logging.info(f"Processed {current_frame_id} / {self.max_frames} frames...")
                self.output_file.flush()
        # ==================================================================
        # (End of benchmark logic)
        # ==================================================================

        # build Gabriel result (still send back to client)
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

    # === [MOD] New Benchmark Arguments ===
    parser.add_argument("--max-frames", type=int, default=900,
                        help="Total frames to process for the benchmark. Server will stop writing to file after this many frames.")
    parser.add_argument("--fps", type=int, default=30,
                        help="Frame rate to configure Bytetrack with (e.g., 30).")
    parser.add_argument("--output", type=str, default="mot_results.txt",
                        help="Output file for MOT results (e.g., 'results/MOT17-02.txt')")
    
    # === 【MOD】 New Tracker Arguments ===
    parser.add_argument("--track-thresh", type=float, default=0.45,
                        help="ByteTrack LOW threshold (default: 0.45). "
                             "HIGH threshold will be (track_thresh + 0.1).")
    parser.add_argument("--match-thresh", type=float, default=0.8,
                        help="ByteTrack IoU match threshold (default: 0.8).")
    parser.add_argument("--track-buffer", type=int, default=30,
                        help="ByteTrack frame buffer for lost tracks (default: 30).")

    # === 【NEW】 可视化参数 ===
    parser.add_argument("--visualize-video", type=str, default=None,
                        help="Optional. Path to save a visualization video (e.g., 'results/vis.mp4').")


    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    # Local test mode (unchanged, still useful for debugging)
    if args.local_test:
        engine = EchoEngine(device=args.device, use_half=args.half, model_path=args.model,
                            tracker_frame_rate=args.fps, max_frames=args.max_frames,
                            output_filename=args.output,
                            track_thresh=args.track_thresh, 
                            match_thresh=args.match_thresh, 
                            track_buffer=args.track_buffer,
                            visualize_video_path=args.visualize_video # <-- 【NEW】
                            )
        cap = cv2.VideoCapture(args.local_test)
        if not cap.isOpened():
            print("Failed to open video:", args.local_test)
            return
        frame_count = 0
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
                    dummy_input = gabriel_pb2.InputFrame()
                    dummy_input.payload_type = gabriel_pb2.PayloadType.IMAGE
                    dummy_input.payloads.append(data)

                    wrapper = engine.handle(dummy_input)

                    frame_count += 1
                    if (frame_count % 100) == 0:
                        logging.info(f"Local test processed {frame_count} frames...")

                    if engine.frame_counter >= engine.max_frames:
                        logging.info(f"Local test reached max_frames ({engine.max_frames}). Stopping.")
                        break

                except Exception as e:
                    print(f"process error on frame {frame_count}: {e}")
                    continue
            
            # 【MOD】 在 local test 结束时关闭文件
            logging.info(f"Local test finished. Results saved to {engine.output_filename}")
            if engine.output_file and not engine.output_file.closed:
                engine.output_file.close()
            # 【NEW】 在 local test 结束时关闭视频
            if engine.video_writer and engine.video_writer.isOpened():
                engine.video_writer.release()
                logging.info(f"Local test visualization video saved to {engine.visualize_video_path}")

        finally:
            cap.release()
        return

    # Normal Gabriel server mode
    def engine_factory():
        # [MOD] Pass all new arguments to the engine
        return EchoEngine(device=args.device,
                          use_half=args.half,
                          model_path=args.model,
                          tracker_frame_rate=args.fps,
                          max_frames=args.max_frames,
                          output_filename=args.output,
                          track_thresh=args.track_thresh, 
                          match_thresh=args.match_thresh, 
                          track_buffer=args.track_buffer,
                          visualize_video_path=args.visualize_video # <-- 【NEW】
                          )

    logging.info(f"Starting Gabriel benchmark server on port {args.port}")
    logging.info(f"Press Ctrl+C to stop the server.")
    try:
        local_engine.run(engine_factory, SOURCE_NAME, INPUT_QUEUE_MAXSIZE, args.port, NUM_TOKENS)
    except KeyboardInterrupt:
        logging.info("Server shut down by user.")
        # 注意：在这里很难安全地关闭 engine 的文件
        # Server 将立即退出。最好的关闭逻辑是在 handle() 中达到 max_frames 时。


if __name__ == "__main__":
    main()
