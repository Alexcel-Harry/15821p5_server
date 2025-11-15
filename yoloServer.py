#!/usr/bin/env python3
"""
Echo Engine based on Gabriel

Modified to support GPU (CUDA) if available.
"""

import argparse
import logging
import io
import cv2
import numpy as np
from ultralytics import YOLO
import torch
import time
from turbojpeg import TurboJPEG

from gabriel_server import cognitive_engine
from gabriel_server import local_engine
from gabriel_protocol import gabriel_pb2

SOURCE_NAME = "echo"
INPUT_QUEUE_MAXSIZE = 60
DEFAULT_PORT = 9099
NUM_TOKENS = 3


class EchoEngine(cognitive_engine.Engine):
    def __init__(self, device=None, use_half=False):
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

        logging.info("Loading YOLO model...")
        self.model = YOLO("yolo11s.pt")

        # set the model to float 16 or float 32 
        try:
            self.model.to(self.device)
            self.model.eval() 
            self.use_half = use_half and ("cuda" in self.device)
            if self.use_half:
                logging.info("Switching model to half precision (fp16).")
                try:
                    try:
                        self.model.model.half()
                    except Exception:
                        self.model.half()
                except Exception:
                    logging.warning("Couldn't switch to half precision; continuing with full precision.")
                    self.use_half = False

            # warm up
            try:
                dummy = np.zeros((640, 480, 3), dtype=np.uint8)
                with torch.no_grad():
                    self.model(dummy, device=self.device)
            except Exception:
                # ignore warmup errors
                pass
            
            try: 
                self.jpeg = TurbeJPEG()
            except Exception:
                self.jpeg = None

        except Exception as e:
            logging.exception("Failed to move model to device; continuing on CPU.")
            self.device = "cpu"
            self.use_half = False

        logging.info("YOLO model loaded.")

    def handle_useless(self, input_frame):
            status = gabriel_pb2.ResultWrapper.Status.SUCCESS
            result_wrapper = cognitive_engine.create_result_wrapper(status)
    
            if len(input_frame.payloads) == 0:
                logging.info("Empty payloads.")
                return result_wrapper
    
            # Start timer
            t0 = time.time()
    
            if input_frame.payload_type != gabriel_pb2.PayloadType.IMAGE:
                status = gabriel_pb2.ResultWrapper.Status.WRONG_INPUT_FORMAT
                return cognitive_engine.create_result_wrapper(status)
    
            img_data = input_frame.payloads[0]
            np_data = np.frombuffer(img_data, dtype=np.uint8)
            img = cv2.imdecode(np_data, cv2.IMREAD_COLOR)  # BGR np.uint8 H W C
    
            try:
                # The ONLY call you need. It handles pre-processing, inference, 
                # and post-processing automatically on the correct device.
                results = self.model(img)
    
            except Exception as e:
                logging.exception("Inference failed")
                status = gabriel_pb2.ResultWrapper.Status.GENERIC_ERROR
                return cognitive_engine.create_result_wrapper(status)
    
            resultTextString = ''
            if results:
                for box in results[0].boxes:
                    classID = int(box.cls[0])
                    cords = box.xywhn[0].tolist()
                    conf = float(box.conf[0])
                    resultTextString += f'{cords[0]},{cords[1]},{cords[2]},{cords[3]},{classID},{conf};'
    
            result = gabriel_pb2.ResultWrapper.Result()
            result.payload_type = gabriel_pb2.PayloadType.TEXT
            result.payload = resultTextString.encode('utf-8')
            result_wrapper.results.append(result)
            
            # End timer
            t1 = time.time()
            print(f"handle time {t1 - t0}")
    
            return result_wrapper

   
#     def handle(self, input_frame):
#         status = gabriel_pb2.ResultWrapper.Status.SUCCESS
#         result_wrapper = cognitive_engine.create_result_wrapper(status)
#     
#         if len(input_frame.payloads) == 0:
#             logging.info("Empty payloads.")
#             return result_wrapper
#         
#         t0 = time.time()
# 
#         if input_frame.payload_type != gabriel_pb2.PayloadType.IMAGE:
#             status = gabriel_pb2.ResultWrapper.Status.WRONG_INPUT_FORMAT
#             return cognitive_engine.create_result_wrapper(status)
#     
#         img_data = input_frame.payloads[0]
#         if self.jpeg is not None:
#             img = self.jpeg.decode(img_data)
#         else: 
#             np_data = np.frombuffer(img_data, dtype=np.uint8)
#             img = cv2.imdecode(np_data, cv2.IMREAD_COLOR)  # BGR np.uint8 H W C
#     
#         try:
#             with torch.no_grad():
#                 results = self.model(img, conf=0.1)  # wrapper，但不带 device 参数
#         except Exception:
#             results = None
#     
#         resultTextString = ''
#         if results is not None:
#             for box in results[0].boxes:
#                 classID = int(box.cls[0])
#                 cords = box.xywhn[0].tolist()
#                 conf = float(box.conf[0])
#                 resultTextString += f'{cords[0]},{cords[1]},{cords[2]},{cords[3]},{classID},{conf};'
#         else:
#             pass
#     
#         result = gabriel_pb2.ResultWrapper.Result()
#         result.payload_type = gabriel_pb2.PayloadType.TEXT
#         result.payload = resultTextString.encode('utf-8')
#         result_wrapper.results.append(result)
#    
# 
#         t1 = time.time()
#         print(f"handle time {t1 - t0}")
#         return result_wrapper
# 
    def handle(self, input_frame):
        status = gabriel_pb2.ResultWrapper.Status.SUCCESS
        result_wrapper = cognitive_engine.create_result_wrapper(status)

        if len(input_frame.payloads) == 0:
            logging.info("Empty payloads.")
            return result_wrapper

        if input_frame.payload_type != gabriel_pb2.PayloadType.IMAGE:
            status = gabriel_pb2.ResultWrapper.Status.WRONG_INPUT_FORMAT
            return cognitive_engine.create_result_wrapper(status)
        img_data = input_frame.payloads[0]
        np_data = np.frombuffer(img_data, dtype=np.uint8)
        img = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
        try:
            with torch.no_grad():
                results = self.model(img)
        except Exception as e:
            logging.exception("Inference failed")
            status = gabriel_pb2.ResultWrapper.Status.GENERIC_ERROR
            return cognitive_engine.create_result_wrapper(status)

        resultTextString = ''
        for box in results[0].boxes:
            classID = int(box.cls[0])
            cords = box.xywhn[0].tolist()
            conf = float(box.conf[0])
            resultTextString += f'{cords[0]},{cords[1]},{cords[2]},{cords[3]},{classID},{conf};'

        result = gabriel_pb2.ResultWrapper.Result()
        result.payload_type = gabriel_pb2.PayloadType.TEXT
        result.payload = resultTextString.encode('utf-8')
        result_wrapper.results.append(result)

        # from datetime import datetime
        # ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # ack = gabriel_pb2.ResultWrapper.Result()
        # ack.payload_type = gabriel_pb2.PayloadType.TEXT
        # ack.payload = f"ACK at {ts}".encode("utf-8")
        # result_wrapper.results.append(ack)

        return result_wrapper

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", "-p", type=int, default=DEFAULT_PORT)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--device", type=str, default=None,
                        help='device to use, e.g. "cuda", "cuda:0", or "cpu". If omitted, auto-detect.')
    parser.add_argument("--half", action="store_true", help="use fp16 half precision on CUDA device")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    def engine_factory():
        return EchoEngine(device=args.device, use_half=args.half)

    local_engine.run(engine_factory, SOURCE_NAME, INPUT_QUEUE_MAXSIZE, args.port, NUM_TOKENS)


if __name__ == "__main__":
    main()

