#!/usr/bin/env python3
"""
Echo Engine based on Gabriel

Functions:
    - Receives input frames from client
    - Echoes payloads back with ACK
"""

import argparse
import logging
import io
import cv2
import numpy as np
from ultralytics import YOLO

from gabriel_server import cognitive_engine
from gabriel_server import local_engine
from gabriel_protocol import gabriel_pb2

SOURCE_NAME = "echo"
INPUT_QUEUE_MAXSIZE = 60
DEFAULT_PORT = 9099
NUM_TOKENS = 1


class EchoEngine(cognitive_engine.Engine):
    def __init__(self):
        """
        Perform startup operations such as warming up the model.
        Nothing to do here for echo server.
        """
        logging.info("Loading YOLO model...")
        self.model = YOLO("yolo11n.pt")
        logging.info("YOLO model loaded.")

    def handle(self, input_frame):
        """
        Process the input frame and return processing results.
        """
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
        # with open("a.txt", "w") as f:
        #    f.write("BYTESTRING::::\n")
        #    f.write(img_data.decode())
        # exit()
        img = cv2.imdecode(np_data, cv2.IMREAD_COLOR)

        results = self.model(img)
        print("Start of Debug Msg")
        #print(results)
        #with open ('a.txt', "w") as f:
            #f.write(str(results))
        resultTextString = ''
        #with open ('b.txt', 'w') as f:
        for box in results[0].boxes:
            #print(str(box))
            classID = int(box.cls[0])
            cords = box.xywhn[0].tolist()
            conf = float(box.conf[0])
            resultTextString += f'{cords[0]},{cords[1]},{cords[2]},{cords[3]},{classID},{conf};'
                #f.write(str([classID, cords, conf]))
            #f.write(str(results))

        #annotated_img = results[0].plot()

        #_, buffer = cv2.imencode(".jpg", annotated_img)
        #annotated_bytes = buffer.tobytes()
        result = gabriel_pb2.ResultWrapper.Result()
        #result.payload_type = gabriel_pb2.PayloadType.IMAGE
        result.payload_type = gabriel_pb2.PayloadType.TEXT
        result.payload = resultTextString.encode('utf-8')
        #result.payload = annotated_bytes
        result_wrapper.results.append(result)

        from datetime import datetime
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        ack = gabriel_pb2.ResultWrapper.Result()
        ack.payload_type = gabriel_pb2.PayloadType.TEXT
        ack.payload = f"ACK at {ts}".encode("utf-8")
        result_wrapper.results.append(ack)
        print(resultTextString)
        #exit()
        return result_wrapper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", "-p", type=int, default=DEFAULT_PORT)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    def engine_factory():
        return EchoEngine()

    local_engine.run(engine_factory, SOURCE_NAME, INPUT_QUEUE_MAXSIZE, args.port, NUM_TOKENS)


if __name__ == "__main__":
    main()
