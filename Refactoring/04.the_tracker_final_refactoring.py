import cv2
import numpy as np
import time
import multiprocessing as mp
from multiprocessing import shared_memory, Lock, Event
from ultralytics import YOLO


# Mock GPIO 클래스
class MockGPIO:
    BOARD = "BOARD"
    BCM = "BCM"
    OUT = "OUT"
    IN = "IN"
    HIGH = "HIGH"
    LOW = "LOW"

    def __init__(self):
        self.pins = {}

    def setmode(self, mode):
        print(f"Setting mode to {mode}")

    def setup(self, channel, mode, initial=None):
        self.pins[channel] = {'mode': mode, 'state': initial}
        print(f"Setting up channel {channel} as {mode} with initial {initial}")

    def output(self, channel, state):
        if channel in self.pins:
            self.pins[channel]['state'] = state
            print(f"Setting channel {channel} to {state}")
        else:
            print(f"Channel {channel} not set up yet!")

    def input(self, channel):
        return self.pins.get(channel, {}).get('state', None)

    def cleanup(self):
        self.pins = {}
        print("Cleaning up all channels")

    def setwarnings(self, flag):
        print(f"Setting warnings {'on' if flag else 'off'}")

    def PWM(self, channel, frequency):
        print(f"Setting up PWM on channel {channel} with frequency {frequency}")
        return MockPWM(channel, frequency)


class MockPWM:
    def __init__(self, channel, frequency):
        self.channel = channel
        self.frequency = frequency
        self.duty_cycle = 0

    def start(self, duty_cycle):
        self.duty_cycle = duty_cycle
        print(f"Starting PWM on channel {self.channel} with duty cycle {duty_cycle}")

    def ChangeDutyCycle(self, duty_cycle):
        self.duty_cycle = duty_cycle
        print(f"Changing duty cycle on channel {self.channel} to {duty_cycle}")

    def stop(self):
        print(f"Stopping PWM on channel {self.channel}")


try:
    import RPi.GPIO as GPIO
except ImportError:
    GPIO = MockGPIO()


# Tracker 생성 및 관리 클래스
class TrackerManager:
    def __init__(self):
        self.trackers = []

    def create_tracker(self, frame, rect):
        tracker = cv2.legacy.TrackerMOSSE_create()  # MOSSE 트래커 생성
        tracker.init(frame, rect)  # 트래커 초기화
        self.trackers.append(tracker)  # 생성된 트래커를 리스트에 추가
        return tracker

    def update_trackers(self, frame):
        for tracker in self.trackers:
            success, bbox = tracker.update(frame)
            if success:
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)


# YOLO 모델 클래스 (객체 탐지)
class YOLOModel:
    def __init__(self, model_path):
        self.model = YOLO(model_path, task='detect')  # YOLO 모델 초기화 시 task 명시
        self.class_dict = {0: 'cane', 1: 'crutches', 2: 'walker', 3: 'wheelchair', 4: 'white_cane'}  # 클래스 ID와 이름 매핑

    def detect(self, img):
        results = self.model(img, imgsz=640, conf=0.4)[0]  # 이미지에서 객체 탐지 수행
        return results


# 비디오 재생 프로세스
def video_capture_process(video_path, shm_name, shm_bbox_name, shape, lock, bbox_lock, event, stop_event, tracker_manager):
    capture = cv2.VideoCapture(video_path)
    shm = shared_memory.SharedMemory(name=shm_name)
    bbox_shm = shared_memory.SharedMemory(name=shm_bbox_name)
    frame_array = np.ndarray(shape, dtype=np.uint8, buffer=shm.buf)
    bbox_array = np.ndarray((10, 5), dtype=np.float32, buffer=bbox_shm.buf)  # 최대 10개의 박스 좌표와 클래스 정보

    frame_count = 0

    while not stop_event.is_set():  # stop_event로 프로세스 종료를 관리
        ret, frame = capture.read()
        if not ret:
            stop_event.set()  # 비디오 끝나면 종료 신호 전송
            event.set()  # 모델 추론 프로세스도 종료되도록 신호 전송
            break

        frame_resized = cv2.resize(frame, (shape[1], shape[0]))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)  # BGR을 RGB로 변환

        # 공유 메모리로 프레임 전송
        lock.acquire()
        np.copyto(frame_array, frame_rgb)
        lock.release()

        # 매 60프레임마다 추론을 수행하도록 이벤트 설정
        if frame_count % 60 == 0:
            event.set()  # 모델 추론 프로세스에 신호 전송

        # 사각형 좌표 및 클래스 정보를 공유 메모리로부터 읽어와 비디오 프레임에 그리기
        bbox_lock.acquire()
        for i in range(10):  # 최대 10개의 사각형
            if bbox_array[i][2] > 0 and bbox_array[i][3] > 0:  # 유효한 사각형만 그림
                x, y, w, h = bbox_array[i][:4]
                class_id = int(bbox_array[i][4])
                class_label = YOLOModel('').class_dict[class_id]  # 클래스 라벨 가져오기
                cv2.rectangle(frame_resized, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
                cv2.putText(frame_resized, class_label, (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        bbox_lock.release()

        # 결과 보여주기
        cv2.imshow("Video Stream", frame_resized)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            event.set()  # 추론 프로세스도 종료되도록 신호 전송
            break

        frame_count += 1
        time.sleep(0.03)  # 비디오가 끊김 없이 재생되도록

    capture.release()
    cv2.destroyAllWindows()


# 모델 추론 프로세스
def model_inference_process(shm_name, shm_bbox_name, shape, lock, bbox_lock, event, stop_event, tracker_manager, model_path):
    yolo_model = YOLOModel(model_path)
    shm = shared_memory.SharedMemory(name=shm_name)
    bbox_shm = shared_memory.SharedMemory(name=shm_bbox_name)
    frame_array = np.ndarray(shape, dtype=np.uint8, buffer=shm.buf)
    bbox_array = np.ndarray((10, 5), dtype=np.float32, buffer=bbox_shm.buf)  # 최대 10개의 박스 좌표와 클래스 정보

    while not stop_event.is_set():
        event.wait()  # 비디오 프로세스로부터 신호 대기
        if stop_event.is_set():
            break  # 종료 신호가 오면 종료

        lock.acquire()
        frame = frame_array.copy()  # 공유 메모리에서 프레임 복사
        lock.release()

        # YOLO 모델을 사용해 객체 탐지
        results = yolo_model.detect(frame)

        # 탐지된 객체 좌표와 클래스 ID를 bbox_array로 공유
        bbox_lock.acquire()
        bbox_array.fill(0)  # 이전 값 초기화
        for i, result in enumerate(results[:10]):  # 최대 10개의 객체
            xywh = result.boxes.xywh
            class_id = int(result.boxes.cls[0])  # 클래스 ID 가져오기
            x, y, w, h = int(xywh[0][0]), int(xywh[0][1]), int(xywh[0][2]), int(xywh[0][3])
            bbox_array[i] = [x - w // 2, y - h // 2, w, h, class_id]  # 좌표와 클래스 ID 기록
        bbox_lock.release()

        event.clear()  # 이벤트 리셋


# 메인 프로세스
if __name__ == '__main__':
    video_path = 'Wheelchair_Elevator_inside_in.mp4'
    model_path = '03.ELSA_SEG_model.pt'
    pwm_pins = [12, 32]
    frame_shape = (480, 640, 3)

    # 공유 메모리 생성 (프레임과 바운딩 박스 정보)
    shm = shared_memory.SharedMemory(create=True, size=int(np.prod(frame_shape) * np.uint8().nbytes))
    bbox_shm = shared_memory.SharedMemory(create=True, size=10 * 5 * np.float32().nbytes)  # 최대 10개의 박스 정보 + 클래스 정보
    lock = Lock()
    bbox_lock = Lock()
    event = Event()
    stop_event = Event()

    # TrackerManager 생성
    tracker_manager = TrackerManager()

    # 비디오 캡처 및 모델 추론 프로세스 생성
    video_process = mp.Process(target=video_capture_process,
                               args=(video_path, shm.name, bbox_shm.name, frame_shape, lock, bbox_lock, event, stop_event, tracker_manager))
    inference_process = mp.Process(target=model_inference_process,
                                   args=(shm.name, bbox_shm.name, frame_shape, lock, bbox_lock, event, stop_event, tracker_manager, model_path))

    video_process.start()
    inference_process.start()

    video_process.join()
    inference_process.join()

    # 공유 메모리 해제
    shm.close()
    shm.unlink()
    bbox_shm.close()
    bbox_shm.unlink()
