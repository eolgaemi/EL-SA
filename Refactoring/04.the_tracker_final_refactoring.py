import cv2
import numpy as np
import time
import multiprocessing as mp
from multiprocessing import shared_memory, Lock, Event
from ultralytics import YOLO


# Mock GPIO 클래스
# 실제 하드웨어의 GPIO 대신 사용될 모의 클래스입니다.
# GPIO 제어가 필요한 경우 이 클래스가 대신 사용되며, 실제 하드웨어 없이 테스트 가능합니다.
class MockGPIO:
    BOARD = "BOARD"
    BCM = "BCM"
    OUT = "OUT"
    IN = "IN"
    HIGH = "HIGH"
    LOW = "LOW"

    def __init__(self):
        # 핀 상태를 저장할 딕셔너리
        self.pins = {}

    # GPIO 모드 설정
    def setmode(self, mode):
        print(f"Setting mode to {mode}")

    # GPIO 핀을 입력 또는 출력 모드로 설정
    def setup(self, channel, mode, initial=None):
        self.pins[channel] = {'mode': mode, 'state': initial}
        print(f"Setting up channel {channel} as {mode} with initial {initial}")

    # GPIO 핀에 신호를 출력
    def output(self, channel, state):
        if channel in self.pins:
            self.pins[channel]['state'] = state
            print(f"Setting channel {channel} to {state}")
        else:
            print(f"Channel {channel} not set up yet!")

    # GPIO 핀에서 입력 상태를 읽음
    def input(self, channel):
        return self.pins.get(channel, {}).get('state', None)

    # 모든 핀을 초기 상태로 되돌림
    def cleanup(self):
        self.pins = {}
        print("Cleaning up all channels")

    # GPIO 경고를 설정
    def setwarnings(self, flag):
        print(f"Setting warnings {'on' if flag else 'off'}")

    # PWM 객체를 생성하여 제어할 수 있도록 함
    def PWM(self, channel, frequency):
        print(f"Setting up PWM on channel {channel} with frequency {frequency}")
        return MockPWM(channel, frequency)


# Mock PWM 클래스
# PWM(펄스 폭 변조)을 모의하는 클래스입니다.
class MockPWM:
    def __init__(self, channel, frequency):
        self.channel = channel
        self.frequency = frequency
        self.duty_cycle = 0

    # PWM 신호를 시작
    def start(self, duty_cycle):
        self.duty_cycle = duty_cycle
        print(f"Starting PWM on channel {self.channel} with duty cycle {duty_cycle}")

    # PWM 듀티 사이클 변경
    def ChangeDutyCycle(self, duty_cycle):
        self.duty_cycle = duty_cycle
        print(f"Changing duty cycle on channel {self.channel} to {duty_cycle}")

    # PWM 신호 중지
    def stop(self):
        print(f"Stopping PWM on channel {self.channel}")


# RPi.GPIO 모듈이 없을 경우 MockGPIO를 사용하도록 설정
try:
    import RPi.GPIO as GPIO
except ImportError:
    GPIO = MockGPIO()


# TrackerManager 클래스
# 비디오 프레임에서 객체를 추적하는 트래커를 관리하는 클래스입니다.
class TrackerManager:
    def __init__(self):
        # 트래커 목록을 저장하는 리스트
        self.trackers = []

    # 새 트래커를 생성하여 프레임의 지정된 영역(rect)을 추적
    def create_tracker(self, frame, rect):
        tracker = cv2.legacy.TrackerMOSSE_create()  # MOSSE 트래커 생성
        tracker.init(frame, rect)  # 트래커 초기화
        self.trackers.append(tracker)  # 트래커를 리스트에 추가
        return tracker

    # 프레임 내에서 각 트래커를 업데이트
    def update_trackers(self, frame):
        for tracker in self.trackers:
            success, bbox = tracker.update(frame)
            if success:
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                # 프레임에 추적한 객체의 경계 상자를 그림
                cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)


# YOLO 모델 클래스
# YOLO 객체 탐지 모델을 사용하여 비디오 프레임에서 객체를 탐지하는 클래스입니다.
class YOLOModel:
    def __init__(self, model_path):
        # YOLO 모델 초기화 (모델 경로와 task 명시)
        self.model = YOLO(model_path, task='detect')
        # 탐지할 객체 클래스(교통약자) 매핑
        self.class_dict = {0: 'cane', 1: 'crutches', 2: 'walker', 3: 'wheelchair', 4: 'white_cane'}

    # 주어진 이미지에서 객체 탐지 수행
    def detect(self, img):
        # YOLO 모델로 탐지 수행, 640x640 사이즈의 이미지로 설정, confidence 0.4 이상만 탐지
        results = self.model(img, imgsz=640, conf=0.4)[0]
        return results


# 비디오 재생 프로세스
# 비디오 파일을 읽고, 각 프레임을 공유 메모리를 통해 전달하는 역할
def video_capture_process(video_path, shm_name, shm_bbox_name, shape, lock, bbox_lock, event, stop_event, tracker_manager):
    capture = cv2.VideoCapture(video_path)  # 비디오 파일 열기
    shm = shared_memory.SharedMemory(name=shm_name)  # 프레임 공유 메모리 연결
    bbox_shm = shared_memory.SharedMemory(name=shm_bbox_name)  # 바운딩 박스 정보 공유 메모리 연결
    frame_array = np.ndarray(shape, dtype=np.uint8, buffer=shm.buf)  # 프레임 공유 메모리 배열
    bbox_array = np.ndarray((10, 5), dtype=np.float32, buffer=bbox_shm.buf)  # 바운딩 박스 배열 (최대 10개)

    frame_count = 0  # 프레임 수를 카운트

    while not stop_event.is_set():  # stop_event가 설정되면 프로세스 종료
        ret, frame = capture.read()  # 비디오에서 프레임을 읽음
        if not ret:  # 비디오가 끝나면 종료
            stop_event.set()  # 종료 신호 전송
            event.set()  # 추론 프로세스 종료
            break

        # 프레임을 지정된 크기로 조정하고 색상 변환
        frame_resized = cv2.resize(frame, (shape[1], shape[0]))  # 크기 조정
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)  # BGR을 RGB로 변환

        # 공유 메모리에 프레임 저장
        lock.acquire()
        np.copyto(frame_array, frame_rgb)
        lock.release()

        # 매 60프레임마다 객체 탐지 신호를 보냄
        if frame_count % 60 == 0:
            event.set()  # 모델 추론 프로세스에 신호 전송

        # 바운딩 박스 정보 가져와서 프레임에 그리기
        bbox_lock.acquire()
        for i in range(10):  # 최대 10개의 바운딩 박스
            if bbox_array[i][2] > 0 and bbox_array[i][3] > 0:  # 유효한 바운딩 박스만 처리
                x, y, w, h = bbox_array[i][:4]
                class_id = int(bbox_array[i][4])
                class_label = YOLOModel('').class_dict[class_id]  # 클래스 라벨 가져오기
                cv2.rectangle(frame_resized, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)  # 경계 상자 그리기
                cv2.putText(frame_resized, class_label, (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)  # 라벨 추가
        bbox_lock.release()

        # 결과 보여주기
        cv2.imshow("Video Stream", frame_resized)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q' 키로 종료
            stop_event.set()
            event.set()  # 추론 프로세스 종료
            break

        frame_count += 1  # 프레임 수 증가
        time.sleep(0.03)  # 비디오 재생 속도를 유지

    capture.release()
    cv2.destroyAllWindows()


# 모델 추론 프로세스
# YOLO 모델을 사용해 객체를 탐지하고, 탐지된 정보를 공유 메모리에 저장하는 역할
def model_inference_process(shm_name, shm_bbox_name, shape, lock, bbox_lock, event, stop_event, tracker_manager, model_path):
    yolo_model = YOLOModel(model_path)  # YOLO 모델 초기화
    shm = shared_memory.SharedMemory(name=shm_name)  # 프레임 공유 메모리 연결
    bbox_shm = shared_memory.SharedMemory(name=shm_bbox_name)  # 바운딩 박스 공유 메모리 연결
    frame_array = np.ndarray(shape, dtype=np.uint8, buffer=shm.buf)  # 공유 메모리에서 프레임 읽기
    bbox_array = np.ndarray((10, 5), dtype=np.float32, buffer=bbox_shm.buf)  # 바운딩 박스 배열

    while not stop_event.is_set():  # stop_event가 설정될 때까지 실행
        event.wait()  # 비디오 프로세스로부터 신호 대기
        if stop_event.is_set():
            break  # 종료 신호가 오면 종료

        # 공유 메모리에서 프레임 복사
        lock.acquire()
        frame = frame_array.copy()
        lock.release()

        # YOLO 모델을 사용해 객체 탐지 수행
        results = yolo_model.detect(frame)

        # 탐지된 객체 정보(좌표, 클래스)를 바운딩 박스 배열에 저장
        bbox_lock.acquire()
        bbox_array.fill(0)  # 이전 값 초기화
        for i, result in enumerate(results[:10]):  # 최대 10개의 객체
            xywh = result.boxes.xywh
            class_id = int(result.boxes.cls[0])  # 클래스 ID 가져오기
            x, y, w, h = int(xywh[0][0]), int(xywh[0][1]), int(xywh[0][2]), int(xywh[0][3])
            bbox_array[i] = [x - w // 2, y - h // 2, w, h, class_id]  # 좌표 및 클래스 ID 기록
        bbox_lock.release()

        event.clear()  # 이벤트 리셋


# 메인 프로세스
if __name__ == '__main__':
    # 비디오 파일 경로 및 모델 경로 설정
    video_path = 'Wheelchair_Elevator_inside_in.mp4'
    model_path = '03.ELSA_SEG_model.pt'
    pwm_pins = [12, 32]
    frame_shape = (480, 640, 3)

    # 공유 메모리 생성 (프레임 및 바운딩 박스 정보)
    shm = shared_memory.SharedMemory(create=True, size=int(np.prod(frame_shape) * np.uint8().nbytes))  # 프레임 정보 공유 메모리
    bbox_shm = shared_memory.SharedMemory(create=True, size=10 * 5 * np.float32().nbytes)  # 바운딩 박스 공유 메모리
    lock = Lock()  # 프레임 및 바운딩 박스 동기화를 위한 락
    bbox_lock = Lock()
    event = Event()  # 프로세스 간의 이벤트 신호
    stop_event = Event()  # 종료 신호

    # TrackerManager 객체 생성
    tracker_manager = TrackerManager()

    # 비디오 캡처 및 모델 추론 프로세스 생성
    video_process = mp.Process(target=video_capture_process,
                               args=(video_path, shm.name, bbox_shm.name, frame_shape, lock, bbox_lock, event, stop_event, tracker_manager))
    inference_process = mp.Process(target=model_inference_process,
                                   args=(shm.name, bbox_shm.name, frame_shape, lock, bbox_lock, event, stop_event, tracker_manager, model_path))

    # 프로세스 시작
    video_process.start()
    inference_process.start()

    # 프로세스 종료 대기
    video_process.join()
    inference_process.join()

    # 공유 메모리 해제 및 정리
    shm.close()
    shm.unlink()
    bbox_shm.close()
    bbox_shm.unlink()
