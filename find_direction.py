import numpy as np
import serial
import time

def gcc_phat(sig1, sig2, fs):
    n = len(sig1) + len(sig2) - 1
    X1 = np.fft.fft(sig1, n)
    X2 = np.fft.fft(sig2, n)

    R = X1 * np.conj(X2)
    R_phat = R / (np.abs(R) + 1e-10)

    cc = np.fft.ifft(R_phat)
    cc = np.fft.fftshift(cc.real)

    max_idx = np.argmax(cc)
    lags = np.arange(-n // 2, (n + 1) // 2)
    estimated_lag = lags[max_idx]

    estimated_delta_t = estimated_lag / fs
    return estimated_delta_t, lags, cc


def compute_tdoa_doa(mic1_signal, mic2_signal, fs, window,
                      c_speed=340.0, mic_dist=0.15, silence_thresh=50):
    """
    mic1, mic2 신호 블록 하나를 받아 TDOA(시간차)와 DOA(각도)를 계산.
    무음 구간이면 None을 반환.
    """
    # DC 오프셋 제거
    mic1_signal = mic1_signal - np.mean(mic1_signal)
    mic2_signal = mic2_signal - np.mean(mic2_signal)

    # 신호 크기가 작으면 무음으로 간주
    if np.max(np.abs(mic1_signal)) < silence_thresh or np.max(np.abs(mic2_signal)) < silence_thresh:
        return None

    # Hanning 윈도우 적용 (블록 경계에서의 스펙트럼 누설 감소)
    mic1_w = mic1_signal * window
    mic2_w = mic2_signal * window

    # GCC-PHAT 계산
    estimated_delta_t, lags, cc_val = gcc_phat(mic1_w, mic2_w, fs)

    # DOA 계산
    sin_argument = (c_speed * estimated_delta_t) / mic_dist
    sin_argument = np.clip(sin_argument, -1.0, 1.0)

    estimated_theta_rad = np.arcsin(sin_argument)
    estimated_theta = np.degrees(estimated_theta_rad)

    return estimated_delta_t, estimated_theta


C_SPEED = 340.0
MIC_DIST = 0.15
FS = 8000         # 아두이노 설정에 맞춘 샘플링 주파수 (8kHz)
BLOCK_SIZE = 1024 # 한 번에 처리할 샘플 수
PORT = 'COM5'
BAUD_RATE = 500000

WINDOW = np.hanning(BLOCK_SIZE)  # 매 블록마다 새로 만들지 않도록 미리 생성

def main():
    try:
        ser = serial.Serial(PORT, BAUD_RATE, timeout=1)
        print(f"아두이노와 연결")
    except serial.SerialException:
        print("포트 연결 실패")
        return

    mic1_buffer = []
    mic2_buffer = []

    print("방향 추정 시작")

    ser.reset_input_buffer()

    try:
        while True:
            line = ser.readline().decode('utf-8', errors='ignore').strip()

            if not line:
                continue

            try:
                # 아두이노에서 전송한 "val1,val2" 형식 파싱
                val1_str, val2_str = line.split(',')
                mic1_buffer.append(float(val1_str))
                mic2_buffer.append(float(val2_str))
            except ValueError:
                # 통신 노이즈로 인한 파싱 에러 무시
                continue

            # 설정한 블록 크기만큼 데이터가 쌓이면 연산 수행
            if len(mic1_buffer) >= BLOCK_SIZE:
                mic1_signal = np.array(mic1_buffer[:BLOCK_SIZE])
                mic2_signal = np.array(mic2_buffer[:BLOCK_SIZE])

                # 처리한 만큼만 버퍼에서 제거 (남은 샘플은 다음 블록에 사용)
                mic1_buffer = mic1_buffer[BLOCK_SIZE:]
                mic2_buffer = mic2_buffer[BLOCK_SIZE:]

                result = compute_tdoa_doa(mic1_signal, mic2_signal, FS, WINDOW,
                                           c_speed=C_SPEED, mic_dist=MIC_DIST)

                if result is None:
                    continue  # 무음 구간 스킵

                estimated_delta_t, estimated_theta = result

                print(f"시간차(TDOA): {estimated_delta_t*1000:+.3f} ms | 추정 각도(DOA): {estimated_theta:+.1f}°")

    except KeyboardInterrupt:
        print("\n프로그램 종료.")
    finally:
        ser.close()

if __name__ == "__main__":
    main()