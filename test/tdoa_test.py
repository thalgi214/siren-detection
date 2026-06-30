import wave
import numpy as np


def gcc_phat(sig1, sig2, fs):
    n = len(sig1) + len(sig2) - 1
    X1 = np.fft.fft(sig1, n)
    X2 = np.fft.fft(sig2, n)

    R = X1 * np.conj(X2)
    R_phat = R / (np.abs(R) + 1e-10)

    cc = np.fft.ifft(R_phat)
    cc = np.fft.fftshift(cc.real)

    max_idx = np.argmax(cc)
    lags = np.arange(-n // 2, (n+1) // 2)
    estimated_lag = lags[max_idx]

    estimated_delta_t = estimated_lag / fs
    return estimated_delta_t, lags, cc


def load_wav(path):
    with wave.open(path, 'rb') as wf:
        fs = wf.getframerate()
        sampwidth = wf.getsampwidth()
        n_channels = wf.getnchannels()
        frames = wf.readframes(wf.getnframes())

    dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
    dtype = dtype_map[sampwidth]

    data = np.frombuffer(frames, dtype=dtype).astype(np.float64)
    if n_channels > 1:
        data = data.reshape(-1, n_channels)[:, 0]  # 첫 번째 채널만 사용

    return data, fs


C_SPEED = 340.0
MIC_DIST = 0.15
BLOCK_SIZE = 1024

INPUT1_PATH = "/workspaces/siren-detection/test/input.wav"
INPUT2_PATH = "/workspaces/siren-detection/test/output_delayed.wav"


def main():
    mic1_signal, fs1 = load_wav(INPUT1_PATH)
    mic2_signal, fs2 = load_wav(INPUT2_PATH)

    # 샘플링 레이트 동일하게.
    if fs1 != fs2:
        print(f"경고: 두 파일의 샘플링 레이트가 다릅니다 ({fs1} vs {fs2})")
    fs = fs1

    n_samples = min(len(mic1_signal), len(mic2_signal))
    mic1_signal = mic1_signal[:n_samples]
    mic2_signal = mic2_signal[:n_samples]

    print(f"샘플링 레이트: {fs} Hz | 전체 샘플 수: {n_samples}")

    for start in range(0, n_samples - BLOCK_SIZE + 1, BLOCK_SIZE):
        block1 = mic1_signal[start:start + BLOCK_SIZE]
        block2 = mic2_signal[start:start + BLOCK_SIZE]

        # DC 오프셋 제거
        block1 = block1 - np.mean(block1)
        block2 = block2 - np.mean(block2)

        # 무음 구간 스킵
        if np.max(np.abs(block1)) < 50 or np.max(np.abs(block2)) < 50:
            continue
        
        estimated_delta_t, lags, cc_val = gcc_phat(block1, block2, fs) 
        # lags, cc_val : 디버깅용 변수

        # TDOA 계산
        sin_argument = (C_SPEED * estimated_delta_t) / MIC_DIST
        sin_argument = np.clip(sin_argument, -1.0, 1.0)
        estimated_theta_rad = np.arcsin(sin_argument)
        estimated_theta = np.degrees(estimated_theta_rad)

        print(f"[{start:6d}~{start + BLOCK_SIZE:6d}] "
              f"시간차(TDOA): {estimated_delta_t*1000:+.3f} ms | "
              f"추정 각도(DOA): {estimated_theta:+.1f}°")

if __name__ == "__main__":
    main()
