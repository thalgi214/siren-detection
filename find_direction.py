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
                      c_speed=340.0, mic_dist=0.4, silence_thresh=50):
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

    # Hanning 윈도우 적용 
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
