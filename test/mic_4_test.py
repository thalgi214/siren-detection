import itertools
import struct
import time
from collections import deque

import numpy as np
import spidev
from scipy.signal import decimate

SPI_BUS = 0
SPI_DEV = 0
SPI_SPEED = 10000000
SPI_MODE = 0b00
FRAME_BYTES = 8

SR = 32000
PERIOD = 1.0 / SR
WINDOW_SEC = 0.05
HOP_SEC = 0.5
WINDOW = int(SR * WINDOW_SEC)
HOP = int(SR * HOP_SEC)

BP_LO = 500.0
BP_HI = 2500.0

ADC_CENTER = 2048.0
ADC_SCALE = 2048.0
SILENCE_THRESH = 50 / ADC_SCALE

C_SPEED = 340.0
MIC_L = 0.065
N_MIC = 4
_h = MIC_L / 2.0
MIC_POS = np.array([[+_h, +_h],
                    [+_h, -_h],
                    [-_h, -_h],
                    [-_h, +_h]], dtype=float)
MIC_POS = MIC_POS - MIC_POS.mean(axis=0)

PAIRS = list(itertools.combinations(range(N_MIC), 2))
A_MAT = np.array([MIC_POS[i] - MIC_POS[j] for i, j in PAIRS], dtype=float)
MAX_LAG = np.linalg.norm(A_MAT, axis=1) / C_SPEED * SR


def compute_tdoa(sig_i, sig_j, fs, max_lag):
    a = sig_i - np.mean(sig_i)
    b = sig_j - np.mean(sig_j)

    w = np.hanning(len(a))
    a = a * w
    b = b * w

    n = 1 << (len(a) + len(b) - 2).bit_length()
    Fa = np.fft.rfft(a, n)
    Fb = np.fft.rfft(b, n)

    R = Fb * np.conj(Fa)

    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    R[(freqs < BP_LO) | (freqs > BP_HI)] = 0.0

    UP_FACTOR = 4
    n_up = n * UP_FACTOR
    R_up = np.zeros(n_up // 2 + 1, dtype=complex)
    
    limit = min(len(R), len(R_up))
    R_up[:limit] = R[:limit] * UP_FACTOR

    cc = np.fft.fftshift(np.fft.irfft(R_up, n_up))
    zero = n_up // 2

    m = max(1, int(np.floor(max_lag * UP_FACTOR)))
    lo = zero - m
    hi = zero + m
    k = lo + int(np.argmax(cc[lo:hi + 1]))

    frac = 0.0
    if 0 < k < len(cc) - 1:
        with np.errstate(divide='ignore', invalid='ignore'):
            d = -0.5 * (cc[k + 1] - cc[k - 1]) / (cc[k + 1] - 2.0 * cc[k] + cc[k - 1])
        if np.isfinite(d) and abs(d) <= 1.0:
            frac = d

    tau = (k - zero + frac) / (fs * UP_FACTOR)

    lim = max_lag / fs
    if tau > lim:
        tau = lim
    elif tau < -lim:
        tau = -lim

    return tau


def doa_from_tdoa(mic_pos, pairs, tau, c=C_SPEED):
    P = np.asarray(mic_pos, dtype=float)
    A = np.array([P[i] - P[j] for i, j in pairs], dtype=float)
    b = c * np.asarray(tau, dtype=float)
    u_hat = np.linalg.solve(A.T @ A, A.T @ b)
    phi_deg = np.degrees(np.arctan2(u_hat[1], u_hat[0])) % 360.0
    return float(phi_deg), u_hat


def quantize_phi(phi):
    q = round(phi, -1)
    return float(q % 360.0)


def compute_doa_diagnostic(blocks, fs):
    max_amp = max(np.max(np.abs(b)) for b in blocks)
    if max_amp < SILENCE_THRESH:
        return None
    
    tau = np.array([compute_tdoa(blocks[i], blocks[j], fs, MAX_LAG[m])
                    for m, (i, j) in enumerate(PAIRS)])
    
    phi_all, _ = doa_from_tdoa(MIC_POS, PAIRS, tau)
    
    blocks_8k = [decimate(b, q=4, zero_phase=True) for b in blocks]
        
    return phi_all, blocks_8k


def main():
    spi = spidev.SpiDev()
    spi.open(SPI_BUS, SPI_DEV)
    spi.max_speed_hz = SPI_SPEED
    spi.mode = SPI_MODE

    bufs = [deque(maxlen=WINDOW) for _ in range(N_MIC)]
    since_last = 0

    next_t = time.perf_counter()
    dummy = [0] * FRAME_BYTES

    while True:
        raw = spi.xfer2(list(dummy))
        try:
            vals = struct.unpack('<HHHH', bytes(raw))
        except struct.error:
            vals = None

        if vals is not None and all(0 <= v <= 4095 for v in vals):
            for ch in range(N_MIC):
                bufs[ch].append((vals[ch] - ADC_CENTER) / ADC_SCALE)
            since_last += 1

            if len(bufs[0]) >= WINDOW and since_last >= HOP:
                since_last = 0
                blocks = [np.array(b, dtype=float) for b in bufs]
                
                analysis = compute_doa_diagnostic(blocks, SR)
                if analysis is None:
                    print("[무음]")
                else:
                    phi, blocks_8k = analysis
                    phi_q = quantize_phi(phi)
                    print("방위각 : %6.1f도 -> %3.0f도" % (phi, phi_q))

        next_t += PERIOD
        sleep = next_t - time.perf_counter()
        if sleep > 0:
            time.sleep(sleep)


if __name__ == "__main__":
    main()
