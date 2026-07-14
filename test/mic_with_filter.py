import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_DEBUG_CPU_TYPE"] = "5"
os.environ["USE_MKLDNN"] = "0"
os.environ["USE_AVX"] = "0"

import itertools
import struct
import time
from collections import deque

import numpy as np
import spidev
import torch
import torch.nn as nn
from scipy.signal import decimate

class SirenGRU(nn.Module):
    def __init__(self, input_size=129, hidden_size=128, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )
        self.fc = nn.Linear(hidden_size, input_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.gru(x)
        return self.sigmoid(self.fc(out))

class SirenCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc1 = nn.Linear(15872, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

SPI_BUS = 0
SPI_DEV = 0
SPI_SPEED = 10000000
SPI_MODE = 0b00
FRAME_BYTES = 8

SR = 32000
PERIOD = 1.0 / SR

WINDOW_SEC = 0.05                   
HOP_SEC = 0.5                       
CNN_WINDOW_SEC = 2.0                

WINDOW = int(SR * WINDOW_SEC)
HOP = int(SR * HOP_SEC)
CNN_WINDOW = int(SR * CNN_WINDOW_SEC)

BP_LO = 500.0
BP_HI = 3500.0

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

device = torch.device('cpu')

gru_model = SirenGRU().to(device)
gru_model.load_state_dict(torch.load('siren_gru.pth', map_location=device))
gru_model.eval()

cnn_model = SirenCNN().to(device)
cnn_model.load_state_dict(torch.load('siren_classifier.pth', map_location=device))
cnn_model.eval()


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
    if tau > lim: tau = lim
    elif tau < -lim: tau = -lim

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
    return phi_all


def run_pipeline_inference(cnn_blocks):
    mono_sig = np.mean(cnn_blocks, axis=0)
    
    cnn_8k = decimate(mono_sig, q=4, zero_phase=True)
    
    n_fft = 256
    hop_length = 128
    padded_sig = np.pad(cnn_8k, (n_fft // 2, n_fft // 2), mode='reflect')
    
    window = np.hanning(n_fft)
    stft_res = np.array([np.fft.rfft(padded_sig[i:i+n_fft] * window) 
                         for i in range(0, len(padded_sig) - n_fft + 1, hop_length)])
    
    magnitude = np.abs(stft_res)
    
    input_tensor = torch.tensor(magnitude, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        mask = gru_model(input_tensor)
        mask = mask.squeeze(0).cpu().numpy()
        
    clean_spectrogram = magnitude * mask
    
    cnn_input = torch.tensor(clean_spectrogram.T, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        prob = cnn_model(cnn_input).item()
        
    return prob > 0.5


def send_to_stm32(spi, phi, is_detected):
    start_byte = 0xA5
    active_flag = 1 if is_detected else 0
    angle_val = int(phi)
    
    tx_data = struct.pack('<BBH', start_byte, active_flag, angle_val)
    spi.xfer2(list(tx_data))


def main():
    spi = spidev.SpiDev()
    spi.open(SPI_BUS, SPI_DEV)
    spi.max_speed_hz = SPI_SPEED
    spi.mode = SPI_MODE

    bufs = [deque(maxlen=WINDOW) for _ in range(N_MIC)]
    cnn_bufs = [deque(maxlen=CNN_WINDOW) for _ in range(N_MIC)]
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
                val = (vals[ch] - ADC_CENTER) / ADC_SCALE
                bufs[ch].append(val)
                cnn_bufs[ch].append(val)
            since_last += 1

            if len(bufs[0]) >= WINDOW and since_last >= HOP:
                since_last = 0
                blocks = [np.array(b, dtype=float) for b in bufs]
                
                phi = compute_doa_diagnostic(blocks, SR)
                if phi is None:
                    print("[무음]")
                    send_to_stm32(spi, 0.0, False)
                else:
                    phi_q = quantize_phi(phi)
                    
                    is_detected = False
                    if len(cnn_bufs[0]) == CNN_WINDOW:
                        cnn_blocks = [np.array(b, dtype=float) for b in cnn_bufs]
                        is_detected = run_pipeline_inference(cnn_blocks)
                        
                    status_str = "[감지]" if is_detected else "[미감지]"
                    print("%6.1f도 -> %3.0f도 | 사이렌 유무: %s" % (phi, phi_q, status_str))
                    
                    send_to_stm32(spi, phi_q, is_detected)

        next_t += PERIOD
        sleep = next_t - time.perf_counter()
        if sleep > 0:
            time.sleep(sleep)


if __name__ == "__main__":
    main()
