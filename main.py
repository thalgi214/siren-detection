import os
import numpy as np
import serial
import torch
import torchaudio
from torchaudio.transforms import Spectrogram

SR = 8000
N_FFT = 256
N_BINS = N_FFT // 2 + 1
HOP_LENGTH = 128
DURATION_SEC = 2
CLASSIFY_LEN = SR * DURATION_SEC

C_SPEED = 340.0
MIC_DIST = 0.15
SILENCE_THRESH = 50 / 512.0
CLASSIFY_THRESH = 0.5

PORT = 'COM5'
BAUD_RATE = 500000

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GRU_PATH = "./siren_gru.pth"
CNN_PATH = "./siren_classifier.pth"
DEBUG_WAV_DIR = "./debug_wav"


# GRU 필터 모델
class SirenGRU(torch.nn.Module):
    def __init__(self, input_size=N_BINS, hidden_size=128, num_layers=2):
        super().__init__()
        self.gru = torch.nn.GRU(input_size, hidden_size, num_layers,
                                batch_first=True, bidirectional=False)
        self.fc = torch.nn.Linear(hidden_size, input_size)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        out, _ = self.gru(x)
        return self.sigmoid(self.fc(out))

# CNN 분류 모델
class SirenClassifier2D(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, 3, 1, 1),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(8, 16, 3, 1, 1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2)
        )
        n_frames = SR * DURATION_SEC // HOP_LENGTH + 1
        pooled_time = n_frames // 4
        pooled_bins = N_BINS // 4
        flat_dim = 16 * pooled_time * pooled_bins

        self.fc1 = torch.nn.Linear(flat_dim, 32)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(32, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

# 두 음원으로 TDOA 및 DOA(각도) 구하기
def compute_doa(sig1, sig2, fs): 
    sig1 = sig1 - np.mean(sig1)
    sig2 = sig2 - np.mean(sig2)

    # 무음이면 None 반환
    if np.max(np.abs(sig1)) < SILENCE_THRESH or np.max(np.abs(sig2)) < SILENCE_THRESH:
        return None

    w = np.hanning(len(sig1))
    sig1 = sig1 * w
    sig2 = sig2 * w

    # GCC-PHAT
    n = len(sig1) + len(sig2) - 1
    X1 = np.fft.fft(sig1, n)
    X2 = np.fft.fft(sig2, n)
    R = X1 * np.conj(X2)
    R_phat = R / (np.abs(R) + 1e-10)
    cc = np.fft.ifft(R_phat)
    cc = np.fft.fftshift(cc.real)

    max_idx = np.argmax(cc)
    lags = np.arange(-n // 2, (n + 1) // 2)
    delta_t = lags[max_idx] / fs

    # TDOA -> 각도 계산
    sin_arg = (C_SPEED * delta_t) / MIC_DIST  
    sin_arg = np.clip(sin_arg, -1.0, 1.0)  
    theta = np.degrees(np.arcsin(sin_arg))  
    return delta_t, theta 


stft = Spectrogram(n_fft=N_FFT, hop_length=HOP_LENGTH, power=1.0).to(DEVICE)

gru_model = SirenGRU().to(DEVICE)
gru_model.load_state_dict(torch.load(GRU_PATH, map_location=DEVICE))
gru_model.eval()

cnn_model = SirenClassifier2D().to(DEVICE)
cnn_model.load_state_dict(torch.load(CNN_PATH, map_location=DEVICE))
cnn_model.eval()

hann = torch.hann_window(N_FFT).to(DEVICE)

os.makedirs(DEBUG_WAV_DIR, exist_ok=True)

# 시리얼 연결
ser = serial.Serial(PORT, BAUD_RATE, timeout=1)
ser.reset_input_buffer()

mic1_buffer = []
mic2_buffer = []
debug_idx = 0

while True:
    line = ser.readline().decode('utf-8', errors='ignore').strip()
    if not line:
        continue

    try:
        val1, val2 = line.split(',')
        v1 = (float(val1) - 512.0) / 512.0
        v2 = (float(val2) - 512.0) / 512.0
    except ValueError:
        continue

    mic1_buffer.append(v1)
    mic2_buffer.append(v2)

    # 2초 다 모이면 처리 시작
    if len(mic1_buffer) >= CLASSIFY_LEN:
        block1 = np.array(mic1_buffer[:CLASSIFY_LEN])
        block2 = np.array(mic2_buffer[:CLASSIFY_LEN])
        mic1_buffer = []
        mic2_buffer = []

        # mic1으로 마스크 만들기
        wave1 = torch.from_numpy(block1).float().unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            # STFT: 파형 -> 스펙트로그램 (1, 129, Time)
            spec = stft(wave1)
            spec_2d = spec[0]  # (129, Time)
            gru_input = spec_2d.transpose(0, 1).unsqueeze(0)  # (1, Time, 129)
            pred_mask = gru_model(gru_input)  # (1, Time, 129)
            mask_2d = pred_mask.squeeze(0).transpose(0, 1)  # (129, Time)
            enhanced = spec_2d * mask_2d  # (129, Time)
            cnn_input = enhanced.unsqueeze(0).unsqueeze(0)  # (1, 1, 129, Time)
            prob = cnn_model(cnn_input).item()

        siren = prob >= CLASSIFY_THRESH
        print("[분류] p=%.3f -> %s" % (prob, "사이렌 감지" if siren else "사이렌 없음"))

        # 사이렌이면 방향 계산
        if siren:
            enhanced_waves = []
            for block in [block1, block2]:
                w = torch.from_numpy(block).float().unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    spec = torch.stft(w, n_fft=N_FFT, hop_length=HOP_LENGTH,
                                      window=hann, return_complex=True)
                    mag = torch.abs(spec[0])
                    phase = torch.angle(spec[0])
                    enh_spec = (mag * mask_2d) * torch.exp(1j * phase)
                    enh_wave = torch.istft(enh_spec.unsqueeze(0), n_fft=N_FFT,
                                           hop_length=HOP_LENGTH, window=hann,
                                           length=CLASSIFY_LEN)
                enhanced_waves.append(enh_wave.squeeze(0).cpu().numpy())

            enh1, enh2 = enhanced_waves

            # 디버깅용 wav 저장
            torchaudio.save("%s/enhanced_%d_mic1.wav" % (DEBUG_WAV_DIR, debug_idx),
                            torch.from_numpy(enh1).unsqueeze(0), SR)
            torchaudio.save("%s/enhanced_%d_mic2.wav" % (DEBUG_WAV_DIR, debug_idx),
                            torch.from_numpy(enh2).unsqueeze(0), SR)
            debug_idx += 1

            # DOA 계산 
            result = compute_doa(enh1, enh2, SR)
            if result is not None:
                delta_t, theta = result
                print("시간차(TDOA): %+.3f ms | 추정 각도(DOA): %+.1f도" % (delta_t * 1000, theta))