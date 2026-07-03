import numpy as np
import serial
import torch
from nnAudio.features.gammatone import Gammatonegram

# ===== 오디오/모델 설정 (학습 스크립트와 동일하게 유지) =====
SR = 8000
N_FFT = 512
N_BINS = 64
HOP_LENGTH = 128
DURATION_SEC = 2
CLASSIFY_LEN = SR * DURATION_SEC   # 16000 샘플 (분류 버퍼 크기)

# ===== 방향 추정 설정 =====
C_SPEED = 340.0
MIC_DIST = 0.15
BLOCK_SIZE = 1024                  # TDOA 블록 크기
SILENCE_THRESH = 50 / 512.0        # 정규화 스케일에 맞춘 무음 임계값

# ===== 판정 설정 =====
CLASSIFY_THRESH = 0.5              # 사이렌 판정 임계값

# ===== 시리얼 설정 =====
PORT = 'COM5'
BAUD_RATE = 500000

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GRU_PATH = "./siren_gru.pth"
CNN_PATH = "./siren_classifier.pth"

WINDOW = np.hanning(BLOCK_SIZE)


class SirenGRU(torch.nn.Module):
    def __init__(self, input_size=N_BINS, hidden_size=128, num_layers=2):
        super().__init__()
        self.gru = torch.nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )
        self.fc = torch.nn.Linear(hidden_size, input_size)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        out, _ = self.gru(x)
        return self.sigmoid(self.fc(out))


class SirenClassifier2D(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
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
    return estimated_delta_t


def compute_doa(mic1_signal, mic2_signal, fs, window,
                c_speed=C_SPEED, mic_dist=MIC_DIST, silence_thresh=SILENCE_THRESH):
    mic1_signal = mic1_signal - np.mean(mic1_signal)
    mic2_signal = mic2_signal - np.mean(mic2_signal)

    if np.max(np.abs(mic1_signal)) < silence_thresh or np.max(np.abs(mic2_signal)) < silence_thresh:
        return None

    mic1_w = mic1_signal * window
    mic2_w = mic2_signal * window

    estimated_delta_t = gcc_phat(mic1_w, mic2_w, fs)

    sin_argument = (c_speed * estimated_delta_t) / mic_dist
    sin_argument = np.clip(sin_argument, -1.0, 1.0)

    estimated_theta = np.degrees(np.arcsin(sin_argument))
    return estimated_delta_t, estimated_theta


def classify_siren(mic1_block, gt, gru_model, cnn_model):
    """mic1 2초 블록(정규화된 numpy) -> 사이렌 확률"""
    wave = torch.from_numpy(mic1_block).float().unsqueeze(0).to(DEVICE)  # (1, 16000)

    with torch.no_grad():
        m_gt = gt(wave)                              # (1, 64, Time)
        gru_input = m_gt[0].transpose(0, 1).unsqueeze(0)   # (1, Time, 64)
        pred_mask = gru_model(gru_input)
        mask_2d = pred_mask.squeeze(0).transpose(0, 1)     # (64, Time)
        enhanced_gt = m_gt[0] * mask_2d                    # (64, Time)

        cnn_input = enhanced_gt.unsqueeze(0).unsqueeze(0)  # (1, 1, 64, Time)
        prob = cnn_model(cnn_input).item()

    return prob


def load_models():
    gt = Gammatonegram(
        sr=SR, n_fft=N_FFT, n_bins=N_BINS, hop_length=HOP_LENGTH,
        fmin=100.0, fmax=4000.0, power=1.0
    ).to(DEVICE)

    gru_model = SirenGRU().to(DEVICE)
    gru_model.load_state_dict(torch.load(GRU_PATH, map_location=DEVICE))
    gru_model.eval()

    cnn_model = SirenClassifier2D().to(DEVICE)
    cnn_model.load_state_dict(torch.load(CNN_PATH, map_location=DEVICE))
    cnn_model.eval()

    return gt, gru_model, cnn_model


def main():
    gt, gru_model, cnn_model = load_models()

    try:
        ser = serial.Serial(PORT, BAUD_RATE, timeout=1)
    except serial.SerialException:
        print("포트 연결 실패")
        return

    mic1_buffer = []   # 분류용 (mic1, 16000샘플)
    mic1_block = []    # 방향용 (mic1, 1024샘플)
    mic2_block = []    # 방향용 (mic2, 1024샘플)

    siren_active = False   # 사이렌 판정 상태 (다음 분류까지 유지)

    ser.reset_input_buffer()

    try:
        while True:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if not line:
                continue

            try:
                val1_str, val2_str = line.split(',')
                # ADC(0~1023) -> -1~1 정규화
                v1 = (float(val1_str) - 512.0) / 512.0
                v2 = (float(val2_str) - 512.0) / 512.0
            except ValueError:
                continue

            mic1_buffer.append(v1)
            mic1_block.append(v1)
            mic2_block.append(v2)

            # --- 방향 추정: 사이렌 활성 상태에서 1024샘플마다 ---
            if len(mic1_block) >= BLOCK_SIZE:
                if siren_active:
                    result = compute_doa(
                        np.array(mic1_block[:BLOCK_SIZE]),
                        np.array(mic2_block[:BLOCK_SIZE]),
                        SR, WINDOW
                    )
                    if result is not None:
                        dt, theta = result
                        print(f"시간차(TDOA): {dt*1000:+.3f} ms | 추정 각도(DOA): {theta:+.1f}°")
                mic1_block = mic1_block[BLOCK_SIZE:]
                mic2_block = mic2_block[BLOCK_SIZE:]

            # --- 분류: 2초(16000샘플) 채워지면 실행 후 버퍼 비움 ---
            if len(mic1_buffer) >= CLASSIFY_LEN:
                block = np.array(mic1_buffer[:CLASSIFY_LEN])
                mic1_buffer = []   # 매번 새로 채움

                prob = classify_siren(block, gt, gru_model, cnn_model)
                siren_active = prob >= CLASSIFY_THRESH
                state = "사이렌 감지" if siren_active else "사이렌 없음"
                print(f"[분류] p={prob:.3f} -> {state}")

    except KeyboardInterrupt:
        pass
    finally:
        ser.close()


if __name__ == "__main__":
    main()