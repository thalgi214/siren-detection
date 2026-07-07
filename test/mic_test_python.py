import sys
import numpy as np
import serial
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore

PORT = "COM8"
BAUD = 460800
FS = 8000 
NFFT = 1024

ser = serial.Serial(PORT, BAUD, timeout=0.1)
sample_buf = np.zeros(NFFT)
byte_buf = bytearray()
freqs = np.fft.rfftfreq(NFFT, d=1.0 / FS)
window = np.hanning(NFFT)

app = QtWidgets.QApplication(sys.argv)
win = pg.GraphicsLayoutWidget(title="MAX9814 Realtime FFT")
win.resize(900, 400)
plot = win.addPlot()
plot.setLabel('left', 'Magnitude (dB)')
plot.setLabel('bottom', 'Frequency', units='Hz')
plot.setXRange(0, FS / 2)
plot.setYRange(0, 150)
curve = plot.plot(pen='c')
win.show()


def update():
    global sample_buf
    byte_buf.extend(ser.read(ser.in_waiting))  # 바이트 버퍼에 누적
    vals = []   # 헤더 기준 값 파싱
    i = 0
    while i + 2 < len(byte_buf):
        if byte_buf[i] == 0xAA:  # 헤더(0xAA) 탐색
            lo = byte_buf[i + 1]
            hi = byte_buf[i + 2]
            if hi <= 0x0F:   # 12비트 검증
                vals.append(lo | (hi << 8))
                i += 3
                continue
        i += 1       # 어긋나면 1바이트 밀어 재정렬
    del byte_buf[:i] # 처리분 제거, 나머지 보존

    if not vals: # 새 값 없으면 종료
        return
    vals = np.array(vals, dtype=float)  # numpy 배열로 변환

    # 버퍼에 최신 샘플 밀어넣기
    if len(vals) >= NFFT:
        sample_buf = vals[-NFFT:]
    else:
        sample_buf = np.roll(sample_buf, -len(vals))
        sample_buf[-len(vals):] = vals

    # DC 성분 제거 후 윈도우 적용
    x = (sample_buf - np.mean(sample_buf)) * window

    # 실수 FFT -> 크기 -> dB
    spectrum = np.abs(np.fft.rfft(x))
    spectrum_db = 20 * np.log10(spectrum + 1e-6)

    curve.setData(freqs, spectrum_db)


timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(30)   # 약 33Hz 화면 갱신

if __name__ == "__main__":
    sys.exit(app.exec_())import sys
import numpy as np
import serial
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore

PORT = "COM8"
BAUD = 460800
FS = 8000 
NFFT = 1024

ser = serial.Serial(PORT, BAUD, timeout=0.1)
sample_buf = np.zeros(NFFT)
byte_buf = bytearray()
freqs = np.fft.rfftfreq(NFFT, d=1.0 / FS)
window = np.hanning(NFFT)

app = QtWidgets.QApplication(sys.argv)
win = pg.GraphicsLayoutWidget(title="MAX9814 Realtime FFT")
win.resize(900, 400)
plot = win.addPlot()
plot.setLabel('left', 'Magnitude (dB)')
plot.setLabel('bottom', 'Frequency', units='Hz')
plot.setXRange(0, FS / 2)
plot.setYRange(0, 150)
curve = plot.plot(pen='c')
win.show()


def update():
    global sample_buf
    byte_buf.extend(ser.read(ser.in_waiting))  # 바이트 버퍼에 누적
    vals = []   # 헤더 기준 값 파싱
    i = 0
    while i + 2 < len(byte_buf):
        if byte_buf[i] == 0xAA:  # 헤더(0xAA) 탐색
            lo = byte_buf[i + 1]
            hi = byte_buf[i + 2]
            if hi <= 0x0F:   # 12비트 검증
                vals.append(lo | (hi << 8))
                i += 3
                continue
        i += 1       # 어긋나면 1바이트 밀어 재정렬
    del byte_buf[:i] # 처리분 제거, 나머지 보존

    if not vals: # 새 값 없으면 종료
        return
    vals = np.array(vals, dtype=float)  # numpy 배열로 변환

    # 버퍼에 최신 샘플 밀어넣기
    if len(vals) >= NFFT:
        sample_buf = vals[-NFFT:]
    else:
        sample_buf = np.roll(sample_buf, -len(vals))
        sample_buf[-len(vals):] = vals

    # DC 성분 제거 후 윈도우 적용
    x = (sample_buf - np.mean(sample_buf)) * window

    # 실수 FFT -> 크기 -> dB
    spectrum = np.abs(np.fft.rfft(x))
    spectrum_db = 20 * np.log10(spectrum + 1e-6)

    curve.setData(freqs, spectrum_db)


timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(30)   # 약 33Hz 화면 갱신

if __name__ == "__main__":
    sys.exit(app.exec_())