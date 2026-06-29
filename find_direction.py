import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

#GCC-PHAT 알고리즘을 이용해 두 음원 신호 사이의 시간 지연(TDOA)을 추정하는 함수
def gcc_phat(sig1, sig2, fs):
    
    # FFT
    n = len(sig1) + len(sig2) - 1
    X1 = np.fft.fft(sig1, n)
    X2 = np.fft.fft(sig2, n)
    
    # convolution X1 & X2*
    # R(f)=X1​(f)⋅X2∗​(f)=∣X1​(f)∣∣X2​(f)∣ej(θ1​(f)−θ2​(f))
    R = X1 * np.conj(X2)
    
    # PHAT 가중치 적용 -> 크기를 1로 나누어 위상 정보만 남김 
    # 분모가 0이 되는 것 방지하기 위해 미세한 값(1e-10) 추가
    R_phat = R / (np.abs(R) + 1e-10)
    
    # IFFT -> fftshift(정렬)
    cc = np.fft.ifft(R_phat)
    cc = np.fft.fftshift(cc.real)
    '''
    왼쪽 절반: 음수 lag (sig2가 sig1보다 빠른 경우)
    중앙: lag = 0
    오른쪽 절반: 양수 lag (sig2가 sig1보다 느린 경우)
    '''

    max_idx = np.argmax(cc)
    lags = np.arange(-n//2, n//2) #샘플 지연
    estimated_lag = lags[max_idx] 
    
    # 샘플 지연 개수를 시간(초) 단위로 변환
    estimated_delta_t = estimated_lag / fs
    
    return estimated_delta_t, lags, cc

