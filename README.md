# siren-detection

1. 프로젝트 개요
본 프로젝트는 도로 환경의 잡음 속에서 긴급차량의 사이렌 소리를 탐지하고, 그 음원의 방향과 위치를 추정하는 시스템이다.
전체 파이프라인은 IRM 필터를 통한 잡음 억제, TDOA 연산을 통한 도달 시간차 추정, 다중 어레이 융합의 세 단계로 구성된다.

2. 소프트웨어 구성
2.1 신호 전처리 — Ideal Ratio Mask 기반 잡음 억제

2.2 도달 시간차 추정 — GCC-PHAT, TDOA연산

2.3 음원 위치 추정 — BFGS

3. 하드웨어 구성


4. 참조 문헌
M. H. Soni and H. A. Patil, "Effectiveness of ideal ratio mask for non-intrusive quality assessment of noise suppressed speech," 2017 25th European Signal Processing Conference (EUSIPCO), Kos, Greece, 2017, pp. 573-577, doi: 10.23919/EUSIPCO.2017.8081272.

C. Knapp and G. Carter, "The generalized correlation method for estimation of time delay," in IEEE Transactions on Acoustics, Speech, and Signal Processing, vol. 24, no. 4, pp. 320-327, August 1976, doi: 10.1109/TASSP.1976.1162830. 

Madhu, N. and Martin, R. (2008). Acoustic Source Localization with Microphone Arrays. In Advances in Digital Speech Transmission (eds R. Martin, U. Heute and C. Antweiler). https://doi.org/10.1002/9780470727188.ch6

## 진동모터 연결 참고
https://ko.ineed-motor.com/news/how-to-build-a-vibration-motor-circuit-35660850.html

## Sallen-key 2차 저역 통과 필터 (셀런키 필터)
https://m.blog.naver.com/joa_quin/221143654000