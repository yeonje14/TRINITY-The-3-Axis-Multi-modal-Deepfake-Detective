# 🛡️ TRINITY: The 3-Axis Multi-modal Deepfake Detective

> **"화질이 나쁘면 움직임(물리)을 보고, 화질이 좋으면 픽셀(디지털)을 본다."**
> **"Low quality? Check the Physics. High quality? Check the Pixels."**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch MPS](https://img.shields.io/badge/PyTorch-MPS%20Accelerated-E9967A.svg)](https://pytorch.org/)
[![Hardware: M4 Mac](https://img.shields.io/badge/Hardware-Apple%20M4-000000.svg)](https://www.apple.com/macbook-air/)
[![Service: KakaoTalk](https://img.shields.io/badge/Service-KakaoTalk%20Chatbot-F7E600.svg)](https://i.kakao.com/)

---

### 🌍 Language Select
[🇰🇷 **한국어 (Korean)**](#-한국어-korean) | [🇺🇸 **English**](#-english) | [🇯🇵 **日本語 (Japanese)**](#-日本語-japanese)

---

<br>

## 🇰🇷 한국어 (Korean)

### 1. 프로젝트 개요 (Overview)
**TRINITY**는 기존 단일 딥러닝 모델의 한계를 극복하기 위해 물리적 법칙(움직임), 생체 신호(혈류), 디지털 흔적(패턴)을 결합한 **3축 앙상블 딥페이크 탐지 시스템**입니다.
* **환경:** **MacBook Air (M4)** (Apple Silicon GPU 가속 **MPS** 활용)
* **목표:** 다양한 화질과 생성 기법에 대응하는 강건한(Robust) 탐지 엔진 구축 및 **카카오톡 챗봇**을 통한 디지털 취약 계층(노년층) 보호.

### 2. 시스템 파이프라인 (Pipeline)
**"사용자 요청부터 결과 전송까지, 비용 '0원'의 자동화 아키텍처"**
1. **Interface:** 카카오톡 챗봇으로 의심 영상 전송.
2. **Network:** **Cloudflare Tunnel**을 통한 HTTPS 암호화 터널링 (유동 IP/포트 포워딩 해결).
3. **Control:** **Flask + Celery + Redis** 조합으로 M4 멀티코어 비동기 분산 처리.
4. **Preprocessing:** `yt-dlp`를 활용한 **3-Point Biopsy** (전/중/후 10초 핀셋 추출).
5. **Acceleration:** **PyTorch MPS(Metal)** 백엔드로 M4 GPU 성능 100% 가동.

### 3. 핵심 분석 엔진: Trinity 3-Axis
단일 모델의 약점을 상호 보완하는 **3가지 핵심 논문 기술의 앙상블**입니다.

#### **A축: 물리 법칙 감시팀 (Temporal Consistency)**
> *"화질이 뭉개져도 행동(뼈대와 박자)은 거짓말을 못 한다."*

* **가면 검사관 (Texture Inspector)**
  * **원리:** 얼굴 회전 시 피부 픽셀이 골격을 따라오지 못하는 **Texture Sticking** 현상 탐지.
  * **핵심:** 실제 피부는 뼈와 분리될 수 없으므로, 이 불일치는 명백한 가짜의 증거입니다.
  * **Reference:** [ICASSP 2019 (Inconsistent Head Poses)](https://arxiv.org/pdf/1811.00661.pdf)
* **싱크로율 감시관 (Sync Watcher)**
  * **원리:** 턱의 움직임($T_1$)과 입술 근육 반응($T_2$) 사이의 미세한 **위상 지연(Phase Lag)** 탐지.
  * **핵심:** 최신 AI라도 0.1초 단위의 미세한 물리적 근육 박자까지 맞추기는 어렵습니다.
  * **Reference:** [CVPR 2021 (LipForensics)](https://arxiv.org/pdf/2011.06734.pdf)

#### **B축: 생체 신호 감시팀 (Physiological)**
> *"인공지능이 그린 얼굴에는 혈류(심장 박동)가 흐르지 않는다."*

* **원리:** **EVM(영상 증폭)** 기술로 미세 피부색 변화를 증폭하고, **FFT(주파수 분석)**로 실제 심박 신호(rPPG) 존재 여부 판별.
* **Reference:** [IEEE TPAMI (FakeCatcher)](https://arxiv.org/pdf/1901.02212.pdf)

#### **C축: 디지털 정밀 분석팀 (Visual & Pattern)**
> *"화질이 좋을수록 숨겨진 디지털 지문이 선명하게 드러난다."*

* **베테랑 형사 (Pattern Detective)**
  * **원리:** 수만 장의 가짜 영상을 학습한 CNN이 딥페이크 특유의 뭉개진 패턴과 경계선 아티팩트 식별.
  * **Reference:** [ICCV 2019 (FaceForensics++)](https://arxiv.org/pdf/1901.08971.pdf)
* **현미경 분석관 (Microscope Analyst)**
  * **원리:** **EfficientNet-B0**와 **Compound Scaling**을 통해 픽셀 단위의 미세한 생성형 노이즈(격자무늬 등) 정밀 타격.
  * **Reference:** [ICML 2019 (EfficientNet)](https://arxiv.org/pdf/1905.11946.pdf)

### 4. 하드웨어 최적화 (M4 Mac)
* **MPS (Metal Performance Shaders):** NVIDIA CUDA를 대체하여 Apple Silicon GPU 가속 적용.
* **Unified Memory:** CPU-GPU 메모리 공유 구조를 활용하여 고해상도 영상 처리 병목 제거.
* **Green Computing:** 저전력 고효율 M4 칩셋을 활용한 지속 가능한 홈 서버 구축.

---

<br>

## 🇺🇸 English

### 1. Project Overview
**TRINITY** is a **3-axis ensemble deepfake detection system** designed to overcome the limitations of single-modal models by combining **Physical Laws (Motion)**, **Physiological Signals (Blood Flow)**, and **Digital Traces (Patterns)**.
* **Environment:** **MacBook Air (M4)** (Accelerated via Apple Silicon **MPS**)
* **Goal:** To build a robust detection engine capable of handling various video qualities and generation techniques, provided via a **KakaoTalk Chatbot** for accessibility to the digital vulnerable (elderly).

### 2. System Pipeline
**"Zero-Cost Automated Architecture from Request to Response"**
1. **Interface:** User sends suspicious video via KakaoTalk Chatbot.
2. **Network:** HTTPS secure tunneling via **Cloudflare Tunnel** (Solving dynamic IP issues).
3. **Control:** Asynchronous distributed processing using **Flask + Celery + Redis** on M4 multi-cores.
4. **Preprocessing:** **3-Point Biopsy** using `yt-dlp` (Extracting 10s clips from start/mid/end).
5. **Acceleration:** 100% M4 GPU utilization via **PyTorch MPS (Metal)** backend.

### 3. Core Analysis Engine: Trinity 3-Axis

#### **Axis A: Temporal Consistency Team**
> *"Even if quality degrades, physics (skeleton & timing) cannot lie."*

* **Texture Inspector**
  * **Principle:** Detects **Texture Sticking**, where skin pixels fail to follow the skeleton during head rotation.
  * **Ref:** [ICASSP 2019](https://arxiv.org/pdf/1811.00661.pdf)
* **Sync Watcher**
  * **Principle:** Detects **Phase Lag** between jaw movement and lip muscle response.
  * **Ref:** [CVPR 2021](https://arxiv.org/pdf/2011.06734.pdf)

#### **Axis B: Physiological Signal Team**
> *"Artificial faces do not have blood flow (heartbeat)."*

* **Principle:** Uses **EVM (Eulerian Video Magnification)** to amplify subtle skin color changes and **FFT** to detect real heart rate signals (rPPG).
* **Ref:** [IEEE TPAMI](https://arxiv.org/pdf/1901.02212.pdf)

#### **Axis C: Digital Precision Analysis Team**
> *"Higher quality reveals clearer digital fingerprints."*

* **Pattern Detective**
  * **Principle:** CNN trained on thousands of fake videos identifies specific artifacts and blurring patterns.
  * **Ref:** [ICCV 2019](https://arxiv.org/pdf/1901.08971.pdf)
* **Microscope Analyst**
  * **Principle:** Uses **EfficientNet-B0** and **Compound Scaling** to target pixel-level generative noise (Checkerboard artifacts).
  * **Ref:** [ICML 2019](https://arxiv.org/pdf/1905.11946.pdf)

### 4. Hardware Optimization (M4 Mac)
* **MPS Acceleration:** Replaces NVIDIA CUDA with Apple Metal Performance Shaders.
* **Unified Memory:** Eliminates bottlenecks in high-res video processing.
* **Green Computing:** Sustainable home server using low-power M4 silicon.

---

<br>

## 🇯🇵 日本語 (Japanese)

### 1. プロジェクト概要
**TRINITY**は、物理法則（動き）、生体信号（血流）、デジタル痕跡（パターン）を結合した**3軸アンサンブル・ディープフェイク検知システム**です。
* **環境:** **MacBook Air (M4)** (Apple Silicon GPU加速 **MPS** 活用)
* **目標:** 画質や生成手法に関わらず機能する堅牢な（Robust）検知エンジンの構築、および**カカオトーク(KakaoTalk)**を通じたデジタル弱者（高齢者など）の保護。

### 2. システムパイプライン
**「リクエストから結果送信まで、コストゼロの自動化アーキテクチャ」**
1. **Interface:** カカオトークチャットボットで疑わしい動画を送信。
2. **Network:** **Cloudflare Tunnel**によるHTTPS暗号化トンネリング。
3. **Control:** **Flask + Celery + Redis**によるM4マルチコア非同期分散処理。
4. **Preprocessing:** `yt-dlp`を活用した**3点生検（3-Point Biopsy）**（動画の最初・中間・最後を10秒ずつ抽出）。
5. **Acceleration:** **PyTorch MPS(Metal)**バックエンドでM4 GPU性能を100%稼働。

### 3. コア分析エンジン：Trinity 3-Axis

#### **A軸：物理法則監視チーム (時間的一貫性)**
> *「画質が崩れても、行動（骨格と拍子）は嘘をつかない。」*

* **仮面検査官 (Texture Inspector)**
  * **原理:** 顔の回転時に皮膚ピクセルが骨格に追従できない**Texture Sticking（テクスチャの固着）**現象を検知。
  * **Ref:** [ICASSP 2019](https://arxiv.org/pdf/1811.00661.pdf)
* **シンクロ率監視官 (Sync Watcher)**
  * **原理:** 顎の動きと唇の筋肉反応の間の微細な**位相遅延（Phase Lag）**を検知。最新のAIでも0.1秒単位の物理的なズレは模倣困難です。
  * **Ref:** [CVPR 2021](https://arxiv.org/pdf/2011.06734.pdf)

#### **B軸：生体信号監視チーム (生理学的信号)**
> *「AIが描いた顔には血流（心拍）が流れていない。」*

* **原理:** **EVM（映像増幅）**技術で微細な肌色変化を増幅し、**FFT**で実際の心拍信号（rPPG）の有無を判別。
* **Ref:** [IEEE TPAMI](https://arxiv.org/pdf/1901.02212.pdf)

#### **C軸：デジタル精密分析チーム (視覚＆パターン)**
> *「画質が良いほど、隠されたデジタル指紋が鮮明に現れる。」*

* **ベテラン刑事 (Pattern Detective)**
  * **原理:** 数万枚の偽造映像を学習したCNNが、ディープフェイク特有の崩れたパターンや境界線のアーティファクトを識別。
  * **Ref:** [ICCV 2019](https://arxiv.org/pdf/1901.08971.pdf)
* **顕微鏡分析官 (Microscope Analyst)**
  * **原理:** **EfficientNet-B0**と**Compound Scaling**を用いて、ピクセル単位の微細な生成ノイズ（格子模様など）を精密打撃。
  * **Ref:** [ICML 2019](https://arxiv.org/pdf/1905.11946.pdf)

### 4. ハードウェア最適化 (M4 Mac)
* **MPS加速:** NVIDIA CUDAの代わりにApple Metal Performance Shadersを活用。
* **ユニファイドメモリ:** CPU-GPUメモリ共有構造を活用し、ボトルネックを除去。
* **グリーンコンピューティング:** 低電力・高効率なM4チップを活用した持続可能なホームサーバー。

---

## 📚 References
1. **[A-Axis]** Yang et al., *"Exposing DeepFakes Using Inconsistent Head Poses"*, ICASSP 2019.
2. **[A-Axis]** Haliassos et al., *"LipForensics: Irregularities in Semantic High-Level Representations"*, CVPR 2021.
3. **[B-Axis]** Demir et al., *"FakeCatcher: Detection of Synthetic Portrait Videos using Biological Signals"*, IEEE TPAMI.
4. **[C-Axis]** Rössler et al., *"FaceForensics++: Learning to Detect Manipulated Facial Images"*, ICCV 2019.
5. **[C-Axis]** Tan et al., *"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"*, ICML 2019.

---
© 2026 TRINITY Project. Developed by Yeonje Lee.
