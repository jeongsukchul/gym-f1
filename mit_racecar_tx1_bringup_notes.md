# MIT RACECAR / Jetson TX1 Bring-up Notes

## 0. 목적

현재 보유 중인 구형 MIT RACECAR 계열 1/10 autonomous car를 살리고,
F1TENTH simulator에서 학습한 policy를 실제 차량에 배포할 수 있도록 준비한다.

이 문서는 지금까지의 조사/결정 사항을 정리한 것이다.

이후 이 대화에서는 주로 다음 두 가지에 집중한다.

1. **Hardware checking / bring-up**
2. **System identification (SysID)**

소프트웨어 구조/정책 배포 코드는 별도로 Codex에서 `catkin_ws/src`를 조사하면서 진행할 수 있다.

---

# 1. 현재 차량/컴퓨팅 상태

## Jetson

현재 onboard compute:

- NVIDIA Jetson TX1 계열
- Ubuntu 18.04.6 LTS
- L4T:

```text
# R32 (release), REVISION: 7.2, GCID: 30192233,
BOARD: t186ref, EABI: aarch64,
DATE: Sun Apr 17 09:53:50 UTC 2022
```

즉 JetPack 4.6.x / L4T R32.7.2 계열이다.

현재 ROS:

```bash
ls /opt/ros
```

결과:

```text
melodic
```

```bash
echo $ROS_DISTRO
```

결과:

```text
melodic
```

따라서 현재 TX1에는 **ROS1 Melodic**이 이미 설치되어 있다.

기존 workspace:

```text
/home/nvidia/catkin_ws
```

확인됨.

---

# 2. ROS2 관련 결정

Ubuntu 18.04 / TX1에 ROS2 Eloquent를 새로 얹는 것은 현재 목적상 필요하지 않다.

실제로:

```bash
sudo apt install ros-eloquent-ros-base
```

는

```text
E: Unable to locate package ros-eloquent-ros-base
```

가 발생했다.

하지만 현재 프로젝트에서 핵심은 최신 Neo/ROS2 자체가 아니라:

```text
observation -> learned policy -> action
```

pipeline을 실제 차량에서 잘 재현하는 것이다.

따라서 현재 권장 방향:

```text
TX1
Ubuntu 18.04
ROS1 Melodic
existing catkin_ws
```

를 최대한 보존한다.

ROS2 migration은 우선 중단한다.

---

# 3. 최종 연구용 abstraction

F1TENTH simulator에서 학습한 policy를 실차에 배포한다.

권장 action convention:

```text
[v_cmd, delta_cmd]
```

where

- `v_cmd`: desired velocity [m/s]
- `delta_cmd`: desired steering angle [rad]

Neo native API의

```text
speed ∈ [-1, 1]
angle ∈ [-1, 1]
```

같은 normalized actuator abstraction을 연구용 최상위 API로 사용할 필요는 없다.

실차 backend에서만:

```text
v_cmd [m/s]
    ↓
VESC ERPM / velocity tracking

delta_cmd [rad]
    ↓
servo command
```

으로 변환한다.

---

# 4. F1TENTH simulator -> 실차 pipeline

최종 목표:

```text
F1TENTH simulator
       ↓
trained policy
       ↓
exported model
       ↓
TX1 onboard inference
       ↓
[v_cmd, delta_cmd]
       ↓
vehicle adapter
       ↓
VESC / steering servo
```

가장 중요한 것은 ROS 버전이 아니라 **sim-to-real interface alignment**이다.

---

# 5. Observation alignment

예시 policy observation:

```text
[
    LiDAR scan,
    velocity,
    yaw rate
]
```

실차에서는 simulator와 아래를 동일하게 맞춰야 한다.

## LiDAR

확인할 항목:

- number of beams
- angular field of view
- angular ordering
- clockwise / counter-clockwise convention
- minimum range
- maximum range
- clipping
- normalization
- invalid/inf handling
- temporal stacking 여부

예:

```text
raw Hokuyo scan
   ↓
crop
   ↓
resample
   ↓
clip
   ↓
normalize
   ↓
policy observation
```

## Velocity

초기에는 VESC ERPM 기반으로 추정 가능:

\[
\hat v
=
\frac{\mathrm{ERPM}-b_v}{k_v}
\]

추후 필요하면 LiDAR odometry / visual odometry / external tracking으로 보정한다.

## Yaw rate

IMU gyro의 z-axis를 사용할 수 있다.

중요:

```text
left turn -> positive yaw rate?
right turn -> negative yaw rate?
```

가 simulator convention과 동일해야 한다.

---

# 6. Action adapter

기본 actuator mapping:

\[
\mathrm{ERPM}
=
k_v v_{\rm cmd}+b_v
\]

\[
u_{\rm servo}
=
k_\delta \delta_{\rm cmd}+b_\delta
\]

단, 위 gain/offset은 차량마다 직접 calibration해야 한다.

F1TENTH config의 예제 값을 그대로 사용하면 안 된다.

---

# 7. Steering calibration

차량을 반드시 stand 위에 올리고 motor 구동은 끈 상태에서 수행한다.

목표:

\[
u_{\rm servo}
\leftrightarrow
\delta
\]

측정 예:

```text
servo command
-0.3
-0.2
-0.1
 0.0
 0.1
 0.2
 0.3
```

각 command에서 실제 front wheel steering angle을 측정한다.

최소한 linear approximation:

\[
u_{\rm servo}
=
k_\delta \delta + b_\delta
\]

를 fitting한다.

특히 먼저 확인할 것:

```text
delta_cmd = 0
```

일 때 실제 front wheel이 정면을 향하는가.

---

# 8. Velocity calibration / SysID 1단계

초기에는 복잡한 dynamic bicycle SysID보다:

```text
commanded velocity
    ↓
VESC ERPM
    ↓
actual vehicle speed
```

관계를 먼저 맞춘다.

초기 목표:

\[
v_{\rm cmd}
\leftrightarrow
\mathrm{ERPM}
\leftrightarrow
v_{\rm measured}
\]

예:

```text
0.5 m/s
1.0 m/s
1.5 m/s
2.0 m/s
```

등의 저속 command에서 실제 이동 거리 / 시간으로 speed를 확인한다.

---

# 9. SysID 단계적 계획

## Stage 1 — actuator calibration

### Steering

\[
u_{\rm servo}
\rightarrow
\delta
\]

### Motor

\[
u_{\rm motor}
\rightarrow
v
\]

또는

\[
\mathrm{ERPM}
\rightarrow
v
\]

---

## Stage 2 — kinematic bicycle validation

초기 모델:

\[
x_{t+1}
=
x_t+\Delta t\,v_t\cos\psi_t
\]

\[
y_{t+1}
=
y_t+\Delta t\,v_t\sin\psi_t
\]

\[
\psi_{t+1}
=
\psi_t
+
\Delta t
\frac{v_t}{L}
\tan\delta_t
\]

여기서 wheelbase \(L\)은 직접 측정 가능하다.

먼저 실제 trajectory와 이 모델의 residual을 본다.

---

## Stage 3 — dynamic model (필요할 때만)

고속 주행 / racing에서 kinematic model이 부족해지면:

- mass \(m\)
- yaw inertia \(I_z\)
- front/rear cornering stiffness \(C_f, C_r\)
- tire-road friction \(\mu\)
- \(l_f, l_r\)
- steering delay
- motor delay

등을 추가한다.

처음부터 여기까지 할 필요는 없다.

---

# 10. F1TENTH와 RACECAR/Neo의 차이

## F1TENTH

명시적인 vehicle dynamics를 사용.

대표 dynamic state:

\[
[X,Y,\delta,v,\psi,r,\beta]
\]

대표 low-level dynamics input:

\[
[\dot\delta,a]
\]

또는 high-level control abstraction:

```text
[v_cmd, delta_cmd]
```

실차에서도 AckermannDrive 계열의 physical-unit interface를 쓰는 것이 일반적이다.

---

## RACECAR Neo

Neo API는:

```text
speed ∈ [-1,1]
angle ∈ [-1,1]
```

형태의 normalized abstraction이다.

`speed=0.5`는 `0.5 m/s`라는 뜻이 아니다.

따라서 F1TENTH policy를 그대로 이식하려면 Neo normalized action보다:

```text
[v_cmd (m/s), delta_cmd (rad)]
```

를 유지하는 것이 낫다.

---

# 11. Neo simulator에 대한 결정

현재 목표에서는 Neo simulator를 사용하지 않는다.

학습:

```text
F1TENTH simulator
```

실차:

```text
old MIT RACECAR hardware
```

를 사용한다.

Neo software ecosystem 전체를 이식할 필요도 없다.

---

# 12. onboard vs external policy

최종적으로는 **policy를 onboard Jetson에서 실행하는 것이 권장**된다.

이유:

- Wi-Fi latency
- jitter
- packet loss
- autonomous racing control frequency
- safety

때문이다.

최종 구조:

```text
Hokuyo
IMU
VESC
   ↓
TX1
   ↓
observation builder
   ↓
policy inference
   ↓
[v_cmd, delta_cmd]
   ↓
low-level vehicle adapter
   ↓
VESC
```

외부 PC는:

- SSH
- logging
- monitoring
- model transfer
- visualization
- parameter tuning

정도에 사용한다.

---

# 13. TX1에서 PyTorch가 반드시 필요한 것은 아님

현재 TX1은 legacy software stack이다:

```text
Ubuntu 18.04
JetPack 4.6.x
ROS Melodic
older CUDA/Python
```

따라서 최신 PyTorch environment를 그대로 이식하는 것보다 model export를 고려할 수 있다.

후보:

1. TorchScript
2. ONNX
3. TensorRT
4. 단순 MLP라면 NumPy / Eigen forward implementation

예:

\[
h_1=\mathrm{ReLU}(W_1o+b_1)
\]

\[
h_2=\mathrm{ReLU}(W_2h_1+b_2)
\]

\[
a=W_3h_2+b_3
\]

따라서 핵심은 policy runtime compatibility이지 ROS2 compatibility가 아니다.

---

# 14. 현재 전원 상태

기존 MIT RACECAR는 원래 보통 전원계가 두 개였다.

```text
electronics battery
   ├─ Jetson
   ├─ LiDAR
   ├─ camera
   ├─ IMU
   └─ powered USB hub

motor battery
   └─ VESC
       ├─ motor
       └─ steering servo
```

현재 차량에는 motor battery만 확인되었고,
electronics battery는 없거나 분실된 상태로 보인다.

---

# 15. 새로운 electronics battery 방향

전용 19V DC powerbank를 반드시 쓸 필요는 없다.

Jetson TX1은 USB-C PD를 직접 입력받는 장치는 아니지만:

```text
USB-C PD powerbank
     ↓
15V PD trigger cable
     ↓
DC 5.5 × 2.5 mm
     ↓
Jetson TX1 DC input
```

구성이 가능하다.

TX1의 DC barrel:

```text
OD 5.5 mm
ID 2.5 mm
center positive
```

15V PD를 사용하는 이유:

- 15V × 3A = 45W
- TX1 구동에 충분한 수준
- 20V PD는 TX1 입력 상한을 넘어갈 수 있어 피하는 편이 안전

권장 powerbank 특성:

```text
20,000~25,000mAh
USB-C PD ≥ 65W
15V/3A PDO 지원
USB-A 별도 출력
```

후보로 UGREEN PB720 100W 20K 등이 논의되었다.

---

# 16. Hokuyo UST-10LX 전원

UST-10LX는 USB-powered sensor가 아니다.

기본 구조:

```text
Hokuyo UST-10LX
├─ Ethernet → Jetson
└─ Power/I/O cable
```

power:

```text
Brown = +VIN
Blue  = GND
```

입력은 대략:

```text
10–30 V DC
```

range이다.

현재 차량에서 LiDAR power cable 끝이 USB 형태라면,
원래 wiring이 아니라 누군가 개조한 케이블일 가능성이 높다.

이 USB가 정말 표준 5V USB인지 확인되기 전에는
Jetson USB 포트 등에 임의로 연결하면 안 된다.

---

# 17. Networking

현재 Ethernet port는 Hokuyo LiDAR에 사용할 계획이다.

따라서 TX1 remote access:

```text
Wi-Fi wlan0
```

를 사용한다.

권장 network separation:

```text
Wi-Fi wlan0:
192.168.1.x/24

LiDAR eth0:
192.168.0.x/24

Hokuyo:
192.168.0.10
```

처럼 서로 subnet이 겹치지 않게 한다.

---

# 18. HDMI / boot 상태

TX1은 HDMI 연결 후 정상적으로 화면 출력이 확인되었다.

현재 login user:

```text
nvidia
```

기존 system은 정상 부팅한다.

따라서:

- OS reinstall 금지
- 무리한 Ubuntu upgrade 금지
- 기존 `catkin_ws` 삭제 금지
- 기존 drivers 확인 전 `apt upgrade` 남발 금지

가 권장된다.

---

# 19. 현재 USB 상태

센서/VESC는 의도적으로 연결하지 않은 상태에서:

```bash
lsusb
```

결과:

```text
Bus 002 Device 003: ID 2109:0813 VIA Labs, Inc.
Bus 002 Device 002: ID 2109:0813 VIA Labs, Inc.
Bus 002 Device 001: ID 1d6b:0003 Linux Foundation 3.0 root hub
Bus 001 Device 005: ID c0f4:0201
Bus 001 Device 004: ID 1ea7:0066
Bus 001 Device 003: ID 2109:2813 VIA Labs, Inc.
Bus 001 Device 002: ID 2109:2813 VIA Labs, Inc.
Bus 001 Device 001: ID 1d6b:0002 Linux Foundation 2.0 root hub
```

VIA Labs entries는 USB hub 계열로 보인다.

VESC / LiDAR / IMU는 아직 연결하지 않았으므로
현재 출력에서 해당 device를 찾으려고 할 필요는 없다.

---

# 20. Codex에서 다음으로 조사할 것

현재 workspace:

```text
/home/nvidia/catkin_ws
```

Codex에서 우선:

```bash
find ~/catkin_ws/src -maxdepth 2 -type d
```

또는:

```bash
tree -L 3 ~/catkin_ws/src
```

를 사용해 기존 package 구조를 조사한다.

특히 찾을 것:

```text
vesc
vesc_driver
vesc_ackermann
ackermann_mux
racecar
urg_node
hokuyo
imu
zed
joy
teleop
```

등.

추가로:

```bash
grep -R "speed_to_erpm" ~/catkin_ws/src
grep -R "steering_angle_to_servo" ~/catkin_ws/src
grep -R "ackermann" ~/catkin_ws/src
```

를 수행하면 기존 차량 calibration/config를 찾을 가능성이 높다.

---

# 21. Codex에서 보존해야 할 원칙

기존 `catkin_ws`는 가능한 한 수정하지 말고 먼저 읽고 분석한다.

특히:

```text
config files
launch files
VESC gains
servo offsets
LiDAR IP
IMU topics
existing udev rules
```

을 우선 확인한다.

새 코드는 가능하면 별도 package:

```text
racecar_policy
racecar_observation
racecar_vehicle_interface
racecar_safety
```

등으로 추가한다.

---

# 22. 앞으로 이 대화에서 다룰 범위

이 ChatGPT thread에서는 이후 주로:

## Hardware checking

- battery/power
- Jetson
- VESC
- servo
- motor
- LiDAR
- IMU
- camera
- USB hub
- wiring
- temperatures
- network
- safety

## SysID

- steering calibration
- ERPM-speed calibration
- wheelbase
- steering limits
- throttle dead-zone
- steering dead-zone
- motor delay
- steering delay
- velocity tracking
- kinematic bicycle validation
- dynamic bicycle identification
- sim-to-real dynamics mismatch

을 다룬다.

코드 구조/기존 ROS package 분석은 Codex 쪽에서 진행하는 것을 기본으로 한다.

---

# 23. 가장 가까운 다음 hardware 단계

센서/VESC 연결 전 현재 상태에서:

1. electronics power 안정성 확인
2. Wi-Fi 연결 및 SSH 확인
3. TX1 온도/전원 확인
4. 기존 ROS workspace 보존
5. 이후 장치를 **하나씩** 연결

권장 순서:

```text
VESC communication only
→ steering only
→ motor on stand
→ LiDAR
→ IMU
→ camera
```

각 subsystem을 개별적으로 검증한 후 integration한다.

---

# 24. Safety

실차 actuator test에서는:

- 차량을 stand 위에 올릴 것
- motor test 전에 주변 비울 것
- 처음에는 low command만 사용할 것
- steering과 motor를 동시에 처음 테스트하지 말 것
- software watchdog을 둘 것
- command timeout 시 `v_cmd = 0`
- 물리적인 emergency stop 또는 즉시 motor battery disconnect 수단 확보

가 권장된다.

최종적으로:

\[
\text{communication/policy failure}
\Rightarrow
v_{\rm cmd}=0
\]

가 항상 성립해야 한다.
