WALT — Wrapper for Adaptive LLM Tuning

경량 Flask 서버 기반의 LLM 래퍼(Wrapper) 입니다.
일반 노트북/PC(16–32GB RAM)에서도 동작하도록 설계되었고, 메모리 인지형 자동 모델 선택, 요약 기반 컨텍스트 축약, 웹 UI/대시보드, 상태/로그 API, 간이 재학습 트리거, CORS & Bearer 인증(일부 엔드포인트) 를 제공합니다.
Windows 전용 CLI 실행 도구인 waltcli.cmd 로 init / run / stop / status / retrain / switch-model / stream-test / logs / update / clean 등을 한 파일로 제어합니다.

Copyright © 2025 Wanseok Kim(김완석)

License: MIT

Contact: ws-kim@naver.com, wskim4home@gmail.com



---

주요 기능

자동/수동 모델 선택: 가용 메모리·장치 상황을 감지하여 TinyLlama / Phi-2 / GPT4All-J / Mistral-7B / CodeLlama-7B 등 후보를 자동 선택. switch-model로 수동 스위치 가능.

요약 기반 컨텍스트 축약: 긴 대화는 요약 파이프라인(BART 등)으로 압축 후 프롬프트에 삽입.

웹 UI & 대시보드: /(UI), /status(상태), /debug_log(로그 뷰).

운영 API: /health, /data_status, /model_version, /retrain(Bearer), /feedback(Bearer).

보안: CORS 허용 범위 설정, Bearer 인증(선택 엔드포인트).

스트리밍(SSE): /chat?stream=true로 문장/토큰 단위 스트리밍.


> ※ “GPT”라는 이름의 상용 모델 API는 직접 사용하지 않습니다(옵션 확장 가능). 기본은 Hugging Face Transformers 로컬 모델입니다.


---

시스템 요구 사항

Windows 10/11, PowerShell

Python 3.10+

(선택) CUDA GPU

디스크 여유 10GB 이상(모델 캐시)


---

빠른 시작 (Windows)

1. 저장소/코드 폴더에 waltcli.cmd, walt.py(또는 saltl.py)를 둡니다.


2. 최초 1회 초기화:



waltcli init

3. 실행:


waltcli run --port=5000 --token=나의토큰

4. 상태 확인:


waltcli status

5. 중지:


waltcli stop

> 기본 설정 파일은 %USERPROFILE%\.walt\walt.env 에 저장됩니다.
포트/토큰/오프라인 모드 등은 실행 시 플래그나 .env로 변경 가능합니다.


---

CLI 사용법

waltcli init                     > venv 생성, 필수 패키지 설치, 기본 .env 작성
waltcli run [--port=5000] [--host=0.0.0.0] [--offline] [--token=XXX] [--app=walt.py]
waltcli stop                     > 서버 중지(PID 또는 포트 기반)
waltcli check                    > /health 확인
waltcli status                   > /status JSON 출력
waltcli retrain [TOKEN]          > /retrain 호출(Bearer)
waltcli switch-model MODEL_ID    > 수동 모델 스위치(예: codellama/CodeLlama-7b-Instruct-hf)
waltcli stream-test              > SSE 스트리밍 테스트(curl 필요)
waltcli logs                     > 최근 로그 200줄
waltcli update                   > 라이브러리 업데이트(또는 git pull)
waltcli clean                    > venv/캐시/로그 정리
waltcli env                      > 현재 환경 변수 표시
waltcli help                     > 도움말

예시

waltcli init
waltcli run --port=8001 --offline --token=mytoken
waltcli switch-model codellama/CodeLlama-7b-Instruct-hf
waltcli retrain mytoken
waltcli stream-test


---

환경 변수(.walt\walt.env)

Key	기본값	설명

WALT_HOME	%USERPROFILE%\.walt	작업/캐시/로그 기본 디렉터리
WALT_PORT	5000	서버 포트
WALT_HOST	0.0.0.0	바인딩 호스트
WALT_AUTH_TOKEN	s3cr3t	Bearer 토큰(운영 엔드포인트 접근)
WALT_OFFLINE	0	1이면 오프라인 모드(원격 다운로드 차단)
WALT_APP_FILE	walt.py	앱 엔트리 파일(없으면 saltl.py 탐색)


---

API 개요

GET / : 웹 UI(채팅/상태/로그)

POST /chat?model=<hf-id>&stream=<true|false>
Body(JSON): { "message": "<사용자 입력>" }

GET /status : 모델/리소스/메트릭 요약

GET /health : 헬스 체크(OK/DOWN)

GET /data_status : 데이터 적재/크기 등 요약

GET /model_version : 현재 모델/디바이스 정보

GET /debug_log : 최근 로그(Js UI에서 폴링)

POST /retrain : Bearer 토큰 필요(간이 재학습 트리거)

POST /feedback : Bearer 토큰 필요(간단 피드백 수집)


> stream=true 시 SSE(Server-Sent Events)로 부분 응답을 순차 전송합니다.



---

모델 자동/수동 선택

자동

실행 시 가용 메모리·장치(CPU/GPU)를 점검하며 다음 순서로 시도(환경에 맞춰 조정됨):

TinyLlama 1.1B

Phi-2 (~2.7B)

GPT4All-J (~6B)

Mistral-7B

CodeLlama-7B (Instruct)


로컬 캐시(HF) 우선, 실패 시 재시도/다운로드(온라인 모드).


수동(권장 방법)

waltcli switch-model codellama/CodeLlama-7b-Instruct-hf

또는 직접 호출:

POST /chat?model=codellama/CodeLlama-7b-Instruct-hf
{"message": "모델 스위치 후 워밍업"}

대략 가이드(환경 따라 다름)

모델	권장 메모리(시작점)	비고

TinyLlama 1.1B	16GB	가장 가벼움
Phi-2 (~2.7B)	16–24GB	합리적 균형
GPT4All-J (~6B)	24–32GB	초기 로드 다소 큼
Mistral-7B	32GB+	고품질/다목적
CodeLlama-7B	32GB+	코드 태스크 강점


> 더 작은 양자화(gguf 등)·GPU 사용 시 체감 향상.


---

데이터 적재

기본 샘플 데이터는 도메인 중립의 공개 코퍼스 일부를 병합해 로컬 JSON 으로 저장합니다.

daily_dialog, conv_ai_2, nps_chat

persona_chat은 현재 허브에서 명칭/권한 차이로 실패할 수 있습니다.
→ 실패 시 자동 건너뛰거나, 대체 데이터(예: blended_skill_talk)로 바꾸세요.


> 에러 예시: Dataset 'persona_chat' doesn't exist... → 정상. 다른 데이터로 대체하거나 무시해도 서버 기동은 됩니다.


---

스트리밍(SSE)란?

모델 추론 결과를 한 번에 보내는 대신 문장/토큰 단위로 분할 전송하는 방식입니다.
UI에서 Streaming 옵션을 켜거나, CLI로 테스트:

waltcli stream-test


---

보안

개발 기본값은 CORS 허용(*)과 개발용 Flask 서버입니다.

운영 시 반드시:

토큰 변경 (--token 또는 .env)

방화벽/프록시 뒤에서 실행

프로덕션 WSGI(예: waitress) 사용 고려


---

문제 해결(FAQ)

1) GPT4All-J 로드가 매우 느림 / 처음엔 오래 걸렸다

첫 구동 시 체크포인트 샤드 다운로드/캐시 및 가중치 로드로 시간이 걸립니다.

디스크 속도, 실시간 백신 검사, 메모리 여유, 가상 메모리(pagefile) 설정이 영향.

가속 팁:

모델을 미리 받아두기(HF 캐시 경로에 사전 배치)

%USERPROFILE%\.walt\hf_cache 경로를 백신 실시간 검사 제외

더 작은 모델/양자화 모델 사용, 또는 GPU 사용

오프라인 모드(--offline)로 불필요한 네트워크 대기 제거



2) persona_chat 로드 에러가 보인다

공개 허브에서 이름/권한이 바뀌어 정상적으로 실패할 수 있습니다.
자동 건너뛰며 서버는 기동합니다. 필요하면 대체 코퍼스를 지정하세요.


3) 요약 파이프라인 경고 / model_kwargs not used: ['cache_dir']

Transformers 버전 차이로 나타날 수 있는 경고입니다.
pip install -U transformers 후 재시도하거나, 요약 파이프라인 초기화 시 cache_dir 전달을 생략/수정하도록 설정하세요(코드에 반영됨).


4) 개발 서버 경고

Flask 기본 서버는 개발용입니다. 운영 배포 시 WSGI 서버 사용을 권장합니다.


---

아키텍처(요약)

+--------------------------------------------------------------+
| Web UI / API Gateway  (/, /chat, /status, /debug_log, ...)  |
|  - 채팅 UI, 상태 대시보드, 로그 뷰어, CORS, Bearer 인증      |
+------------------------------+-------------------------------+
|        Wrapper Layer         |          Data Layer           |
|  - 컨텍스트 요약/프롬프트    |  - 로컬 JSON/로그/지표        |
|  - 후처리/평가(경량)         |  - (옵션) 임베딩/벡터DB       |
+------------------------------+-------------------------------+
|           Model Layer (auto-pick / manual switch)            |
|  TinyLlama / Phi-2 / GPT4All-J / Mistral-7B / CodeLlama-7B   |
|  (GPU 사용 시 가속, 실패 시 폴백)                             |
+--------------------------------------------------------------+


---

라이선스

MIT License
© 2025 Wanseok Kim(김완석)
별도 명시가 없는 한, 본 저장소의 코드/스크립트/문서는 MIT 하에 자유롭게 사용/수정/배포 가능합니다.


---

크레딧

Hugging Face Transformers / Datasets

PyTorch

Flask / Flask-CORS

(옵션) FAISS



---

Roadmap (요약)

[ ] 임베딩 + 벡터DB(RAG) 기본화

[ ] PII 마스킹/프롬프트 인젝션 가드

[ ] 상관관계 ID·표준 에러 코드·SSE 개선

[ ] 평가/관측 표준 대시보드

[ ] 상용 모델(클라우드) 하이브리드 라우팅


---

연락처

이메일: ws-kim@naver.com / wskim4home@gmail.com

제안/버그 리포트 환영합니다.

