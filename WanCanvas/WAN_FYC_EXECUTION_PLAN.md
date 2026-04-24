# Wan2.2 + FYC Video Outpainting 실행 계획

## 목적
이 문서는 **Wan2.2-TI2V-5B 기반 Video Outpainting**에 FYC의 핵심 방법론
(**layout / relative geometry / known-region preservation**)을
**DiT/Wan 실제 실행 경로에 붙이는 작업**을 위해 만든 실행 기준 문서다.

## 현재 상태 요약
현재 코드는 아래까지는 되어 있다.
- FYC 샘플 계약 / geometry / known-region / multi-round planning 존재
- layout / geometry / mask-summary encoder 존재
- FYC -> Wan bridge / wrapper / dry-run scaffold 존재

하지만 아직 아래는 안 되어 있다.
- `WanOutpaintWrapper.forward()` real forward 미구현
- prompt-only TI2V가 아니라 real outpainting inference로 닫히지 않음
- train dry-run만 있고 `forward + backward + optimizer.step()` 없음

즉, **지금은 bridge/dry-run 구조**이고,
**아직 “FYC를 Wan DiT에 실제 적용했다”라고 강하게 말할 단계는 아니다.**

## 최종 목표
아래 문장이 기술적으로 참이 되게 만든다.

> FYC의 핵심 방법론을 U-Net 기반 구조에서 Wan/DiT 기반 구조로 옮겨,
> Wan2.2_5B를 이용한 고해상도 Video Outpainting 실행 경로를 구현했다.

## 반드시 만족해야 하는 완료 조건
1. `WanOutpaintWrapper.forward()`가 실제 Wan forward를 수행한다.
2. FYC conditioning이 Wan 내부 conditioning 경로에서 실제로 소비된다.
3. inference가 prompt-only TI2V가 아니라 anchor/mask/plan 기반 outpainting을 수행한다.
4. training smoke pass에서 `forward + backward + optimizer.step()`가 돈다.
5. high-resolution outpainting 전략(멀티라운드 + tile refine + merge)이 명시되고 동작한다.
6. 쓸모없어진 dry-run/TI2V-only 코드는 `WanCanvas_Trash`로 이동된다.

---

## 작업 원칙
- 기존의 좋은 자산은 최대한 재사용한다.
- **bridge-only 구조를 확장**하는 게 아니라, **real Wan execution path**로 바꾼다.
- 새 구조가 통과하기 전에는 기존 파일을 버리지 않는다.
- **replace first, archive later** 규칙을 지킨다.

---

## 단계별 실행 계획

### 1. 아키텍처 확정
먼저 아래를 확정한다.
- FYC의 `layout / geometry / mask`를 Wan 어디에 넣을지
- phase-1에서 어떤 모듈을 freeze / train 할지
- 어떤 파일을 유지 / 수정 / 폐기할지

### 권장 방향
- **1순위:** Wan의 실제 conditioning stream에 연결
- **2순위:** known-region / anchor latent는 I2V-style latent conditioning 활용

이 단계가 끝나면 **아키텍처 노트 + trash matrix**를 만든다.

---

### 2. bridge를 real Wan input path로 변경
수정 대상:
- `wancanvas/models/wan_outpaint_wrapper.py`
- `wancanvas/models/fyc_sample_bridge.py`
- `wancanvas/models/fyc_conditioning.py`

핵심 작업:
- wrapper의 `forward()` 활성화
- placeholder prompt embed 제거
- FYC tokens를 Wan이 실제로 소비하는 tensor로 변환
- 필요 시 projector/adapter 추가

이 단계 완료 기준:
- 더 이상 “bundle ready / no forward call” 상태가 아니어야 함

---

### 3. real outpainting inference runner 구현
현재 `ti2v_runner.py`는 minimal TI2V surface라서 목표에 맞지 않는다.

해야 할 일:
- plan 결과를 실제 generation loop에 연결
- anchor / known-region / tile / round 기반 생성
- tile merge / overlap 처리
- 결과로 **실제 outpainted video** 저장

재사용 대상:
- `wancanvas/pipelines/window_scheduler.py`
- `wancanvas/pipelines/known_region.py`
- `wancanvas/pipelines/wan_outpaint_pipeline.py`

---

### 4. real training smoke path 구현
현재 dry-run training을 실제 smoke training으로 교체한다.

초기 권장 학습 정책:
- freeze: VAE, text encoder, 대부분의 Wan backbone
- train: FYC projector/adapters + Wan conditioning 경로용 LoRA/adapter

필수 작업:
- `target_video`를 실제 supervision에 연결
- outpaint / known-region / seam 관련 loss 구성
- backward
- optimizer step

이 단계 완료 기준:
- 1 batch 학습 smoke가 실제로 돈다.

---

### 5. high-resolution 전략 추가
목표가 high-resolution이므로 base generation만으로 끝내면 안 된다.

권장 순서:
1. aligned base pass
2. multi-round expansion
3. high-res tile refine
4. overlap merge

---

### 6. 폴더 정리 및 Trash 이동
## Active tree에 남길 것
- `wancanvas/data/*`
- `wancanvas/pipelines/*`
- `wancanvas/backbones/*`
- 실제로 쓰이는 `wancanvas/models/*`

## 수정 후 계속 사용할 것
- `wancanvas/models/wan_outpaint_wrapper.py`
- `wancanvas/models/fyc_sample_bridge.py`
- `wancanvas/models/fyc_conditioning.py`

## 대체 후 `WanCanvas_Trash` 이동 후보
- `wancanvas/train/contracts.py`
- `wancanvas/train/dry_run.py`
- `scripts/train_wancanvas.py`
- `scripts/run_ti2v_inference.py`
- `configs/infer-real-ti2v.yaml`
- `configs/infer-smoke.yaml`
- `configs/train-skeleton.yaml`
- `tests/test_training_dry_run.py`
- `tests/test_ti2v_runner.py`

규칙:
- **새 경로가 통과한 뒤 이동**
- active code에서 import가 남아 있으면 이동 금지

---

## 최종 검증
아래가 모두 만족되면 완료로 본다.
- train smoke pass 성공
- inference smoke pass 성공
- archived file import 잔존 없음
- README/문서가 새 구조를 정확히 설명함

---

## Codex 실행 순서
1. 아키텍처 노트 작성
2. trash matrix 작성
3. real Wan conditioning forward 구현
4. real outpainting inference 구현
5. real training smoke 구현
6. high-resolution pass 추가
7. obsolete 코드 Trash 이동
8. 최종 검증 및 README 정리

---

## 참고 문서
- 상세 PRD: `.omx/plans/prd-wan-fyc-dit-video-outpainting.md`
- 테스트 스펙: `.omx/plans/test-spec-wan-fyc-dit-video-outpainting.md`
- 컨텍스트 스냅샷: `.omx/context/wan-fyc-dit-video-outpainting-plan-20260424T081741Z.md`
