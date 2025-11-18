# core 레이어 개요

`core/` 디렉터리는 프로젝트 전반에서 사용하는 **언어 공용 코어 레이어**를 둔 곳이다.
여기에는 여러 언어별 서브 프로젝트가 들어오며, 각 언어에서 공통으로 재사용할 수 있는
**유틸리티, 에러 표현, 공용 타입** 등을 제공한다.

현재는 다음 서브 프로젝트가 존재한다.

- `core/python` → `project-core` 서브 프로젝트
  (파이썬 패키지 이름: `project_core`)

향후에는 필요에 따라 다음과 같은 디렉터리가 추가될 수 있다.

- `core/golang`
- `core/rust`
- `core/typescript`
- …

각 언어별 코어는 가능한 한 **동일한 개념·역할**을 공유하지만,
구체적인 API와 구현은 언어 특성에 맞게 설계한다.

---

## 1. 역할 및 위치

이 레포는 하나의 monorepo이며, 파이썬 서브 프로젝트들 간 의존성 계층은 다음과 같다.

    project_core  →  modeling  →  backend  →  frontend

여기서:

- `core/python` 의 `project-core` 서브 프로젝트는 **최하위 공용 유틸 레이어**이다.
- 상위 계층(`modeling`, `backend`, `frontend`)은 필요한 경우에만
  `project_core`를 의존성으로 추가해서 사용한다.
- 반대로, `core` 레이어는 상위 서브 프로젝트를 import 하거나 직접 참조하지 않는다.

이 계층 규칙은 불변식 **[INV-064]**(“서브 프로젝트 간 import 계층”)을 따른다.

---

## 2. 디렉터리 구조

현재 `core/` 디렉터리 구조는 대략 다음과 같다.

- `core/`
  - `README.md` (이 파일)
  - `python/`
    - `pyproject.toml` (`project-core` 서브 프로젝트 정의)
    - `src/project_core/`
      - `__init__.py`
      - `result.py`
      - …

각 언어별 디렉터리는 **서로 다른 빌드/패키징 시스템**을 쓸 수 있다.

- 예:
  - Python: `pyproject.toml + uv`
  - Go: `go.mod`
  - Rust: `Cargo.toml`
  - TypeScript: `package.json`, `tsconfig.json` …

다만, **“core 레이어 안에 있다”는 사실 자체가**
“상위 애플리케이션 계층에서 공용으로 재사용되는 유틸 모듈”이라는 의미를 가진다.

---

## 3. 설계 원칙

core 레이어 전반에는 다음과 같은 원칙을 적용한다.

1. **상위 계층에 대한 의존 금지**
   - `core` 안의 어떤 서브 프로젝트도 `modeling`, `backend`, `frontend`에 의존하지 않는다.
   - 필요한 공용 개념은 core 쪽으로 당겨서 정의하고,
     상위 계층이 이를 사용하도록 한다.

2. **언어별로 독립적인 빌드/배포 단위**
   - `core/python` (`project-core`), `core/golang`, `core/rust` 등은
     각각 독립적인 라이브러리/모듈로 배포할 수 있도록 설계한다.
   - monorepo 안에 있지만, **각자의 세상에서 쓸 수 있는 라이브러리**를 목표로 한다.

3. **개념의 일관성**
   - 예를 들어 Python의 `Result/Ok/Err` 개념을 다른 언어로 옮길 때,
     이름과 의미를 최대한 맞춘다.
   - 언어별 구현 세부사항은 달라도,
     “core 레이어가 제공하는 개념/역할”은 비슷하게 유지한다.

---

## 4. project-core (Python) 서브 프로젝트와의 관계

`core/python` 디렉터리에는 `project-core` 서브 프로젝트가 있다.

- 프로젝트 이름: `project-core`
- 파이썬 패키지 이름: `project_core`

이 서브 프로젝트는 현재 다음을 제공한다.

- `Result[T, E]`, `Ok[T]`, `Err[E]` 타입
- `is_ok`, `is_err` 헬퍼 함수

자세한 내용과 사용 예시는 `core/python/README.md`를 참고한다.

---

## 5. 개발/기여 시 주의사항

- 새로운 언어별 코어 디렉터리를 추가할 때는:
  - 먼저 해당 언어에서 어떤 공용 개념이 필요한지 최소 단위로 정의한다.
  - 상위 계층(`modeling`, `backend`, `frontend`)과의 의존 방향을
    **core → 상위** 단방향으로 유지한다.
- core 레이어는 **“정말로 여러 계층에서 공통으로 쓸 것들만 넣는 곳”**으로 유지한다.
  - 특정 서브 프로젝트에만 의미가 있는 유틸은 core가 아니라
    해당 서브 프로젝트 내부에 두는 것을 우선 고려한다.
