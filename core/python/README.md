# project-core 서브 프로젝트

이 디렉터리는 프로젝트 전반에서 공유하는 **코어 유틸리티와 에러 표현 계층**을 제공하는
`project-core` 서브 프로젝트이다.
현재는 주로 **Result / Ok / Err** 타입을 중심으로, 도메인/서비스 계층 간의
공용 에러 표현을 담당한다.

- **프로젝트 이름**: `project-core` (pyproject.toml 상의 이름)
- **파이썬 패키지 이름**: `project_core` (import 에 사용하는 이름)

모든 설명은 **이 디렉터리(`core/python/`) 기준**으로 작성한다.

---

## 1. 역할 및 위치

이 레포는 하나의 monorepo이며, 파이썬 서브 프로젝트 간 의존성 계층은 다음과 같다.

    project_core  →  modeling  →  backend  →  frontend

`project-core`는 이 계층의 **최하위 공용 유틸 레이어** 역할을 한다.

- `project_core` 패키지는:
  - 표준 라이브러리, 서드파티 라이브러리, `project_core` 내부 코드만 import 한다.
  - `modeling`, `backend`, `frontend`를 import 하지 않는다. (위 계층 의존 금지)
- 다른 서브 프로젝트는 `project-core`를 의존성으로 추가한 뒤, 다음과 같이 사용한다.

    from project_core import Result, Ok, Err

이 계층 규칙은 불변식 **[INV-064]**(“서브 프로젝트 간 import 계층”)을 따른다.

---

## 2. 제공 기능 개요

현재 `project_core` 패키지는 다음과 같은 기능을 제공한다.

- **Result / Ok / Err 타입**
  - 도메인/서비스 계층 간에 값을 전달할 때,
    **성공/실패를 명시적으로 구분하는 래퍼 타입**을 제공한다.
  - Python 3.12 PEP 695 제너릭 문법을 사용한다.
- **is_ok / is_err 헬퍼 함수**
  - `Result` 값이 `Ok` 인지 `Err` 인지 간단히 검사할 수 있는 유틸리티를 제공한다.

모듈 구성은 대략 다음과 같다.

- `project_core/__init__.py`
  `Ok`, `Err`, `Result`, `is_ok`, `is_err` 를 퍼블릭 API로 재노출한다.
- `project_core/result.py`
  Result 타입의 실제 구현이 들어 있다.

---

## 3. Result / Ok / Err 사용 예시

### 3-1. 개념적 타입 구조

(아래 코드는 개념 설명용 예시이다. 실제 구현은 `project_core/result.py`를 참고한다.)

    from dataclasses import dataclass

    @dataclass(slots=True, frozen=True)
    class Ok[T]:
        __match_args__ = ("value",)
        value: T


    @dataclass(slots=True, frozen=True)
    class Err[E]:
        __match_args__ = ("error",)
        error: E


    type Result[T, E] = Ok[T] | Err[E]

- `Ok[T]`  : 성공 시 값을 담는 래퍼
- `Err[E]` : 실패(에러) 정보를 담는 래퍼
- `Result[T, E]` : `Ok[T] | Err[E]` 유니언 타입

패턴 매칭에서 자연스럽게 사용할 수 있도록 `__match_args__`를 정의한다.

### 3-2. 기본 사용 예시

    from project_core import Result, Ok, Err


    def divide(x: float, y: float) -> Result[float, str]:
        if y == 0:
            return Err("division by zero")
        return Ok(x / y)


    result = divide(4.0, 2.0)

    match result:
        case Ok(value=v):
            print("ok:", v)
        case Err(error=e):
            print("error:", e)

- 성공 시: `Ok(value=2.0)`
- 실패 시: `Err(error="division by zero")`

이 패턴은 **도메인 서비스/유즈케이스/모델링 레이어의 public 함수**에서
외부로 값을 돌려줄 때 사용하는 기본 인터페이스가 된다.
자세한 철학은 불변식 **[INV-042] “Result 스타일 에러 처리”**를 따른다.

### 3-3. is_ok / is_err 헬퍼

    from project_core import Result, Ok, Err, is_ok, is_err


    def divide(x: float, y: float) -> Result[float, str]:
        if y == 0:
            return Err("division by zero")
        return Ok(x / y)


    result = divide(1.0, 0.0)

    print(is_ok(result))  # False
    print(is_err(result)) # True

- 짧은 분기나 단위 테스트에서 유용하게 사용할 수 있는 헬퍼이다.
- 긴 흐름에서는 `match` 구문을 사용하는 것을 기본 스타일로 삼는다.

---

## 4. 다른 서브 프로젝트와의 관계

`project-core`는 **어느 서브 프로젝트에서도 재사용 가능한 공용 유틸만**을 담는다.

- `modeling` 전용 유틸은 `modeling` 내부에 둔다.
- `backend` 전용 유틸은 `backend` 내부에 둔다.
- `frontend`에서 파이썬 코드가 필요하다면 `project_core`를 import 할 수 있지만,
  가능한 한 HTTP API를 통해 `backend`를 호출하는 방식을 우선한다.

의존성 방향은 항상 아래와 같이 유지한다.

    project_core  →  modeling  →  backend  →  frontend

예시:

    # modeling 쪽 코드 예시
    from project_core import Result, Ok, Err

    # backend 쪽 코드 예시
    from project_core import Result, Ok, Err

이 규칙을 지키면, 나중에 `backend`를 Go 등 다른 언어로 교체하더라도
`project-core`와 `modeling`의 설계를 그대로 재사용하기 쉬워진다.

---

## 5. 개발/기여 시 주의사항

- 새로운 코어 유틸리티를 추가하기 전에 다음을 먼저 고민한다.
  - “이 기능이 정말로 모든 서브 프로젝트에서 공통으로 쓸 만한가?”
  - `modeling` 또는 `backend`에 더 가깝다면 해당 서브 프로젝트에 두는 것을 우선 고려한다.
- Result/Ok/Err의 내부 구현은 **향후 다른 Result 구현체(표준 라이브러리, 서드파티 등)**로
  교체 가능하도록 유지하는 것을 목표로 한다.
  - 외부 코드에서는 `Ok/Err/Result` 타입과 `match` 패턴에만 의존하도록 작성하는 것을 권장한다.
- 제너릭/타입 힌트와 관련해서는 프로젝트 전반의 불변식:
  - [INV-045] (PEP 695 제너릭 우선),
  - [INV-046] (typing.TypeVar/Generic 최소화),
  - [INV-047] (type 키워드를 사용한 타입 별칭)
  에 맞추어 점진적으로 정리해 나간다.
