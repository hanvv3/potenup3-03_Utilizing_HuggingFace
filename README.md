# uv 프로젝트 관리하기

```
uv init # 프로젝트 생성
uv venv # 가상환경 생성
uv add 라이브러리 # 라이브러리 설치
```

## gpu 사용이 필요한 프로젝트를 하는 경우

```
uv init # 프로젝트 생성
uv venv # 가상환경 생성 (python 최신버전 피하기)
```

복잡한 모델의 경우에는 torch 의존성에 취약하기 때문에 pyproject.toml에서 관리해줘야 한다.

pyproject.toml 파일 관리: 맨 아래에 추가
```
[tool.uv.sources]
torch = { index = "pytorch-cu126" }
torchvision = { index = "pytorch-cu126" }

[[tool.uv.index]]
name = "pytorch-cu126"
url = "https://download.pytorch.org/whl/cu126"
explicit = true
```

그 다음에 torch, torchvision, torchaudio 를 설치해야 한다.
```
uv add torch torchvision torchaudio
```

# 기존 사용하고 있던 프로젝트 올바르게 바꾸기

* venv: 새로운 환경에 파이썬 버전을 설치한다.
* pyproject.toml: 설치할 때 필요한 라이브러리와 호환가능한 버전 명시
* 라이브러리 추가/삭제: `uv add 라이브러리`/`uv remove 라이브러리`

## 파이썬 버전을 바꾸고 싶어요.
* 가상환경을 지우고, 다시 설치하고, `uv sync`

## torch 버전을 바꾸고 싶어요.
* 라이브러리 삭제 `uv remove torch torchvision torchaudio`
* pyproject.toml 파일 아래에 위의 cuda를 명시
* 다시 설치 `uv add torch torchvision torchaudio`
* 재시작
* 터미널에서 버전 확인 방법

##