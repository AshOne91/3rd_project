# SKN12-3rd-5TEAM
## 프로젝트 : LangChain 및 RAG 활용 의료 LLM 개발(MediChain)

---
## 팀 소개
### 팀 명 : Windows
## 팀 멤버

| <img src="https://cdn.discordapp.com/attachments/1377154931663962197/1379616067894247444/2Q.png?ex=6840e316&is=683f9196&hm=0ccf631b168d7c31e9399748acb80162066afa1122591c9e911c79e7497cef78&" width="700"/> | <img src="https://cdn.discordapp.com/attachments/1090653800563486763/1379631543454138400/ddef0542b3be7530.png?ex=6840f180&is=683fa000&hm=889489a13110879d1ac364f1fbfd25cb1730ea46fcca83009b63c2f1b742bef7&" width="700"/> | <img src="https://cdn.discordapp.com/attachments/1377154931663962197/1379615665199255622/49836_55372_349.png?ex=6840e2b6&is=683f9136&hm=84bb12bc687de639f4bb2924806127d216fa87231ce1481b53d4e0846531e5fc&" width="700"/> | <img src="https://cdn.discordapp.com/attachments/1377154931663962197/1379617527549919232/Z.png?ex=6840e472&is=683f92f2&hm=3c304b9c2f15b99c904f0e892930512c4489f2bad96d3f02b8110d3ceafeca8e&" width="700"/> | <img src="https://cdn.discordapp.com/attachments/1377154931663962197/1379615995517341776/Z.png?ex=6840e305&is=683f9185&hm=351753f3e14d112ee5f684976e1d86a137dfaa137112933d3fc57093b350bf28&" width="700"/> |
|:--:|:--:|:--:|:--:|:--:|
| **권성호** | **남의헌** | **이준배** | **이준석** | **손현성** |

---
## 프로젝트 목적

#### 의료 관련 문서를 학습한 후 사용자의 질문에 대해 LLM을 활용하여 정확한 답변을 제공하는 QA 챗봇 시스템 개발

---
## 기술 스택

**언어 및 데이터 처리** <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/> <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white"/> , <img src="https://img.shields.io/badge/Json-000000?style=for-the-badge&logo=json&logoColor=white"/>

**머신러닝/딥러닝**<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/> <img src="https://img.shields.io/badge/Hugging Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=white"/>

**임베딩 및 벡터 검색** <img src="https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white"/> <img src="https://img.shields.io/badge/FAISS-84BC34?style=for-the-badge&logo=faiss&logoColor=white"/>

**LLM & API** <img src="https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white"/> <img src="https://img.shields.io/badge/Hugging Face Inference API-FFD21E?style=for-the-badge&logo=hfa&logoColor=white"/>

**앱 및 프론트엔드** <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>

**개발 환경** <img src="https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white"/>

---

## 구성도

<pre><code>
llm_category_chatbot/
│
├── app.py                        # Streamlit 앱 실행 진입점
├── main.py                       # 전처리/임베딩 등 백엔드 초기 작업 진입점
├── config.py                     # 공통 설정 (모델명, 경로 등)
├── requirements.txt              # 설치할 패키지 목록
├── vector_db/                    # FAISS 인덱스 저장 폴더
│   ├── faiss_index.faiss
│   ├── doc_embeddings.pkl
│   ├── doc_ids.json
│   └── categories.json
├── data/
│   ├── raw_docs/                 # 원문 문서 저장 폴더
│   └── processed_docs/           # 전처리 후 카테고리별 문서 저장 폴더
│
├── preprocessing/
│   └── preprocess.py             # 문서 전처리 및 재구성 스크립트
│
├── rag/
│   ├── embedder.py               # 임베딩 생성기 (예: QLoRA, SentenceBERT 등)
│   ├── vector_store.py           # FAISS 인덱싱 및 검색 모듈
│   └── categorizer.py            # 질문을 기반으로 카테고리 분류
│
├── llm/
│   ├── router.py                 # 카테고리 → 전문 LLM에 질문 라우팅
│   └── responder.py              # Few-shot Prompt 구성 및 응답 생성
│
├── chatbot/
│   └── chatbot_core.py           # Streamlit에서 사용하는 질문 처리 파이프라인
│
└── assets/
    └── logo.png                  # Streamlit UI에 표시할 로고 등 리소스
</code></pre>

---
## 데이터 전처리

#### 내용 추출 및 병합

    JSON 파일의 content 필드만 추출

    여러 파일 또는 문서를 하나의 텍스트 데이터로 병합

    불용어 제거

    한국어 불용어 리스트 기반으로 불필요한 단어 제거

    텍스트 정제 (예: 특수문자, 공백 등)

#### 임베딩
    사용 모델:

    jhgan/ko-sroberta-multitask (한국어 특화 모델)

    sentence-transformers/all-MiniLM-L6-v2 (다국어 대응 모델)

    각 문장을 임베딩하여 고차원 벡터로 변환

#### 벡터 DB 생성 (FAISS)
    FAISS 라이브러리를 사용하여 임베딩 벡터 저장 및 인덱싱

    유사도 기반 검색을 위한 벡터 데이터베이스 구축

현성님 전처리 pdf 링크

---
## 시스템 아키텍처

![_](https://cdn.discordapp.com/attachments/1346621776909570109/1378963876887920761/1.png?ex=683e83b0&is=683d3230&hm=2e18d608a69697dbb5f690ae05a75da85ca6e756ec91f5d8afb4a2dd474a2aba&)

---
## LLM 모델 테스트

- GPT-3.5 Turbo
- GPT-3.5 Turbo(FT)
- GPT-4o-mini
- KULLM(max 1024, sampling)
- KULLM(max 256, no sampling)

상위 5개 모델 테스트 결과 GPT-4o-mini 모델이 가장 우수함

세부 내용은 다음 pdf 참조

[LLM 테스트 계획 및 결과 보고서](https://github.com/AshOne91/3rd_project/blob/main/output/%ED%85%8C%EC%8A%A4%ED%8A%B8%20%EA%B3%84%ED%9A%8D%20%EB%B0%8F%20%EA%B2%B0%EA%B3%BC%20%EB%B3%B4%EA%B3%A0%EC%84%9C.pdf)

---
## 사용방법

![](https://cdn.discordapp.com/attachments/1377154931663962197/1379632675589062788/image.png?ex=6840f28e&is=683fa10e&hm=67f3bafce129d38ab4fa5149bc5b5ab3d47183499171bb37028e3271475eb713&)

1. 사용자 구분을 위한 세션 아이디 작성

2. 질문 작성 후 "질문하기" 버튼을 눌러 응답 확인
