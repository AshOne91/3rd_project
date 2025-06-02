# SKN12-3rd-5TEAM
## 프로젝트 : LLM을 연동한 내 외부 데이터 학습 후 질의 응답

---
## 팀 소개
### 팀 명 : ????
## 팀 멤버

이미지 들어가는 곳
|:--:|:--:|:--:|:--:|
| **권성호** | **남의헌** | **이준배** | **이준석** | **손현성** |


---
## 프로젝트 목적 : LLM을 활용한 의료 관련 QA 챗봇 구성

---
## 기술 스택
<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/> 
<img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white"/> 
<img src="https://img.shields.io/badge/Json-000000?style=for-the-badge&logo=json&logoColor=white"/> 
<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/> 
<img src="https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white"/> 
<img src="https://img.shields.io/badge/Hugging Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=white"/> 
<img src="https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white"/> 
<img src="https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white"/> 
<img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/> 

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

준배님 pdf 링크
