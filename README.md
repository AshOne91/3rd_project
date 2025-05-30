```
llm_category_chatbot/
│
├── app.py                        # Streamlit 앱 실행 진입점
├── main.py                       # 전처리/임베딩 등 백엔드 초기 작업 진입점
├── config.py                     # 공통 설정 (모델명, 경로 등)
├── requirements.txt              # 설치할 패키지 목록
├── vector_db/                 # ✅ FAISS 인덱스 저장 폴더
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
```
