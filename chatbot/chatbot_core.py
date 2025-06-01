# chatbot/chatbot_core.py

import numpy as np
import config
from rag.vector_store import load_category_vector_db, load_vector_db_by_path
from rag.embedder import load_embedder
from llm.category_classifier import classify_category_with_llm
from llm.router import get_llm_by_category
from llm.responder import build_rag_chain
from llm.chatbot_llm import chatbot_response

# 초기화 (서버 기동시 1회)
embedder = load_embedder()
category_texts, category_categories, category_embeddings = load_category_vector_db()

# FAISS 인덱스
import faiss
index = faiss.IndexFlatL2(category_embeddings.shape[1])
index.add(category_embeddings.numpy())

def run_chatbot_pipeline(user_input: str, session_id: str = "default") -> str:
    # 1. 분류용 벡터 검색
    input_emb = embedder.encode([user_input])
    D, I = index.search(np.array(input_emb).astype("float32"), k=3)
    top_sims = 1 - D[0]  # L2 거리 → 유사도(코사인 기반이라면 별도 처리)
    top_idx = I[0]
    max_sim = top_sims[0]

    #if max_sim < 0.01:
        # context 없이 바로 챗봇 LLM에 전달
    #    return chatbot_response(user_input, "")
    #else:
    # context 확보
    retrieved_examples = [(category_texts[i], category_categories[i], float(top_sims[j])) for j, i in enumerate(top_idx)]
    # 2. 카테고리 분류 LLM
    predicted_category = classify_category_with_llm(user_input, retrieved_examples)
    # 3. 카테고리별 벡터 DB에서 문서 재검색 (context 추출)

    # 다른분들 벡터db오면 병합하기
    index_file, chunks_file = config.VECTOR_DB_PATHS.get(predicted_category, config.VECTOR_DB_PATHS["junbae_assist"])
    context_docs, context_index = load_vector_db_by_path(index_file, chunks_file)
    doc_emb = embedder.encode([user_input])
    D2, I2 = context_index.search(np.array(doc_emb).astype("float32"), k=3)
    top_docs = [context_docs[i] for i in I2[0]]
    context = "\n".join(top_docs)
    # 4. 카테고리별 전문 LLM으로 1차 답변
    category_llm = get_llm_by_category(predicted_category)
    rag_chain = build_rag_chain(category_llm)
    expert_response = rag_chain.invoke({"question": user_input, "context": context})
    # 5. 최종 챗봇 LLM
    return chatbot_response(user_input, expert_response, session_id=session_id)