# llm/chatbot_llm.py

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

chatbot_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

# 프롬프트를 별도 변수로 선언
chatbot_prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 의료 질문에 전문적으로 답하는 AI 상담사입니다. 사용자의 질문과 기존 응답을 바탕으로 최적의 답변을 생성하세요."),
    ("human", """[사용자 질문]
{question}

[카테고리 전용 LLM 응답]
{draft_answer}

[최종 응답]""")
])

# 응답 생성 함수
def chatbot_response(question: str, draft_answer: str) -> str:
    return (chatbot_prompt | chatbot_llm | StrOutputParser()).invoke({
        "question": question,
        "draft_answer": draft_answer
    })