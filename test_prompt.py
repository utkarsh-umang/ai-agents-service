from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from ai_agents.agents.thumbnail_generator.prompts import thumbnail_prompt

load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

chain = thumbnail_prompt | llm | StrOutputParser()

result = chain.invoke(
    {
        "system_prompt": "Focus on high contrast and bold typography.",
        "title": "10 Python Tips That Will Change How You Code",
        "include_title": True,
        "creative_comments": "Make it feel urgent and exciting",
    }
)

print(result)
