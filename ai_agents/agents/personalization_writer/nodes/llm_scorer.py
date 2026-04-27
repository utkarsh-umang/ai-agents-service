from typing import TypedDict, Dict, List
from langchain_groq.chat_models import ChatGroq
from langchain_core.prompts import PromptTemplate

from dotenv import load_dotenv
import os  
from personalization_writer.state import AgentState

from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).parent / ".env")
    
groq_api_key=os.environ.get("GROQ_API_KEY")


def create_prompt(responses: Dict):
    prompt_dict = {}
    for cat, texts in responses.items():
        prompt_list = []
        clean_texts = [t for t in texts if t is not None]
        for i,text in enumerate(clean_texts):
            prompt_list.append(f"{i+1}: {text[:1000]}")
        prompt_string = "\n".join(prompt_list)
        prompt_dict.update({f"{cat}": f"{prompt_string}"})

    
    return prompt_dict


def llm_response(state: AgentState) -> AgentState:

    prompt_dict = create_prompt(state['scraped_urls'])

    llm = ChatGroq(temperature=0.2, groq_api_key=groq_api_key,model="llama-3.3-70b-versatile")

    BASE_PROMPT = """
    You are given scraped content for a specif category from a website


    Choose to summarize or pick best one from the scraped content provided

    return only the single text block containing scraped data (whether it be summarized or best overall)

    """

    state['llm_responses'] = []

    for cat, content in prompt_dict.items():
        template = """

        {base_prompt}

        Category: {category}

        Scraped content:

        {scraped_content}

        """

        prompt_template = PromptTemplate.from_template(template)
    
        prompt = prompt_template.format(
            base_prompt = BASE_PROMPT,
            category = cat,
            scraped_content = content

        )

        result = llm.invoke(prompt)
        state['llm_responses'].append((cat,result.content))

    return state
