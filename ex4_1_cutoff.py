import os
from datetime import date, datetime

from dotenv import load_dotenv
from langchain import hub  # Para carregar o prompt ReAct oficial

# Importações para Agentes/ReAct
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langchain_groq import ChatGroq

load_dotenv()

# Inicializar o LLM
# https://console.groq.com/docs/model/llama-3.1-8b-instant
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

print("\n--- Exemplo 1: Fato desconhecido + Cálculo ---")

pergunta = "Quem é o CEO da Tesla e qual a idade dele?"

# --- APENAS CHAME O LLM DIRETAMENTE ---
resposta = llm.invoke(pergunta)

print(f"\nPERGUNTA:\n{pergunta}\n")
print(f"\nRESPOSTA:\n{resposta.content}\n")

# uv run python ex4_1_cutoff.py