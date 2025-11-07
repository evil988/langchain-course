"""Exemplo de prompt incentivando cadeias de raciocínio (chain of thought)."""

import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

load_dotenv()

# Chave de API
groq_api_key = os.environ["GROQ_API_KEY"]

# Inicializar o LLM
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

# Template que incentiva o modelo a mostrar o raciocínio antes da resposta
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "Você é um tutor de matemática paciente. Resolva o problema a seguir "
                "exibindo um raciocínio estruturado. Sempre responda no formato:\n"
                "Raciocínio: <passos detalhados>\n"
                "Resposta final: <resultado em uma frase curta>"
            ),
        ),
        ("human", "{pergunta}"),
    ]
)

# Parser simples para capturar a saída textual
output_parser = StrOutputParser()

# Montar a Chain usando LCEL
chain = prompt_template | llm | output_parser

print("--- Exemplo Chain of Thought ---")
pergunta_exemplo = (
    "Se João tem 3 cadernos e compra mais 5, mas dá 2 para Maria, "
    "quantos cadernos ficam com João?"
)
resposta = chain.invoke({"pergunta": pergunta_exemplo})
print(f"Pergunta: {pergunta_exemplo}")
print(f"RESPOSTA:\n{resposta}\n")

# uv run python ex3_chain_of_thought.py