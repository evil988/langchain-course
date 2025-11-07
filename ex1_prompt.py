"""Exemplo executável de prompt."""

import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

# Chave de API
groq_api_key = os.environ["GROQ_API_KEY"]

# Inicializar o LLM
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

# Criar o Template do Prompt
prompt = ChatPromptTemplate.from_template(
    """
    [CONTEXTO]
    Você é um assistente de IA. Sua tarefa é resumir o {topico}.

    [INSTRUÇÃO IMPLÍCITA]
    Evite jargões e linguagem complexa, foque na clareza.

    [COMANDOS IMPERATIVOS]
    1. Gere um resumo do {topico}.
    2. O resumo DEVE ter exatamente 1 frase.
    3. A resposta final deve ser apenas o resumo, sem saudações (como "Claro!").
    """
)

# Criar o Parser de Saída
output_parser = StrOutputParser()

# Montar a Chain (Usando LCEL)
chain = prompt | llm | output_parser

# Tópico de exemplo
topico_exemplo = """
A inteligência artificial generativa (GenAI) é um tipo de IA capaz de criar 
conteúdo novo, como texto, imagens, música e código. Ela funciona 
aprendendo padrões de grandes conjuntos de dados (treinamento) e, em seguida, 
usando esse conhecimento para gerar novas saídas originais. Modelos famosos 
incluem o GPT-4 da OpenAI e o Gemini do Google.
"""

print("--- Exemplo Prompt ---")
resposta_1 = chain.invoke({
    "topico": topico_exemplo
})
print(f"RESPOSTA 1:\n{resposta_1}\n\n")

# uv run python ex1_prompt.py