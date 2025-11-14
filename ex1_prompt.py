"""Exemplo executável de prompt."""

import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

# Inicializar o LLM
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

# Criar o Template do Prompt
prompt = ChatPromptTemplate.from_template(
    """
    [🎭 PAPEL]
    Você é um especialista em comunicação clara e concisa. Seu superpoder é 
    destilar ideias complexas em linguagem simples e acessível.

    [📚 CONTEXTO]
    O usuário precisa de um resumo "TL;DR" (Too Long; Didn't Read) sobre 
    um {topico} específico. O objetivo é a compreensão imediata.

    [🎯 TAREFA]
    1. Gere um resumo focado na clareza sobre o {topico}.
    2. Evite jargões, siglas e linguagem técnica complexa.
    3. O resumo deve capturar a essência central do tópico.

    [📝 FORMATO]
    - O resumo DEVE ter exatamente 1 (uma) frase de 10 palavras.
    - A sua resposta deve conter *apenas* a frase do resumo.
    - Não inclua saudações, preâmbulos ou qualquer texto extra.
    """
)

# Criar o Parser de Saída
output_parser = StrOutputParser()

# Montar a Chain (Usando LCEL)
chain = prompt | llm | output_parser

# Tópico de exemplo
topico_exemplo = """
A inteligência artificial generativa (GenAI) é um tipo de IA que cria conteúdo novo e original – texto, imagens, música, vídeos, código 
– a partir de prompts, aprendendo padrões em bilhões de dados via redes neurais (especialmente transformers) durante o treinamento, e 
depois gerando saídas criativas na fase de inferência, usando técnicas como amostragem aleatória para variar respostas; exemplos famosos 
incluem GPT-4 (OpenAI), mestre em linguagem e código, Gemini (Google), multimodal para texto/imagem/áudio/vídeo, DALL-E 3 (imagens de texto), 
Stable Diffusion (open-source), Suno AI (músicas completas) e GitHub Copilot (programação); aplicações vão de marketing automático, educação 
personalizada e saúde a entretenimento, mas traz desafios como plágio implícito, desinformação, vieses e impacto em empregos criativos – resumindo, 
é um co-criador incansável que replica e reimagina a criatividade humana, limitado só pelo prompt e pela ética.
"""

print("--- Exemplo Prompt ---")
resposta_1 = chain.invoke({
    "topico": topico_exemplo
})
print(f"RESPOSTA 1:\n{resposta_1}\n\n")

# uv run python ex1_prompt.py