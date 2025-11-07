""" Exemplo prompt "Few-Shot" """
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

# Chave de API
groq_api_key = os.environ["GROQ_API_KEY"]

# Inicializar o LLM
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

# Exemplos usados pelo template few-shot
examples = [
    {
        "texto": "Amei o novo restaurante, a comida estava divina!",
        "sentimento": "Positivo",
    },
    {
        "texto": "O atendimento demorou muito e o pedido veio errado.",
        "sentimento": "Negativo",
    },
    {
        "texto": "A reunião foi marcada para as 15h da próxima sexta-feira.",
        "sentimento": "Neutro",
    },
]

# Prompt que estrutura cada exemplo few-shot
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", 'Texto: "{texto}"'),
        ("ai", "{sentimento}"),
    ]
)

# Componente few-shot da LangChain que injeta os exemplos acima
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

# Prompt completo com instruções e tarefas
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            'Sua tarefa é classificar o sentimento de um texto. Responda APENAS com "Positivo", "Negativo" ou "Neutro".',
        ),
        few_shot_prompt,
        ("human", 'Texto: "{texto_usuario}"'),
    ]
)

# Criar o Parser de Saída
output_parser = StrOutputParser()

# Montar a Chain (Usando LCEL)
chain = prompt_template | llm | output_parser

# Teste 1: Um texto claramente positivo
print("--- Exemplo Few-Shot: Classificação de Sentimento ---")
texto_teste_1 = "Que filme espetacular! Havia muito tempo que não via algo assim."
resposta_1 = chain.invoke({
    "texto_usuario": texto_teste_1
})
print(f"Texto: {texto_teste_1}")
print(f"RESPOSTA: {resposta_1}\n")

# Teste 2: Um texto claramente negativo
texto_teste_2 = "Estou muito decepcionado com a qualidade do produto, quebrou no primeiro dia."
resposta_2 = chain.invoke({
    "texto_usuario": texto_teste_2
})
print(f"Texto: {texto_teste_2}")
print(f"RESPOSTA: {resposta_2}\n")

# Teste 3: Um texto que deve ser neutro
texto_teste_3 = "O documento será enviado por e-mail amanhã."
resposta_3 = chain.invoke({
    "texto_usuario": texto_teste_3
})
print(f"Texto: {texto_teste_3}")
print(f"RESPOSTA: {resposta_3}\n")
