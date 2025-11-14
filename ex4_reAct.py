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

# Plano A
groq_api_key = os.environ["GROQ_API_KEY"]

# Plano B
# groq_api_key = os.environ["GROQ_API_KEY_2"]

# Inicializar o LLM
# https://console.groq.com/docs/model/llama-3.3-70b-versatile
llm = ChatGroq(groq_api_key=groq_api_key, model="llama-3.3-70b-versatile", temperature=0)

@tool
def calcular_idade(data_nascimento: str) -> str:
    """
    Calcula a idade exata a partir de uma data de nascimento.
    O formato da data DEVE ser estritamente 'YYYY-MM-DD'.
    """
    
    # Converte a string para data.
    nascimento = datetime.strptime(data_nascimento.strip(), "%Y-%m-%d").date()

    # Lógica de cálculo da idade
    hoje = date.today()
    idade = hoje.year - nascimento.year - (
        (hoje.month, hoje.day) < (nascimento.month, nascimento.day)
    )

    # Retorna o resultado
    return f"Idade: {idade} anos."


# Definir as Ferramentas
tools = [DuckDuckGoSearchRun(name="web_search"), calcular_idade]

# O Prompt ReAct
prompt = hub.pull("hwchase17/react")

# Combina o LLM, as ferramentas e o prompt.
agent = create_react_agent(llm, tools, prompt)

# Criar o Executor do Agente
# verbose=True é para vermos o ReAct funcionando.
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,

    # Limita o número total de passos (Pensamento + Ação)
    max_iterations=10, 
    
    # Informa ao agente se ele usar uma ferramenta da forma errada
    handle_parsing_errors=True
)

print("\n--- Exemplo: Fato desconhecido + Cálculo ---")

pergunta_2 = "Quem é o atual CEO da Tesla e quantos anos ele tem?"
resposta_2 = agent_executor.invoke({
    "input": pergunta_2
})

print(f"\nRESPOSTA FINAL:\n{resposta_2['output']}\n")

# uv run python ex4_reAct.py