# Prompts Exemplares

Esta branch contém prompts demonstrados no Codando apresentado em 14/11/2025.

## Exercícios

### ex1_prompt.py
Este exercício demonstra a engenharia de prompt básica, utilizando uma estrutura clara para guiar a resposta do modelo. Os elementos básicos para um prompt eficiente, como visto no código, são:

- **[🎭 PAPEL]**: Define a persona que o modelo deve assumir.
- **[📚 CONTEXTO]**: Fornece o plano de fundo da solicitação, explicando o objetivo do usuário.
- **[🎯 TAREFA]**: Descreve a ação específica que o modelo deve realizar.
- **[📝 FORMATO]**: Especifica o formato exato da saída.

### ex2_few_shot.py
Este exercício demonstra o conceito de "Few-Shot Prompting". Nesta técnica, o modelo de linguagem recebe vários exemplos de pares entrada-saída antes de ser solicitado a gerar uma resposta para uma nova entrada. Isso ajuda o modelo a entender o padrão desejado e o formato da saída, mesmo sem instruções explícitas detalhadas.

### ex3_chain_of_thought.py
Este exercício demonstra o "Chain of Thought" (Cadeia de Pensamento). Esta técnica de prompt incentiva o modelo de linguagem a exibir seu processo de raciocínio passo a passo antes de fornecer a resposta final. Ao instruir o modelo a detalhar seus "Raciocínio" e, em seguida, a "Resposta final", é possível resolver problemas complexos de forma mais eficaz, tornando o processo de pensamento do modelo mais transparente e, frequentemente, levando a resultados mais precisos. Isso é particularmente útil para tarefas que exigem múltiplas etapas lógicas ou cálculos.

### ex4_1_cutoff.py
Este exercício ilustra as limitações de "knowledge cutoff" (corte de conhecimento) que os modelos de linguagem podem apresentar. Modelos de IA são treinados com dados até uma determinada data, o que significa que eles não possuem informações sobre eventos muito recentes ou dados que mudam rapidamente. O exemplo demonstra como uma pergunta sobre fatos atuais ou que exigem cálculos baseados em informações dinâmicas pode resultar em respostas desatualizadas ou imprecisas quando o LLM é invocado diretamente, sem acesso a ferramentas externas para buscar informações em tempo real.

### ex4_reAct.py
Este exercício demonstra o padrão ReAct (Reasoning and Acting). ReAct é uma técnica que permite aos modelos de linguagem (LLMs) raciocinar sobre uma tarefa ("Thought") e, em seguida, executar ações ("Action") usando ferramentas externas para coletar informações ou realizar operações. Isso capacita o LLM a superar suas limitações de conhecimento (knowledge cutoff) e a interagir com o mundo real. No exemplo, o agente utiliza uma ferramenta de busca na web para encontrar informações e uma ferramenta personalizada para calcular idades, combinando raciocínio e ação para responder a perguntas complexas.