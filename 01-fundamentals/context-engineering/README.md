# Context Engineering

Todo LLM trabalha com uma janela de contexto, que é o limite de texto que ele consegue enxergar de uma vez. Context engineering é a prática de decidir o que entra nessa janela a cada chamada: instruções, histórico da conversa, documentos recuperados, resultados de ferramentas, exemplos. Se o prompting é saber pedir, context engineering é saber montar tudo o que o modelo precisa para responder bem.

Isso fica mais importante conforme o sistema cresce. Num chat simples, o contexto é só a conversa. Num agente que usa ferramentas, busca documentos e roda por muitos passos, o contexto vira um recurso que precisa ser administrado, porque ele enche rápido.

## Por que não dá para colocar tudo

A janela é limitada e, mesmo quando cabe muita coisa, mais texto não significa melhor resposta. Informação irrelevante distrai o modelo, aumenta o custo e a latência, e há estudos mostrando que modelos tendem a usar pior o que está no meio de um contexto longo (*Lost in the Middle*, Liu et al., 2023). A ideia é colocar o menor conjunto de informação que ainda dê ao modelo o que ele precisa.

## O que costuma entrar no contexto

As instruções fixas (system prompt) vêm primeiro, seguidas do histórico recente da conversa. Depois entram os documentos relevantes para a pergunta atual, normalmente buscados sob demanda em vez de colados de antemão. É o que a técnica de RAG faz, apresentada em *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (Lewis et al., 2020). Por fim, vem a pergunta do usuário e, em agentes, os resultados das ferramentas chamadas até ali.

## Estratégias comuns

Quando o histórico cresce, dá para cortar as mensagens mais antigas ou resumi-las. Para informação externa, vale buscar só os trechos mais relevantes e não o documento inteiro. Em tarefas longas, anotar o progresso fora do contexto (num arquivo, por exemplo) e carregar de volta só quando necessário evita que o modelo perca o fio. Em todos os casos, o princípio é o mesmo: cada token no contexto precisa ter uma razão para estar lá.

## Código

O arquivo [`contexto.py`](./contexto.py) monta um contexto dentro de um orçamento de tokens, usando só a biblioteca padrão do Python: escolhe os documentos mais relevantes para a pergunta e preenche o resto com as mensagens mais recentes do histórico. Para rodar:

```bash
python contexto.py
```

## Para ler depois

- Anthropic, *Effective context engineering for AI agents*: https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents
- Liu et al., *Lost in the Middle: How Language Models Use Long Contexts*, 2023: https://arxiv.org/abs/2307.03172
- Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*, 2020: https://arxiv.org/abs/2005.11401
