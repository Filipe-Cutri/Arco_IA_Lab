# Prompting

Prompt é o texto que você manda para o modelo, e prompting é a prática de escrever esse texto de um jeito que aumente as chances de receber o que você realmente quer. Como um LLM só tem o que está no prompt para trabalhar, a diferença entre uma resposta genérica e uma boa resposta quase sempre está em quanta informação útil você deu.

## Ideias básicas

**Seja específico.** "Resume esse texto" funciona, mas "resume esse texto em três frases para alguém que nunca ouviu falar do assunto" funciona melhor. Diga o objetivo, quem vai ler e o que importa.

**Dê contexto.** O modelo não sabe quem você é, qual é o seu projeto nem o que já tentou. Cole o trecho relevante em vez de descrevê-lo de forma vaga.

**Defina o formato da saída.** Se você precisa de JSON, uma tabela, uma lista curta ou um parágrafo, peça explicitamente. Isso importa ainda mais quando a resposta vai ser lida por um programa.

**Itere.** Quase nunca o primeiro prompt é o melhor. Olhe o que saiu errado, ajuste a instrução e tente de novo.

## Técnicas comuns

No **zero-shot**, você só descreve a tarefa e confia que o modelo entende. No **few-shot**, você inclui alguns exemplos de entrada e saída no próprio prompt, e o modelo copia o padrão. Essa capacidade de aprender pelo contexto ganhou destaque no artigo do GPT-3, *Language Models are Few-Shot Learners* (Brown et al., 2020).

No **chain of thought**, você pede para o modelo raciocinar passo a passo antes de dar a resposta final. Em tarefas de lógica e matemática isso costuma melhorar bastante o resultado, como mostrado em *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models* (Wei et al., 2022).

Também vale separar as instruções fixas, que vão no **system prompt** (papel, tom, regras), do pedido do momento, que vai na mensagem do usuário.

## Código

O arquivo [`exemplos.py`](./exemplos.py) mostra as três ideias com a API da Anthropic: zero-shot, few-shot e system prompt com formato de saída definido. Para rodar:

```bash
pip install anthropic
export ANTHROPIC_API_KEY="sua-chave"
python exemplos.py
```

## Para ler depois

- Brown et al., *Language Models are Few-Shot Learners*, 2020: https://arxiv.org/abs/2005.14165
- Wei et al., *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*, 2022: https://arxiv.org/abs/2201.11903
- Guia de prompt engineering da Anthropic: https://docs.claude.com/en/docs/build-with-claude/prompt-engineering/overview
