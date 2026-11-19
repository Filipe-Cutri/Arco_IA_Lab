# LLMs

Um LLM (Large Language Model) é um modelo treinado com uma quantidade enorme de texto para fazer uma tarefa aparentemente simples: prever o próximo token. Um token é um pedaço de texto, que pode ser uma palavra, parte dela ou um símbolo. Para prever bem, o modelo precisa aprender gramática, fatos, estilo e até alguns padrões de raciocínio, e é daí que vem boa parte do comportamento que a gente vê nos assistentes de hoje.

## Como funciona, em linhas gerais

O texto é quebrado em tokens, cada token vira um vetor numérico (embedding), esses vetores passam por várias camadas de um Transformer e, no final, o modelo devolve uma probabilidade para cada token possível vir a seguir. Escolhe-se um, adiciona-se ao texto e o processo se repete.

## O paper que mudou tudo

Em 2017, pesquisadores do Google publicaram o artigo **"Attention Is All You Need"** (Vaswani et al.), que apresentou a arquitetura Transformer. Até então, o padrão eram redes recorrentes (RNNs e LSTMs), que liam o texto em sequência, uma palavra por vez, o que era lento e dificultava lembrar de coisas distantes no texto.

A ideia central do Transformer é a **atenção**: em vez de ler em fila, cada token olha para todos os outros da sequência e decide quais são mais relevantes para ele. Como isso pode ser calculado em paralelo, o treino escala muito melhor em GPUs. Praticamente todos os LLMs modernos partem dessa arquitetura.

A conta, na versão do paper (scaled dot-product attention), é:

```
Attention(Q, K, V) = softmax(Q·Kᵀ / √d_k) · V
```

Cada token gera três vetores: a **query** (Q), o que ele está procurando; a **key** (K), o que ele oferece para os outros; e o **value** (V), a informação que ele realmente passa adiante. O produto Q·Kᵀ mede a afinidade entre os tokens, o softmax transforma isso em pesos que somam 1, e o resultado é uma mistura dos values. A divisão por √d_k evita que os números fiquem grandes demais e deixem o softmax saturado.

## Código

O arquivo [`attention.py`](./attention.py) tem uma implementação mínima em NumPy, com uma máscara causal (cada token só enxerga os anteriores, como acontece nos modelos que geram texto). Para rodar:

```bash
pip install numpy
python attention.py
```

## Para ler depois

- Vaswani et al., *Attention Is All You Need*, 2017: https://arxiv.org/abs/1706.03762
