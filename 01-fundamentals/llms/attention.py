"""
Scaled dot-product attention em NumPy.

Baseado em "Attention Is All You Need" (Vaswani et al., 2017):
    Attention(Q, K, V) = softmax(Q @ K.T / sqrt(d_k)) @ V
"""

import numpy as np


def softmax(x):
    # subtrai o máximo por estabilidade numérica
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)          # afinidade entre os tokens
    if mask is not None:
        scores = np.where(mask, scores, -1e9)  # posições bloqueadas
    weights = softmax(scores)                # cada linha soma 1
    return weights @ V, weights


if __name__ == "__main__":
    rng = np.random.default_rng(42)

    tokens = ["eu", "gosto", "de", "café"]
    d_model = 8

    # embeddings aleatórios, só para demonstrar o mecanismo
    X = rng.normal(size=(len(tokens), d_model))

    # matrizes de projeção (num modelo real, são aprendidas no treino)
    Wq = rng.normal(size=(d_model, d_model))
    Wk = rng.normal(size=(d_model, d_model))
    Wv = rng.normal(size=(d_model, d_model))

    Q, K, V = X @ Wq, X @ Wk, X @ Wv

    # máscara causal: cada token só enxerga ele mesmo e os anteriores
    causal_mask = np.tril(np.ones((len(tokens), len(tokens)), dtype=bool))

    out, weights = scaled_dot_product_attention(Q, K, V, causal_mask)

    print("Pesos de atenção (linha = token que olha, coluna = token olhado):\n")
    print(f"{'':>8}" + "".join(f"{t:>8}" for t in tokens))
    for t, row in zip(tokens, weights):
        print(f"{t:>8}" + "".join(f"{w:8.2f}" for w in row))

    print("\nShape da saída:", out.shape)
