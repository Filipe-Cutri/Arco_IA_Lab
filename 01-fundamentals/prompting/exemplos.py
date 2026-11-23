"""
Exemplos básicos de prompting com a API da Anthropic.

Requer: pip install anthropic
E a variável de ambiente ANTHROPIC_API_KEY configurada.
"""

import anthropic

client = anthropic.Anthropic()  # lê ANTHROPIC_API_KEY do ambiente
MODEL = "claude-sonnet-5-5"


def ask(prompt, system=None):
    kwargs = {"system": system} if system else {}
    response = client.messages.create(
        model=MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
        **kwargs,
    )
    return response.content[0].text


review = "A entrega demorou, mas o produto é ótimo e superou o que eu esperava."

# 1) Zero-shot: só descreve a tarefa
zero_shot = f"Classifique o sentimento desta avaliação: {review}"

# 2) Few-shot: mostra exemplos para o modelo copiar o padrão
few_shot = f"""Classifique o sentimento de cada avaliação como positivo, negativo ou misto.

Avaliação: Chegou quebrado e ninguém respondeu meu e-mail.
Sentimento: negativo

Avaliação: Perfeito, exatamente como na foto.
Sentimento: positivo

Avaliação: {review}
Sentimento:"""

# 3) System prompt: papel e formato de saída definidos antes do pedido
system = (
    "Você é um classificador de avaliações de clientes. "
    'Responda apenas com JSON no formato {"sentimento": "...", "motivo": "..."}.'
)

if __name__ == "__main__":
    print("--- Zero-shot ---")
    print(ask(zero_shot))

    print("\n--- Few-shot ---")
    print(ask(few_shot))

    print("\n--- System prompt + JSON ---")
    print(ask(review, system=system))
