"""
Montagem básica de contexto com orçamento de tokens.

A ideia: o contexto tem um limite. Reservamos espaço para o que é essencial
(system prompt, pergunta, documentos relevantes) e preenchemos o resto com o
histórico mais recente, descartando o que não cabe.

Sem dependências externas.
"""


def estimate_tokens(text):
    # estimativa grosseira: ~4 caracteres por token
    return max(1, len(text) // 4)


def words(text):
    # minúsculas, sem pontuação, ignorando palavras curtas (o, de, por...)
    cleaned = "".join(c if c.isalnum() or c.isspace() else " " for c in text.lower())
    return {w for w in cleaned.split() if len(w) > 3}


def retrieve(query, docs, k=2):
    """Ranqueia documentos por sobreposição de palavras com a pergunta."""
    q_words = words(query)
    scored = []
    for doc in docs:
        overlap = len(q_words & words(doc))
        scored.append((overlap, doc))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [doc for score, doc in scored[:k] if score > 0]


def build_context(system, history, docs, query, budget=120):
    used = estimate_tokens(system) + estimate_tokens(query)

    # 1) documentos relevantes entram primeiro, enquanto couberem
    selected_docs = []
    for doc in retrieve(query, docs):
        cost = estimate_tokens(doc)
        if used + cost <= budget:
            selected_docs.append(doc)
            used += cost

    # 2) o histórico preenche o resto, das mensagens mais novas para as antigas
    kept = []
    for message in reversed(history):
        cost = estimate_tokens(message)
        if used + cost > budget:
            break
        kept.append(message)
        used += cost
    kept.reverse()

    return {
        "system": system,
        "docs": selected_docs,
        "history": kept,
        "query": query,
        "tokens": used,
        "dropped": len(history) - len(kept),
    }


if __name__ == "__main__":
    system = "Você é um assistente que responde de forma curta e direta."

    docs = [
        "O plano básico custa 49 reais por mês e inclui um projeto.",
        "O suporte funciona de segunda a sexta, das 9h às 18h.",
        "A política de reembolso permite cancelar em até 7 dias.",
    ]

    history = [
        "usuário: oi, tudo bem?",
        "assistente: tudo certo! Como posso ajudar?",
        "usuário: estou pensando em assinar o serviço.",
        "assistente: ótimo, posso tirar dúvidas sobre planos e prazos.",
        "usuário: vocês têm suporte por telefone?",
        "assistente: o suporte é feito por chat e e-mail.",
    ]

    query = "quanto custa o plano básico por mês?"

    ctx = build_context(system, history, docs, query, budget=80)

    print("Documentos escolhidos:")
    for doc in ctx["docs"]:
        print(" -", doc)

    print("\nHistórico mantido:")
    for message in ctx["history"]:
        print(" -", message)

    print(f"\nMensagens descartadas: {ctx['dropped']}")
    print(f"Tokens estimados: {ctx['tokens']} de 80")
