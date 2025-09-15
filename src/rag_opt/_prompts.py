RAG_DATASET_GENERATION_PROMPT = """Based on the following context, generate a {difficulty_instruction} level question and provide a comprehensive answer.

Context:
{contexts}

Generate:
1. A clear, specific question
2. A detailed answer based solely on the provided context

Format your response as JSON:
{{
    "question": "your question here",
    "answer": "your detailed answer here"
}}
- ONLY generate one question and one answer.
Ensure the question is answerable from the given context and the answer is accurate and complete."""


# Contexts 
CONTEXT_PRECISION_PROMPT = """
Given question, answer and context verify if the context was useful in arriving at the given answer. RETURN "1" if useful and "0" if not 

Question:\n {question}\n
=========================
Answer:\n {answer}\n
=========================
Context:\n {context}\n
=========================

- DONT SAY ANY THING IN YOUR ANSWER ONLY INTEGER OUTPUT 1 OR 0 
"""