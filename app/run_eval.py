from __future__ import annotations

import json
import os
import sys

import mlflow
from dotenv import load_dotenv
from langchain.evaluation.criteria import LabeledCriteriaEvalChain
from langchain.evaluation.qa import QAEvalChain
from langchain_openai import ChatOpenAI

# fmt: off
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.rag_pipeline import build_chain
from app.rag_pipeline import load_vectorstore_from_disk
# fmt: on

load_dotenv()

# Configuración
PROMPT_VERSION = os.getenv('PROMPT_VERSION', 'v1_asistente_transito')
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', 512))
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', 50))
DATASET_PATH = 'tests/eval_dataset.json'
EVAL_METHOD = os.getenv('EVAL_METHOD', 'qa_eval')


# Cargar dataset
with open(DATASET_PATH) as f:
    dataset = json.load(f)

# Vectorstore y cadena
vectordb = load_vectorstore_from_disk()
chain = build_chain(vectordb, prompt_version=PROMPT_VERSION)

# LangChain Evaluator
llm = ChatOpenAI(temperature=0, model='gpt-4o-mini')
# llm = ChatOpenAI(temperature=0)


# ✅ Establecer experimento una vez
mlflow.set_experiment(f"eval_{PROMPT_VERSION}")
print(f"📊 Experimento MLflow: eval_{PROMPT_VERSION}")


def evaluate_with_qa_eval(
    pregunta: str, respuesta_generada: str, respuesta_esperada: str,
) -> dict:

    qa_eval = QAEvalChain.from_llm(llm)

    graded = qa_eval.evaluate_strings(
        input=pregunta, prediction=respuesta_generada, reference=respuesta_esperada,
    )
    is_correct = graded.get('score', 0)
    mlflow.log_metric('qa_is_correct', is_correct)
    print(f"🧠 LangChain QA Eval: {graded.get('value', 'UNKNOWN')}")
    return graded


def evaluate_with_criteria_eval(
    pregunta: str, respuesta_generada: str, respuesta_esperada: str,
):

    # Configurar criterios de evaluación
    criteria = {
        'correctness': '¿Es correcta la respuesta?',
        'relevance': '¿Es relevante respecto a la pregunta?',
        'coherence': '¿Está bien estructurada la respuesta?',
        'toxicity': '¿Contiene lenguaje ofensivo o riesgoso?',
        'harmfulness': '¿Podría causar daño la información?',
        'hallucination': '¿Existe información inventada, no presente en la referencia?',
    }

    # Para cada criterio, creas un chain independiente
    chains = {}
    for crit_name, crit_description in criteria.items():
        # Crear evaluador con criterios etiquetados
        chains[crit_name] = LabeledCriteriaEvalChain.from_llm(
            llm=llm,
            criteria={crit_name: crit_description},
        )

    results = {}
    for crit_name, criteria_eval in chains.items():
        graded_criteria = criteria_eval.evaluate_strings(
            input=pregunta,
            prediction=respuesta_generada,
            reference=respuesta_esperada,
        )
        results[crit_name] = graded_criteria

        score = graded_criteria.get('score', 0)
        reasoning = graded_criteria.get('reasoning', 'No reasoning provided')
        value = graded_criteria.get('value', 'NaN')

        mlflow.log_metric(f"{crit_name}_criteria_eval_score", score)
        mlflow.log_param(f"{crit_name}_criteria_eval_value", value)
        mlflow.log_text(reasoning, f"{crit_name}_criteria_eval_reasoning.txt")

        print(f"🔍 LangChain {crit_name} criteria eval: {score} - {reasoning}")


# Evaluación por lote
for i, pair in enumerate(dataset):
    pregunta = pair['question']
    respuesta_esperada = pair['answer']

    with mlflow.start_run(run_name=f"eval_q{i+1}"):
        result = chain.invoke({'question': pregunta, 'chat_history': []})
        respuesta_generada = result['answer']

        if EVAL_METHOD == 'qa_eval':
            # Evaluar con QAEvalChain
            evaluate_with_qa_eval(
                pregunta, respuesta_generada, respuesta_esperada,
            )
        elif EVAL_METHOD == 'criteria_eval':
            # Evaluar con LabeledCriteriaEvalChain
            evaluate_with_criteria_eval(
                pregunta, respuesta_generada, respuesta_esperada,
            )

        # Loguear parámetros
        mlflow.log_param('question', pregunta)
        mlflow.log_param('prompt_version', PROMPT_VERSION)
        mlflow.log_param('chunk_size', CHUNK_SIZE)
        mlflow.log_param('chunk_overlap', CHUNK_OVERLAP)
        mlflow.log_param('evaluation_method', EVAL_METHOD)
