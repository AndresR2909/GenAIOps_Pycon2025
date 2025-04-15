# app/main_interface.py
from __future__ import annotations

import json
import os
import sys

import mlflow
import pandas as pd
import streamlit as st

st.set_page_config(page_title='📚 Chatbot GenAI + Métricas', layout='wide')


# fmt: off
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.rag_pipeline import load_vectorstore_from_disk, build_chain
import altair as alt

# fmt: on


def load_text_artifact(run_id, artifact_path):
    """Funcion para cargar artefactos txt de razonamiento criterios de evaluacion"""
    # Descargamos el contenido del artefacto
    artifact_content = client.download_artifacts(run_id, artifact_path)

    # Leemos el contenido del archivo
    with open(artifact_content) as file:
        artifact_text = file.read()

    return artifact_text


# Carga la base vectorial y la cadena de consulta
vectordb = load_vectorstore_from_disk()
chain = build_chain(vectordb)

# Barra lateral de opciones
modo = st.sidebar.radio(
    'Selecciona una vista:', [
        '🤖 Chatbot',
        '📊 Métricas', '📊 Artefactos Razonamiento',
    ],
)

###################################################
# Sección de Chatbot
###################################################
if modo == '🤖 Chatbot':
    st.title('🤖 Asistente de Recursos Humanos')

    pregunta = st.text_input('¿Qué deseas consultar?')

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    if pregunta:
        with st.spinner('Consultando documentos...'):
            result = chain.invoke(
                {'question': pregunta, 'chat_history': st.session_state.chat_history},
            )
            st.session_state.chat_history.append((pregunta, result['answer']))

    if st.session_state.chat_history:
        for q, a in reversed(st.session_state.chat_history):
            st.markdown(f"**👤 Usuario:** {q}")
            st.markdown(f"**🤖 Bot:** {a}")
            st.markdown('---')

###################################################
# Sección de Métricas
###################################################
elif modo == '📊 Métricas':
    st.title('📈 Resultados de Evaluación')

    client = mlflow.tracking.MlflowClient()
    # Cargamos experimentos que comiencen con "eval_"
    experiments = [
        exp for exp in client.search_experiments() if exp.name.startswith('eval_')
    ]

    if not experiments:
        st.warning('No se encontraron experimentos de evaluación.')
        st.stop()

    exp_names = [exp.name for exp in experiments]
    selected_exp = st.selectbox('Selecciona un experimento:', exp_names)

    experiment = client.get_experiment_by_name(selected_exp)
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id], order_by=[
            'start_time DESC',
        ],
    )

    if not runs:
        st.warning('No hay ejecuciones registradas.')
        st.stop()

    # Recolectamos datos de cada run
    data = []
    for run in runs:
        params = run.data.params
        metrics = run.data.metrics
        artifacts = client.list_artifacts(run.info.run_id)
        list_artifacts = [artifact for artifact in artifacts]
        dict_metrics = {
            'Run ID': run.info.run_id,
            'Pregunta': params.get('question'),
            'Prompt': params.get('prompt_version'),
            'Chunk Size': int(params.get('chunk_size', 0)),
            # Métricas de evaluación
            'correctness_score': metrics.get('correctness_criteria_eval_score', None),
            'toxicity_score': metrics.get('toxicity_criteria_eval_score', None),
            'relevance_score': metrics.get('relevance_criteria_eval_score', None),
            'coherence_score': metrics.get('coherence_criteria_eval_score', None),
            'harmfulness_score': metrics.get('harmfulness_criteria_eval_score', None),
            'hallucination_score': metrics.get(
                'hallucination_criteria_eval_score', None,
            ),
            'correcto (LC)': metrics.get('lc_is_correct', None),
        }
        data.append(dict_metrics)

    # Creamos un dataframe con todos los datos
    df = pd.DataFrame(data)
    drop_columns = [
        'Run ID',
    ]
    st.dataframe(df.drop(columns=drop_columns))

    # Filtrar y agrupar dataset por Chunk Size y Prompt, sacar promedio del resto de columnas
    df_grouped = (
        df.drop(columns=['Run ID', 'Pregunta', 'correcto (LC)'])
        .groupby(['Chunk Size', 'Prompt'])
        .mean()
        .reset_index()
    )

    # Selección de criterios a mostrar en un gráfico
    st.subheader('criterios de evaluación (scores) x Chunk Size y  Prompt')

    st.dataframe(df_grouped)

    # Selección de criterios a mostrar en un gráfico
    st.subheader('Comparar criterios de evaluación (scores)')

    # Posibles métricas disponibles
    metric_choices = [
        'correctness_score',
        'toxicity_score',
        'relevance_score',
        'coherence_score',
        'harmfulness_score',
        'hallucination_score',
    ]
    selected_metrics = st.multiselect(
        'Selecciona los criterios que deseas comparar',
        metric_choices,
        # Por defecto mostrar correctness vs hallucination
        default=['correctness_score', 'hallucination_score'],
    )

    if selected_metrics:
        # Agrupamos por Prompt y Chunk Size para mostrar promedios
        grouped = (
            df.groupby(['Prompt', 'Chunk Size'])
            .agg({metric: 'mean' for metric in selected_metrics})
            .reset_index()
        )

        grouped['config'] = (
            grouped['Prompt'] + ' | ' + grouped['Chunk Size'].astype(str)
        )

        # Mostramos todas las métricas seleccionadas en un único gráfico de barras agrupadas
        if selected_metrics:
            st.subheader(
                'Promedio de métricas seleccionadas por configuración',
            )

            # Convertimos los datos a formato largo para Altair
            chart_data = grouped.melt(
                id_vars=['config'],
                value_vars=selected_metrics,
                var_name='Métrica',
                value_name='Valor',
            )

            # Creamos un gráfico de barras agrupadas
            chart = alt.Chart(chart_data).mark_bar().encode(
                # Sin título en el eje X
                x=alt.X('config:N', axis=alt.Axis(labelAngle=-45, title=None)),
                y=alt.Y('Valor:Q', axis=alt.Axis(title='valor metrica')),
                color='Métrica:N',
                # Sin título en las columnas
                column=alt.Column('Métrica:N', spacing=10, title=None),
                tooltip=['Métrica', 'Valor'],
            ).properties(
                # Ajusta el ancho de las barras para que sean más angostas
                width=alt.Step(100),
            )

            st.altair_chart(chart, use_container_width=False)

    # Seguimos mostrando adicionalmente la métrica de "Correcto (LC)"
    st.subheader("Comparación de 'Correcto (LC)'")
    grouped_lc = (
        df.groupby(['Prompt', 'Chunk Size'])
        .agg({'correcto (LC)': 'mean'})
        .reset_index()
    )
    grouped_lc['config'] = (
        grouped_lc['Prompt'] + ' | ' + grouped_lc['Chunk Size'].astype(str)
    )
    lc_chart_data = grouped_lc.set_index('config')['correcto (LC)']
    st.bar_chart(lc_chart_data)

###################################################
# Sección de Artefactos
###################################################
elif modo == '📊 Artefactos Razonamiento':
    st.title('📈 Razonamientos de evaluacion de preguntas por ejecucion')

    # Posibles métricas disponibles
    metric_choices = [
        'correctness_score',
        'toxicity_score',
        'relevance_score',
        'coherence_score',
        'harmfulness_score',
        'hallucination_score',
    ]

    client = mlflow.tracking.MlflowClient()
    # Cargamos experimentos que comiencen con "eval_"
    experiments = [
        exp for exp in client.search_experiments() if exp.name.startswith('eval_')
    ]

    if not experiments:
        st.warning('No se encontraron experimentos de evaluación.')
        st.stop()

    exp_names = [exp.name for exp in experiments]
    selected_exp = st.selectbox('Selecciona un experimento:', exp_names)

    experiment = client.get_experiment_by_name(selected_exp)
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id], order_by=[
            'start_time DESC',
        ],
    )

    if not runs:
        st.warning('No hay ejecuciones registradas.')
        st.stop()

    # Recolectamos datos de cada run
    data = []
    for run in runs:
        params = run.data.params
        metrics = run.data.metrics
        artifacts = client.list_artifacts(run.info.run_id)
        list_artifacts = [artifact for artifact in artifacts]
        dict_metrics = {
            'Run ID': run.info.run_id,
            'Pregunta': params.get('question'),
            'Prompt': params.get('prompt_version'),
            'Chunk Size': int(params.get('chunk_size', 0)),
            # Métricas de evaluación
            'correctness_score': metrics.get('correctness_criteria_eval_score', None),
            'toxicity_score': metrics.get('toxicity_criteria_eval_score', None),
            'relevance_score': metrics.get('relevance_criteria_eval_score', None),
            'coherence_score': metrics.get('coherence_criteria_eval_score', None),
            'harmfulness_score': metrics.get('harmfulness_criteria_eval_score', None),
            'hallucination_score': metrics.get(
                'hallucination_criteria_eval_score', None,
            ),
            'correcto (LC)': metrics.get('lc_is_correct', None),
        }
        text_artifacts = {}
        if len(list_artifacts) > 0:
            for artifact in list_artifacts:
                path = artifact.path
                name = path.split('.txt')[0]
                text_artifacts[name] = load_text_artifact(
                    run.info.run_id, path,
                )
        else:
            text_artifacts = {
                'coherence_criteria_eval_reasoning': None,
                'correctness_criteria_eval_reasoning': None,
                'hallucination_criteria_eval_reasoning': None,
                'harmfulness_criteria_eval_reasoning': None,
                'relevance_criteria_eval_reasoning': None,
                'toxicity_criteria_eval_reasoning': None,
            }

        dict_metrics.update(text_artifacts)
        data.append(dict_metrics)

    # Creamos un dataframe con todos los datos
    df = pd.DataFrame(data)

    # Mostrar razonamientos
    st.subheader('Razonamientos de evaluacion de criterios')

    # Selección de Run ID para filtrar razonamientos
    run_ids = df['Run ID'].unique()
    selected_run_id = st.selectbox(
        'Selecciona un Run ID para ver los razonamientos:', run_ids,
    )
    preguntas = df['Pregunta'].unique()

    selected_pregunta = st.selectbox(
        'Selecciona una Pregunta para ver los razonamientos:', preguntas,
    )

    # Filtramos el DataFrame por el Run ID y pregunta seleccionado
    filtered_df = df[(df['Run ID'] == selected_run_id) &
                     (df['Pregunta'] == selected_pregunta)]

    # Mostramos los razonamientos del Run ID y Pregunta seleccionados

    st.markdown(f"**Prompt:** {filtered_df['Prompt'].values}")
    st.markdown(f"**Chunk Size:** {filtered_df['Chunk Size'].values}")
    # Mostramos los razonamientos
    st.markdown('**Razonamientos:**')
    st.markdown(
        f"- **Correctness**: {filtered_df['correctness_criteria_eval_reasoning'].values}",
    )
    st.markdown(
        f"- **Toxicity**: {filtered_df['toxicity_criteria_eval_reasoning'].values}",
    )
    st.markdown(
        f"- **Relevance**: {filtered_df['relevance_criteria_eval_reasoning'].values}",
    )
    st.markdown(
        f"- **Coherence**: {filtered_df['coherence_criteria_eval_reasoning'].values}",
    )
    st.markdown(
        f"- **Harmfulness**: {filtered_df['harmfulness_criteria_eval_reasoning'].values}",
    )
    st.markdown(
        f"- **Hallucination**: {filtered_df['hallucination_criteria_eval_reasoning'].values}",
    )
