
import streamlit as st
from pathlib import Path
from eda_agent import CSVAgent

st.set_page_config(page_title="EDA Agent (CSV/NF-e)", layout="wide")

st.title("🧠 EDA Agent — Pergunte sobre seu CSV (com foco em NF-e)")
st.write("""
Carregue um arquivo CSV (qualquer formato comum) e faça perguntas em linguagem natural (PT-BR).
O agente detecta colunas de **valor**/**data**, normaliza números com vírgula, verifica CNPJ/CPF e gera gráficos.
Também produz **conclusões automáticas**, salva **memória (JSON)** e gera **PDF** no padrão da atividade.
""")

uploaded = st.file_uploader("Arraste e solte seu CSV aqui", type=["csv"])
if uploaded is not None:
    ag = CSVAgent().load(uploaded)
    prof = ag.profile()

    c1, c2, c3 = st.columns(3)
    c1.metric("Linhas", f"{prof['linhas']:,}".replace(",", "."))
    c2.metric("Colunas", prof["colunas"])
    c3.metric("Memória", prof["uso_de_memoria"])

    with st.expander("Tipos & Faltantes (%)", expanded=False):
        st.json({"tipos": prof["tipos"], "faltantes_por_coluna_pct": prof["faltantes_por_coluna_pct"]})

    st.write(f"**Valor detectado:** `{ag.pick_value_col()}`  |  **Data detectada:** `{ag.pick_date_col()}`")

    st.divider()
    st.subheader("❓ Perguntas")
    q = st.text_input("Ex.: 'total por fornecedor', 'top 10 de orgao', 'média de amount por class', 'sql: SELECT count(*) FROM df'")
    if st.button("Perguntar", use_container_width=True) and q:
        ans, msg = ag.ask(q)
        st.caption(msg)
        if not ans.empty:
            st.dataframe(ans.head(500))

        # Plot automático (barra) para respostas "grupo-valor" curtas
        try:
            import pandas as _pd
            df_ans = ans.copy()
            if len(df_ans.columns) == 2 and len(df_ans) <= 25:
                col1, col2 = df_ans.columns.tolist()
                if not _pd.api.types.is_numeric_dtype(df_ans[col1]) and _pd.api.types.is_numeric_dtype(df_ans[col2]):
                    st.bar_chart(df_ans.set_index(col1)[col2])
        except Exception:
            pass

        else:
            st.warning("Sem resultados.")

    st.divider()
    st.subheader("📈 Gráficos automáticos")
    charts = ag.generate_charts(outdir=Path("charts_streamlit"))
    for kind, paths in charts.items():
        if not paths: continue
        st.markdown(f"**{kind.title()}**")
        for p in paths:
            st.image(p, use_column_width=True)

    st.divider()
    st.subheader("🧠 Conclusões do Agente & Memória")
    concl = ag.compute_conclusions()
    st.write(concl["texto"])
    with st.expander("Detalhes das conclusões", expanded=False):
        st.json(concl)

    c1, c2 = st.columns(2)
    if c1.button("Salvar Memória (JSON)", use_container_width=True):
        mem_path = ag.save_memory("agent_memory.json")
        st.success(f"Memória salva em {mem_path}")
    if c2.button("Gerar PDF (nome exigido)", use_container_width=True):
        pdf_path = ag.save_pdf_report("Agentes Autônomos – Relatório da Atividade Extra.pdf")
        st.success("PDF gerado com sucesso.")
        st.markdown(f"[📄 Abrir PDF]({'Agentes Autônomos – Relatório da Atividade Extra.pdf'})")

    st.divider()
    st.subheader("📝 Relatório HTML")
    if st.button("Gerar relatório (HTML)", use_container_width=True):
        out = ag.save_html_report("eda_report.html")
        st.success("Relatório gerado.")
        st.markdown(f"[📄 Abrir relatório]({'eda_report.html'})")
else:
    st.info("Envie um CSV para começar.")
