from utils import st, conn

st.set_page_config(
    page_title="Ricercatori",
    page_icon="️📊",
    layout="wide"
)

st.title('🌍 Ricercatori')

st.write('''In questa sezione bla bla bla.
        ''')

# Layout a due colonne
col1, col2 = st.columns([1, 1])

with col1:
    # Aggiungiamo il filtro per selezionare la macro-categoria
    query = "MATCH (r:Researcher) WHERE r.Name <> 'NaN' RETURN r.Name ORDER BY r.Name"
    query_results = conn.query(query)
    string_results = [record['r.Name'] for record in query_results]
    selected_researcher = st.selectbox('Seleziona il ricercatore:', string_results)

# Aggiungiamo l'info box con le informazioni dell'utente selezionato
with col2:
    query = f"MATCH (r:Researcher)-[rs:Research]->(p:Project) WHERE r.Name = '{selected_researcher}' return p"
    query_results = conn.query(query)
    project_results = [record['p'] for record in query_results]
    st.write(project_results)