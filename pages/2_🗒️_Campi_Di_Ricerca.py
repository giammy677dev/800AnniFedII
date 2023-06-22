from utils import st, conn

st.set_page_config(
    page_title="Campi di Ricerca",
    page_icon="️📊",
    layout="wide"
)

st.title('🌍 Campi di Ricerca')

st.write('''In questa sezione bla bla bla.
        ''')

# Layout a due colonne
col1, col2 = st.columns([1, 1])

with col1:
    # Aggiungiamo il filtro per selezionare la macro-categoria
    query = "MATCH (f:Field) WHERE toInteger(f.Field_Code) < 99 AND f.Name <> 'NaN' RETURN f.Name"
    query_results = conn.query(query)
    string_results = [record['f.Name'] for record in query_results]
    selected_macro = st.selectbox('Seleziona la macro-categoria:', string_results)

# Aggiungiamo l'info box con le informazioni dell'utente selezionato
with col2:
    query = "MATCH (f:Field) WHERE SIZE(TRIM(f.Field_Code)) > 2 RETURN f.Name"
    query_results = conn.query(query)
    string_results = [record['f.Name'] for record in query_results]
    selected_macro = st.selectbox('Seleziona la macro-categoria:', string_results)
