from utils import st, conn

st.set_page_config(
    page_title="Campi di Ricerca",
    page_icon="️📊",
    layout="wide"
)

st.title('🌍 Campi di Ricerca')

st.write('''In questa sezione bla bla bla.
        ''')

# Aggiungiamo il filtro per selezionare la macro-categoria
query = "MATCH (u:Utente) RETURN u.screen_name ORDER BY u.screen_name"
query_results = conn.query(query)
string_results = [record['u.screen_name'] for record in query_results]
selected_user = st.selectbox('Seleziona l\'utente:', string_results, index=11)
