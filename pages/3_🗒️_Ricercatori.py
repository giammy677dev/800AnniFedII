from utils import st, conn
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(
    page_title="Ricercatori",
    page_icon="️🔍",
    layout="wide"
)

st.title('🔍 Ricercatori')

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

# Aggiungiamo l'info box con le informazioni del ricercatore selezionato
with col2:
    query = f"MATCH (r:Researcher)-[rs:Research]->(p:Project) WHERE r.Name = '{selected_researcher}' return p.Funding"
    query_results = conn.query(query)
    funding_results = [record['p.Funding'] for record in query_results]

    funding_results = [int(x) for x in funding_results]

    with st.expander("Info Box del ricercatore", expanded=True):
        st.write(
            '''
                {}
                {}
                {}
            '''.format(
                f'<p style="color: white;"><b style="color: #3e8ad2;">Numero di Progetti: </b>{len(funding_results)} </p>',
                f'<p style="color: white;"><b style="color: #3e8ad2;">Fondi totali: </b> {sum(funding_results)} €</p>',
                f'<p style="color: white;"><b style="color: #3e8ad2;">Fondi Medi: </b> {round(sum(funding_results)/len(funding_results),2)} €</p>' if
                len(funding_results) > 1 else f'<p style="color: white;"><b style="color: #3e8ad2;"></p>'
            ),
            unsafe_allow_html=True
        )

# PARTE GRAFICO NEO4J e FILTRI

# TABELLA

query = f"""MATCH (r:Researcher)-[rs:Research]->(p:Project) WHERE r.Name = '{selected_researcher}'
            WITH p, REDUCE(count = 0, pub in p.Publications | count + CASE WHEN pub CONTAINS 'pub.' THEN 1 ELSE 0 END) AS pubCount
            RETURN p.Title, p.Abstract, p.Start_Date, p.End_Date, pubCount, p.Funder, 
            p.Funder_Group, p.Funder_Country, p.Funding, p.Dimensions_URL, p.Source_Linkout
        """

query_results = conn.query(query)
project_results = [[record['p.Title'], record['p.Start_Date'], record['p.End_Date'],
                    record['pubCount'], record['p.Funder'], record['p.Funder_Group'], record['p.Funder_Country'],
                    record['p.Funding'], record['p.Dimensions_URL'], record['p.Source_Linkout']]
                   for record in query_results]

# Converte il dizionario in un DataFrame
df = pd.DataFrame(project_results, columns=['Titolo', 'Data d\'inizio', 'Data di Fine', 'Numero Pubblicazioni', 'Funder',
                                            'Funder Group', 'Funder Country', 'Fondi', 'URL Dimensions',
                                            'Link'])
# Visualizza la tabella
st.table(df)

# Layout a due colonne
col3, col4 = st.columns([1, 1])

# GRAFICO A TORTA MACROCATEGORIE
with col3:
    query = f"""MATCH (r:Researcher)-[rs:Research]->(p:Project)-[:About]->(f:Field)
                WHERE r.Name = '{selected_researcher}' AND toInteger(f.Field_Code) > 0 AND toInteger(f.Field_Code) < 99
                RETURN f.Name AS field, COUNT(*) AS Count
                ORDER By Count DESC"""
    query_results = conn.query(query)
    macro_field_results = [[record['field'], record['Count']] for record in query_results]

    field_list = [record[0] for record in macro_field_results]
    count_list = [record[1] for record in macro_field_results]

    fig = go.Figure(data=[go.Pie(labels=field_list, values=count_list)])

    st.header("Percentuale Macrocategorie")
    st.write(f"""Di seguito viene presentata la percentuale delle macrocategorie
                 dei campi di ricerca su cui il ricercatore considerato ha lavorato.
            """)

    # Visualizzazione del grafico su Streamlit
    st.plotly_chart(fig, use_container_width=True)

# GRAFICO A TORTA MICROCATEGORIE
with col4:
    query = f"""MATCH (r:Researcher)-[rs:Research]->(p:Project)-[:About]->(f:Field)
                WHERE r.Name = '{selected_researcher}' AND toInteger(f.Field_Code) > 99
                RETURN f.Name AS field, COUNT(*) AS Count
                ORDER By Count DESC"""
    query_results = conn.query(query)
    micro_field_results = [[record['field'], record['Count']] for record in query_results]

    field_list = [record[0] for record in micro_field_results]
    count_list = [record[1] for record in micro_field_results]

    fig = go.Figure(data=[go.Pie(labels=field_list, values=count_list)])

    st.header("Percentuale Microcategorie")
    st.write(f"""Di seguito viene presentata la percentuale delle microcategorie
                 dei campi di ricerca su cui il ricercatore considerato ha lavorato.
            """)

    # Visualizzazione del grafico su Streamlit
    st.plotly_chart(fig, use_container_width=True)