from utils import st, conn, today
import plotly.graph_objects as go
from streamlit_agraph import agraph, Node, Edge, Config
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

# Configurazione per Agraph
config = Config(width=1000,
                height=600,
                directed=True,
                physics={"barnesHut": {"gravitationalConstant": -10000,
                                       "centralGravity": 0.5,
                                       "springLength": 100,
                                       "springConstant": 0.04,
                                       "damping": 0.09,
                                       "avoidOverlap": 1}},
                hierarchical=False
                )

nodes = []
edges = []
ids = []

# Definizione dei colori e delle etichette della legenda
colors = ['#3e8ad2', 'yellow']
labels = ['Ricercatore', 'Progetto']

# Layout a due colonne
col3, col4 = st.columns([1, 3])
with col3:
    # Definiamo i valori booleani per filtrare il grafo
    st.subheader('Filtri')
    add_fields = st.checkbox('Visualizza i Campi di Ricerca', key='fields')
    add_ongoing_projects = st.checkbox('Visualizza solo i Progetti in corso', key='ongoing_projects')

    st.divider()

    # Creazione della legenda
    st.subheader('Legenda')
    if add_fields:
        colors.append('green')
        labels.append('Campi di Ricerca')

    for color, label in zip(colors, labels):
        st.markdown(f'<span style="color:{color}">●</span> {label}', unsafe_allow_html=True)


with col4:
    if not add_ongoing_projects:
        query = f"""MATCH (r:Researcher)-[]->(p:Project)-[]->(f:Field)
                        WHERE r.Name = '{selected_researcher}'
                        RETURN r AS Ricercatore, p AS Progetto, f AS Campo
                    """
    else:
        query = f"""MATCH (r:Researcher)-[]->(p:Project)-[]->(f:Field)
                                WHERE f.Field_Code = '{selected_researcher}' AND 
                                datetime({{year: toInteger(split(p.End_Date, '/')[2]),
                                month: toInteger(split(p.End_Date, '/')[1]),
                                day: toInteger(split(p.End_Date, '/')[0])}})
                                > datetime('{today}')
                                RETURN r AS Ricercatore, p AS Progetto, f AS Campo
                            """
    query_results = conn.query(query)
    results = [(record['Ricercatore'], record['Progetto'], record['Campo']) for record in query_results]

    for record in results:
        researcher = record[0]
        if researcher.element_id not in ids:
            ids.append(researcher.element_id)
            if researcher["Name"] != 'NaN':
                nodes.append(Node(id=researcher.element_id,
                                  title=researcher["Name"],
                                  size=15,
                                  color='#3e8ad2')
                             )
            else:
                nodes.append(Node(id=researcher.element_id,
                                  title="Non Definito",
                                  size=15,
                                  color='#3e8ad2')
                             )

        project = record[1]
        if project.element_id not in ids:
            ids.append(project.element_id)
            if project["Title"] != 'NaN':
                nodes.append(Node(id=project.element_id,
                                  title=project["Title"],
                                  size=10,
                                  color='yellow')
                             )
            else:
                nodes.append(Node(id=project.element_id,
                                  title="Non Definito",
                                  size=10,
                                  color='yellow')
                             )

        edges.append(Edge(source=researcher.element_id,
                          label="Ricerca",
                          target=project.element_id,
                          color='grey',
                          font={'size': 10},
                          )
                     )

        if add_fields:
            field = record[2]
            if field.element_id not in ids:
                ids.append(field.element_id)
                if field["Name"] != 'NaN':
                    nodes.append(Node(id=field.element_id,
                                      title=field["Name"],
                                      size=8,
                                      color='green')
                                 )
                else:
                    nodes.append(Node(id=field.element_id,
                                      title="Non Definito",
                                      size=8,
                                      color='green')
                                 )

            edges.append(Edge(source=project.element_id,
                              label="Riguarda",
                              target=field.element_id,
                              color='grey',
                              font={'size': 10}
                              )
                         )

    agraph(nodes=nodes, edges=edges, config=config)

# Tabella
query = f"""MATCH (r:Researcher)-[rs:Research]->(p:Project) WHERE r.Name = '{selected_researcher}'
            RETURN p.Title AS Titolo, p.Funding as Fondi, p.Start_Date AS DataInizio,
                    p.End_Date as DataFine, p.Funder as Finanziatore, p.Funder_Group as Gruppo,
                    p.Program AS Programma
        """

query_results = conn.query(query)
project_results = [(
        'Non Definito' if record['Titolo'] == 'NaN' else record['Titolo'],
        'Non Definito' if record['Fondi'] == 'NaN' else record['Fondi'],
        'Non Definito' if record['DataInizio'] == 'NaN' else record['DataInizio'],
        'Non Definito' if record['DataFine'] == 'NaN' else record['DataFine'],
        'Non Definito' if record['Finanziatore'] == 'NaN' else record['Finanziatore'],
        'Non Definito' if record['Gruppo'] == 'NaN' else record['Gruppo'],
        'Non Definito' if record['Programma'] == 'NaN' else record['Programma'])
    for record in query_results
]

columns = ['Titolo', 'Fondi Investiti (€)', 'Data di Inizio', 'Data di Fine', 'Finanziatore',
           'Gruppo di Finanziamento', 'Programma']

df = pd.DataFrame(project_results, columns=columns)
df.set_index('Titolo', inplace=True)
st.write(df)

if len(df) > 1:
    # Layout a due colonne
    col3, col4 = st.columns([1, 1])

    # GRAFICO A TORTA MACROCATEGORIE
    with col3:
        query = f"""MATCH (r:Researcher)-[rs:Research]->(p:Project)-[:About]->(f:Field)
                    WHERE r.Name = '{selected_researcher}' AND toInteger(f.Field_Code) > 0 AND toInteger(f.Field_Code) < 99
                    RETURN f.Name AS field, COUNT(*) AS Count
                    ORDER By Count DESC"""
        query_results = conn.query(query)
        macro_field_results = [(record['field'], record['Count']) for record in query_results]

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
        micro_field_results = [(record['field'], record['Count']) for record in query_results]

        field_list = [record[0] for record in micro_field_results]
        count_list = [record[1] for record in micro_field_results]

        fig = go.Figure(data=[go.Pie(labels=field_list, values=count_list)])

        st.header("Percentuale Microcategorie")
        st.write(f"""Di seguito viene presentata la percentuale delle microcategorie
                     dei campi di ricerca su cui il ricercatore considerato ha lavorato.
                """)

        # Visualizzazione del grafico su Streamlit
        st.plotly_chart(fig, use_container_width=True)

# Chiudiamo la connessione al DB
conn.close()
