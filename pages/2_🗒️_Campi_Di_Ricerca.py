from utils import st, conn, today
from streamlit_agraph import agraph, Node, Edge, Config

st.set_page_config(
    page_title="Campi di Ricerca",
    page_icon="📊",
    layout="wide"
)

st.title('🌍 Campi di Ricerca')

st.write('''In questa sezione bla bla bla.
        ''')

# Layout a due colonne
col1, col2 = st.columns([1, 1])

with col1:
    # Aggiungiamo il filtro per selezionare la macro-categoria
    query = """MATCH (f:Field) WHERE toInteger(f.Field_Code) < 99 AND f.Name <> 'NaN'
            RETURN toInteger(f.Field_Code) AS Field_Code, f.Name AS Name
            """
    query_results = conn.query(query)
    macro_fields_results = [(record['Field_Code'], record['Name']) for record in query_results]
    selected_macro_name = st.selectbox('Seleziona la macro-categoria:', [name[1] for name in macro_fields_results])
    # Trova il Field_Code corrispondente al campo selezionato
    selected_macro_code = None
    for record in macro_fields_results:
        if record[1] == selected_macro_name:
            selected_macro_code = record[0]

# Aggiungiamo il filtro per selezionare la micro-categoria
with col2:
    query = f"""MATCH (f:Field)
                WHERE f.Field_Code =~ '^{selected_macro_code}\\d{{2}}$'
                RETURN toInteger(f.Field_Code) AS Field_Code, f.Name AS Name
            """
    query_results = conn.query(query)
    micro_fields_results = [(record['Field_Code'], record['Name']) for record in query_results]
    selected_micro_name = st.selectbox('Seleziona la micro-categoria:', [name[1] for name in micro_fields_results])
    # Trova il Field_Code corrispondente al campo selezionato
    selected_micro_code = None
    for record in micro_fields_results:
        if record[1] == selected_micro_name:
            selected_micro_code = record[0]

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
colors = ['green', 'yellow']
labels = ['Campo di Ricerca', 'Progetto']

# Layout a due colonne
col3, col4 = st.columns([1, 3])
with col3:
    # Definiamo i valori booleani per filtrare il grafo
    st.subheader('Filtri')
    add_researchers = st.checkbox('Visualizza i Ricercatori', key='researchers')
    add_ongoing_projects = st.checkbox('Visualizza solo i Progetti in corso', key='ongoing_projects')

    st.divider()

    # Creazione della legenda
    st.subheader('Legenda')
    if add_researchers:
        colors.append('grey')
        labels.append('Ricercatore')

    for color, label in zip(colors, labels):
        st.markdown(f'<span style="color:{color}">●</span> {label}', unsafe_allow_html=True)


with col4:
    if not add_ongoing_projects:
        query = f"""MATCH (r:Researcher)-[]->(p:Project)-[]->(f:Field)
                        WHERE f.Field_Code = '{selected_micro_code}'
                        RETURN f AS Campo, p AS Progetto, r AS Ricercatore
                    """
    else:
        query = f"""MATCH (r:Researcher)-[]->(p:Project)-[]->(f:Field)
                                WHERE f.Field_Code = '{selected_micro_code}' AND 
                                datetime({{year: toInteger(split(p.End_Date, '/')[2]),
                                month: toInteger(split(p.End_Date, '/')[1]),
                                day: toInteger(split(p.End_Date, '/')[0])}})
                                > datetime('{today}')
                                RETURN f AS Campo, p AS Progetto, r AS Ricercatore
                            """
    query_results = conn.query(query)
    results = [(record['Campo'], record['Progetto'], record['Ricercatore']) for record in query_results]

    for record in results:
        field = record[0]
        if field.element_id not in ids:
            ids.append(field.element_id)
            nodes.append(Node(id=field.element_id,
                              title=field["Name"],
                              size=15,
                              color='green')
                         )

        project = record[1]
        if project.element_id not in ids:
            ids.append(project.element_id)
            nodes.append(Node(id=project.element_id,
                              title=project["Title"],
                              size=10,
                              color='yellow')
                         )

        edges.append(Edge(source=project.element_id,
                          label="Riguarda",
                          target=field.element_id,
                          color='black',
                          font={'size': 10}
                          )
                     )

        if add_researchers:
            researcher = record[2]
            if researcher.element_id not in ids:
                ids.append(researcher.element_id)
                nodes.append(Node(id=researcher.element_id,
                                  title=researcher["Name"],
                                  size=8,
                                  color='grey')
                             )

            edges.append(Edge(source=researcher.element_id,
                              label="Ricerca",
                              target=project.element_id,
                              color='black',
                              font={'size': 10}
                              )
                         )

    agraph(nodes=nodes, edges=edges, config=config)

# Explicitly close the connection
conn.close()
