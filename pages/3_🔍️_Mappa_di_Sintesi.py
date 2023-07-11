from utils import st, conn, agraph, Node, Edge, Config

st.set_page_config(
    page_title="Mappa di Sintesi",
    page_icon="🔍",
    layout="wide"
)

st.title('🔍 Mappa di Sintesi')

st.write('''In questa sezione bla bla bla.
        ''')

# Aggiungiamo il filtro per selezionare la macro-categoria
query = """MATCH (f:Field) WHERE toInteger(f.Field_Code) < 99 AND f.Name <> 'NaN'
        RETURN toInteger(f.Field_Code) AS Field_Code, f.Name AS Name
        ORDER BY Name
        """
query_results = conn.query(query)
macro_fields_results = [(record['Field_Code'], record['Name']) for record in query_results]
selected_macro_name = st.selectbox('Seleziona il settore disciplinare:', [name[1] for name in macro_fields_results], index=11)

# Trova il Field_Code corrispondente al campo selezionato
selected_macro_code = None
for record in macro_fields_results:
    if record[1] == selected_macro_name:
        selected_macro_code = record[0]

query = f"""MATCH (f:Field)<-[]-(p:Project)
                WHERE f.Field_Code = '{selected_macro_code}' OR f.Field_Code =~ '^{selected_macro_code}\\d{{2}}$'
                RETURN f AS Field, p AS Progetto
            """
query_results = conn.query(query)
results = [(record['Field'], record['Progetto']) for record in query_results]

nodes_project = []
edges_project = []
ids_project = []

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

macro_field = ""
micro_field = ""

# Layout a due colonne
col1, col2 = st.columns([10, 90])

with col1:
    # Creazione della legenda
    st.subheader('Legenda')

    # Definizione dei colori e delle etichette della legenda
    colors_project = ['orange', 'blue', 'yellow']
    labels_project = ['Macro-settore', 'Settore', 'Progetto']

    for color, label in zip(colors_project, labels_project):
        st.markdown(f'<span style="color:{color}">●</span> {label}', unsafe_allow_html=True)

with col2:
    for record in results:
        field = record[0]
        if field.element_id not in ids_project:
            ids_project.append(field.element_id)
            if field["Name"] != 'NaN':
                if int(field["Field_Code"]) < 99:
                    nodes_project.append(Node(id=field.element_id,
                                              title=field["Name"],
                                              size=15,
                                              color='orange')
                                         )
                    macro_field = field.element_id
                else:
                    nodes_project.append(Node(id=field.element_id,
                                              title=field["Name"],
                                              size=10,
                                              color='blue')
                                         )
                    micro_field = field.element_id
            else:
                nodes_project.append(Node(id=field.element_id,
                                          title="Non Definito",
                                          size=10,
                                          color='blue')
                                     )

        edges_project.append(Edge(source=macro_field,
                                  target=micro_field,
                                  color='grey',
                                  font={'size': 15}
                                  )
                             )

        project = record[1]
        if project.element_id not in ids_project:
            ids_project.append(project.element_id)
            if project["Title"] != 'NaN':
                nodes_project.append(Node(id=project.element_id,
                                          title=project["Title"],
                                          size=5,
                                          color='yellow')
                                     )
            else:
                nodes_project.append(Node(id=project.element_id,
                                          title="Non Definito",
                                          size=5,
                                          color='yellow')
                                     )

        edges_project.append(Edge(source=micro_field,
                                  target=project.element_id,
                                  color='grey',
                                  font={'size': 15}
                                  )
                             )

    agraph(nodes=nodes_project, edges=edges_project, config=config)

# Chiudiamo la connessione al DB
conn.close()
