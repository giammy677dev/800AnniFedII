from utils import st, conn
from py2neo import Graph
from pyvis.network import Network
from streamlit_agraph import agraph, Node, Edge, Config

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
    # micro_fields_results.insert(0, "---") # Aggiungi un valore "Null" come primo elemento
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

if selected_micro_name == '---':
    query = f"""MATCH (r:Researcher)-[]->(p:Project)-[]->(f:Field)<-[]-(o:Organization)-[]->(p:Project)
                WHERE f.Field_Code = '{selected_macro_code}'
                RETURN r AS Ricercatore, p AS Progetto, f AS Campo, o AS Organizzazione
                LIMIT 100
            """
    query_results = conn.query(query)
    results = [(record['Ricercatore'], record['Progetto'], record['Campo'], record['Organizzazione']) for record in query_results]

    for record in results:
        researcher = record[0]
        if researcher.element_id not in ids:
            ids.append(researcher.element_id)
            nodes.append(Node(id=researcher.element_id,
                              label=researcher["Name"],
                              title=researcher["Name"],
                              size=30,
                              color='grey')
                         )

        project = record[1]
        if project.element_id not in ids:
            ids.append(project.element_id)
            nodes.append(Node(id=project.element_id,
                              label=project["Title"],
                              title=project["Title"],
                              size=30,
                              color='yellow')
                         )

        field = record[2]
        if field.element_id not in ids:
            ids.append(field.element_id)
            nodes.append(Node(id=field.element_id,
                              label=field["Name"],
                              title=field["Name"],
                              size=30,
                              color='green')
                         )
        organization = record[3]
        if organization.element_id not in ids:
            ids.append(organization.element_id)
            nodes.append(Node(id=organization.element_id,
                              label=organization["Name"],
                              title=organization["Name"],
                              size=30,
                              color='blue')
                         )

        edges.append(Edge(source=researcher.element_id,
                          label="Ricerca",
                          target=project.element_id,
                          color='black'
                          )
                     )

        edges.append(Edge(source=project.element_id,
                          label="Riguarda",
                          target=field.element_id,
                          color='black'
                          )
                     )

        edges.append(Edge(source=organization.element_id,
                          label="Studia",
                          target=field.element_id,
                          color='black'
                          )
                     )

        edges.append(Edge(source=organization.element_id,
                          label="Finanzia",
                          target=project.element_id,
                          color='black'
                          )
                     )
else:
    query = f"""MATCH (r:Researcher)-[]->(p:Project)-[]->(f:Field)<-[]-(o:Organization)-[]->(p:Project)
                    WHERE f.Field_Code = '{selected_micro_code}'
                    RETURN r AS Ricercatore, p AS Progetto, f AS Campo, o AS Organizzazione
                    LIMIT 100
                """
    query_results = conn.query(query)
    results = [(record['Ricercatore'], record['Progetto'], record['Campo'], record['Organizzazione']) for record in
               query_results]

    for record in results:
        researcher = record[0]
        if researcher.element_id not in ids:
            ids.append(researcher.element_id)
            nodes.append(Node(id=researcher.element_id,
                              label=researcher["Name"],
                              title=researcher["Name"],
                              size=30,
                              color='grey')
                         )

        project = record[1]
        if project.element_id not in ids:
            ids.append(project.element_id)
            nodes.append(Node(id=project.element_id,
                              label=project["Title"],
                              title=project["Title"],
                              size=30,
                              color='yellow')
                         )

        field = record[2]
        if field.element_id not in ids:
            ids.append(field.element_id)
            nodes.append(Node(id=field.element_id,
                              label=field["Name"],
                              title=field["Name"],
                              size=30,
                              color='green')
                         )

        edges.append(Edge(source=researcher.element_id,
                          label="Ricerca",
                          target=project.element_id,
                          color='black'
                          )
                     )
        edges.append(Edge(source=project.element_id,
                          label="Riguarda",
                          target=field.element_id,
                          color='black'
                          )
                     )

agraph(nodes=nodes, edges=edges, config=config)

# Explicitly close the connection
conn.close()
