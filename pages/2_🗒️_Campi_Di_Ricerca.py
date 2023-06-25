from utils import st, conn, today
from streamlit_agraph import agraph, Node, Edge, Config
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from stopwords import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pydeck as pdk
from geopy.geocoders import Nominatim
import re

st.set_page_config(
    page_title="Campi di Ricerca",
    page_icon="📚",
    layout="wide"
)

st.title('📚 Campi di Ricerca')

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
                height=450,
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
        colors.append('#3e8ad2')
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
                          color='grey',
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
                                  color='#3e8ad2')
                             )

            edges.append(Edge(source=researcher.element_id,
                              label="Ricerca",
                              target=project.element_id,
                              color='grey',
                              font={'size': 10}
                              )
                         )

    agraph(nodes=nodes, edges=edges, config=config)

# Costruiamo la tabella dei progetti appartenenti alla micro-categoria selezionata

if not add_ongoing_projects:
    query = f"""MATCH (p:Project)-[]->(f:Field)
                WHERE f.Field_Code = '{selected_micro_code}'
                RETURN p.Title AS Titolo, p.Funding as Fondi, p.Start_Date AS DataInizio,
                        p.End_Date as DataFine, p.Funder as Finanziatore, p.Funder_Group as Gruppo,
                        p.Program AS Programma
            """
else:
    query = f"""MATCH (p:Project)-[]->(f:Field)
                WHERE f.Field_Code = '{selected_micro_code}' AND 
                        datetime({{year: toInteger(split(p.End_Date, '/')[2]),
                        month: toInteger(split(p.End_Date, '/')[1]),
                        day: toInteger(split(p.End_Date, '/')[0])}})
                        > datetime('{today}')
                RETURN p.Title AS Titolo, p.Funding as Fondi, p.Start_Date AS DataInizio,
                        p.End_Date as DataFine, p.Funder as Finanziatore, p.Funder_Group as Gruppo,
                        p.Program AS Programma
            """

query_results = conn.query(query)
results = [(
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

df = pd.DataFrame(results, columns=columns)
df.set_index('Titolo', inplace=True)
st.write(df)

# Istogrammi numero di progetti e fondi investiti nel tempo
query = f"""MATCH (p:Project)-[]->(f:Field)
            WHERE f.Field_Code = '{selected_micro_code}'
            WITH p, p.Start_Year AS year, toInteger(p.Funding) AS funding
            RETURN year, count(p) AS projectCount, sum(funding) AS totalFunding
         """
results = conn.query(query)
columns = ['year', 'projectCount', 'totalFunding']

# Mettiamo i risultati in un DataFrame
df_projects = pd.DataFrame(results, columns=columns)

# Creazione del grafico temporale per il numero di progetti anno per anno
fig_projects = go.Figure(data=go.Bar(x=df_projects['year'], y=df_projects['projectCount']))

# Personalizzazione del grafico dei progetti
fig_projects.update_layout(
    title='Numero di Progetti per Anno',
    xaxis_title='Anno',
    yaxis_title='Conteggio Progetti',
    yaxis=dict(tickvals=np.arange(0, max(df_projects['projectCount']) + 1, 1))  # Valori dei tick come numeri interi
)

# Creazione del grafico temporale per il numero di fondi investiti anno per anno
fig_funding = go.Figure(data=go.Bar(x=df_projects['year'], y=df_projects['totalFunding']))

# Personalizzazione del grafico dei fondi investiti
max_y = np.max(df_projects['totalFunding']) * 1.25
fig_funding.update_layout(
    title='Fondi Investiti per Anno',
    xaxis_title='Anno',
    yaxis_range=[0, max_y],
    yaxis_title='Fondi Investiti (€)'
)

# Mostra i grafici su due colonne
col5, col6 = st.columns([1, 1])
with col5:
    st.plotly_chart(fig_projects)
with col6:
    st.plotly_chart(fig_funding)

# Aggiungiamo la WordCloud per la micro-categoria selezionata
query = f"""MATCH (p:Project)-[]->(f:Field)
            WHERE f.Field_Code = '{selected_micro_code}'
            WITH p.Abstract AS testo
            WITH testo, SPLIT(toLower(testo), ' ') AS parole
            UNWIND parole AS parola
            WITH REPLACE(REPLACE(REPLACE(parola, ':', ''), ',', ''), '.', '') AS word_without_punckt, 
                COUNT(DISTINCT testo) AS frequenza
            WHERE frequenza > 1 AND NOT word_without_punckt IN {stopwords}
            RETURN word_without_punckt AS parola, frequenza
            ORDER BY frequenza DESC
        """
query_results = conn.query(query)
frequency_results = [(record['parola'], record['frequenza']) for record in query_results]

frequency_dictionary = {}
for tupla in frequency_results:
    parola = str(tupla[0])
    frequenza = int(tupla[1])

    # Rimuovi eventuali caratteri di nuova riga dal testo
    parola = re.sub(r'\n', ' ', parola)

    # Aggiungi la parola e la sua frequenza al dizionario
    if parola in frequency_dictionary:
        frequency_dictionary[parola] += frequenza
    else:
        frequency_dictionary[parola] = frequenza

wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(frequency_dictionary)

# Visualizza il tag cloud in Streamlit
fig, ax = plt.subplots()
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis('off')
st.pyplot(fig)

st.write("-------------------------------------------------------------")
st.header("Mappa geografica")
st.write("Mappa geografica per visualizzare le città che hanno interagito maggiormente nel micro-campo precedentemente "
         "selezionato. ")

# Query per conteggio macro-categorie
query = f"""MATCH (f:Field)-[r1]-(p:Project)-[r2]-(o:Organization) 
            WHERE f.Field_Code = '{selected_micro_code}' AND o.Name <> 'University of Naples Federico II'
            RETURN o.City AS City, count(p) as Conteggio_Progetti, o.Name as Nome_Organizzazione
            ORDER BY Conteggio_Progetti DESC
        """
query_results = conn.query(query)

# Creazione di un DataFrame dai risultati della query
city_data = pd.DataFrame(query_results, columns=['City', 'Conteggio_Progetti', 'Nome_Organizzazione'])


# Creazione della mappa interattiva
@st.cache_data
def map_creation(micro_code):
    if micro_code is not None:
        # Creazione di un'istanza del geocoder Nominatim
        geolocator = Nominatim(user_agent="my_geocoder")

        # Ottenimento delle coordinate geografiche per ogni città
        city_data['Latitude'] = np.nan
        city_data['Longitude'] = np.nan

        for index, row in city_data.iterrows():
            city_name = row['City']
            location = geolocator.geocode(city_name)
            if location:
                latitude = location.latitude
                longitude = location.longitude
                city_data.at[index, 'Latitude'] = latitude
                city_data.at[index, 'Longitude'] = longitude

        # Creazione del DataFrame chart_data per la visualizzazione sulla mappa
        chart_data = city_data[['Latitude', 'Longitude', 'Conteggio_Progetti', 'Nome_Organizzazione']].copy()

        # Converti il DataFrame in un elenco di dizionari
        chart_data_dict = chart_data.to_dict('records')

        # Preparo il layer per l'istogramma che deve comparire nel grafico
        layerIstogramma = pdk.Layer(
            'ColumnLayer',
            data=chart_data_dict,
            get_position='[Longitude, Latitude]',
            get_elevation='Conteggio_Progetti',
            elevation_scale=10000,
            get_color='[200, 30, 0, 100]',
            radius=7000,
            pickable=True,
            auto_highlight=True,
            extruded=True
        )

        # Preparo il layer per l'area col raggio specificato
        layerArea = pdk.Layer(
            'ScatterplotLayer',
            data=chart_data_dict,
            get_position='[Longitude, Latitude]',
            get_color='[200, 200, 0, 25]',
            get_radius=50000
        )

        st.pydeck_chart(
            pdk.Deck(
                map_style=None,
                initial_view_state=pdk.ViewState(
                    latitude=42.41183986816765,
                    longitude=12.865247589554036,
                    zoom=5,
                    pitch=50,
                ),
                layers=[layerIstogramma, layerArea],
                tooltip={
                    "html": "<b>Nome Organizzazione: </b> {Nome_Organizzazione} <br />"
                            "<b>Numero progetti: </b> {Conteggio_Progetti} <br />"
                }
            )
        )


map_creation(selected_micro_code)

# Selezioniamo il singolo progetto della micro-categoria selezionata
st.write("-------------------------------------------------------------")
st.header("Seleziona Progetto Microcategoria")
st.write("In questa sezione è possibile visualizzare le info su uno dei progetto della microcategoria selezionata ")

query = f"""MATCH (p:Project)-[]->(f:Field)
            WHERE f.Field_Code = "{selected_micro_code}"
            RETURN p.Title AS Titolo
            """
query_results = conn.query(query)
project_micro_fields_results = [record['Titolo'] for record in query_results]
selected_project_micro_name = st.selectbox('Seleziona un progetto della microcategoria selezionata:', project_micro_fields_results)

query = f"""MATCH (p:Project)-[]->(f:Field)
            WHERE p.Title = "{selected_project_micro_name}"
            RETURN DISTINCT p.Title AS Titolo, p.Abstract AS Abstract, p.Funding as Fondi, p.Start_Date AS DataInizio,
                   p.End_Date as DataFine, p.Funder as Finanziatore, p.Funder_Group as Gruppo, 
                   p.Publications as Pubblicazioni, p.Program AS Programma, p.Dimensions_URL AS DimensionsURL, 
                   p.Source_Linkout as Link
            """
query_results = conn.query(query)

project_info = [(
        'Non Definito' if record['Titolo'] == 'NaN' else record['Titolo'],
        'Non Definito' if record['Abstract'] == 'NaN' else record['Abstract'],
        'Non Definito' if record['Fondi'] == 'NaN' else record['Fondi'],
        'Non Definito' if record['DataInizio'] == 'NaN' else record['DataInizio'],
        'Non Definito' if record['DataFine'] == 'NaN' else record['DataFine'],
        'Non Definito' if record['Finanziatore'] == 'NaN' else record['Finanziatore'],
        'Non Definito' if record['Gruppo'] == 'NaN' else record['Gruppo'],
        'Non Definito' if record['Pubblicazioni'] == 'NaN' else record['Pubblicazioni'],
        'Non Definito' if record['Programma'] == 'NaN' else record['Programma'],
        'Non Definito' if record['DimensionsURL'] == 'NaN' else record['DimensionsURL'],
        'Non Definito' if record['Link'] == 'NaN' else record['Link'])
    for record in query_results
]

with st.expander("Info Box del progetto", expanded=True):
    st.write(
        '''
            {}
            {}
            {}
            {}
            {}
            {}
            {}
            {}
            {}
            {}
            {}
        '''
        .format(
            f'<p style="color: white;"><b style="color: #3e8ad2;">Titolo: </b>{project_info[0][0]} </p>',
            f'<p style="color: white;"><b style="color: #3e8ad2;">Abstract: </b>{project_info[0][1]} </p>',
            f'<p style="color: white;"><b style="color: #3e8ad2;">Fondi: </b>{project_info[0][2]} </p>',
            f'<p style="color: white;"><b style="color: #3e8ad2;">Data di Inizio: </b>{project_info[0][3]} </p>',
            f'<p style="color: white;"><b style="color: #3e8ad2;">Data di Fine: </b>{project_info[0][4]} </p>',
            f'<p style="color: white;"><b style="color: #3e8ad2;">Finanziatore: </b>{project_info[0][5]} </p>',
            f'<p style="color: white;"><b style="color: #3e8ad2;">Gruppo di Finanziamento: </b>{project_info[0][6]} </p>',
            f'<p style="color: white;"><b style="color: #3e8ad2;">Numero di pubblicazioni: </b>{project_info[0][7].count("pub.")} </p>',
            f'<p style="color: white;"><b style="color: #3e8ad2;">Programma: </b>{project_info[0][8]} </p>',
            f'<p style="color: white;"><b style="color: #3e8ad2;">Dimensions URL: </b><a href="{project_info[0][9]}">{project_info[0][9]}</a></p>' if
            project_info[0][9] != "Non Definito"
            else f'<p style="color: white;"><b style="color: #3e8ad2;">Dimensions URL: </b>{project_info[0][9]} </p>',
            f'<p style="color: white;"><b style="color: #3e8ad2;">Link: </b><a href="{project_info[0][10]}">{project_info[0][10]}</a></p>' if
            project_info[0][10] != "Non Definito"
            else f'<p style="color: white;"><b style="color: #3e8ad2;">Link: </b>{project_info[0][10]} </p>'
        ),
        unsafe_allow_html=True
    )

nodes_project = []
edges_project = []
ids_project = []

# Definizione dei colori e delle etichette della legenda
colors_project = ['#3e8ad2', 'yellow', 'orange']
labels_project = ['Ricercatore', 'Progetto', 'Organizzazioni']

# Layout a due colonne
col7, col8 = st.columns([1, 3])

with col7:
    # Creazione della legenda
    st.subheader('Legenda')

    for color, label in zip(colors_project, labels_project):
        st.markdown(f'<span style="color:{color}">●</span> {label}', unsafe_allow_html=True)

with col8:
    query = f"""MATCH (r:Researcher)-[]->(p:Project)<-[]-(o:Organization)
                WHERE p.Title = "{selected_project_micro_name}"
                RETURN r AS Ricercatore, p AS Progetto, o AS Organizzazione
            """

    query_results = conn.query(query)
    results = [(record['Ricercatore'], record['Progetto'], record['Organizzazione']) for record in query_results]

    for record in results:
        researcher = record[0]
        if researcher.element_id not in ids_project:
            ids_project.append(researcher.element_id)
            nodes_project.append(Node(id=researcher.element_id,
                                      title=researcher["Name"],
                                      size=10,
                                      color='#3e8ad2')
                                 )
        project = record[1]
        if project.element_id not in ids_project:
            ids_project.append(project.element_id)
            nodes_project.append(Node(id=project.element_id,
                                      title=project["Title"],
                                      size=15,
                                      color='yellow')
                                 )

        edges_project.append(Edge(source=researcher.element_id,
                                  label="Ricerca",
                                  target=project.element_id,
                                  color='grey',
                                  font={'size': 10}
                                  )
                             )

        organization = record[2]
        if organization.element_id not in ids_project:
            ids_project.append(organization.element_id)
            nodes_project.append(Node(id=organization.element_id,
                                      title=organization["Name"],
                                      size=8,
                                      color='orange')
                                 )

        edges_project.append(Edge(source=organization.element_id,
                                  label="Finanzia",
                                  target=project.element_id,
                                  color='grey',
                                  font={'size': 10}
                                  )
                             )

    agraph(nodes=nodes_project, edges=edges_project, config=config)

# Explicitly close the connection
conn.close()
