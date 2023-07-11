from utils import st, conn, px, pd
import plotly.graph_objects as go

st.set_page_config(
    page_title="Eccellenze",
    page_icon="️🎖️",
    layout="wide"
)

st.title("🎖️Eccellenze")
st.write("Piccola introduzione")

# Definisci il valore minimo e massimo dell'intervallo per lo slider
year_range = st.slider("Seleziona il periodo temporale da analizzare", min_value=1970, max_value=2023, value=(2010, 2023))

# Estrai i valori minimo e massimo selezionati dallo slider
min_year = year_range[0]
max_year = year_range[1]

# Query per conteggio macro-categorie
query = f"""MATCH (p:Project)-[r]-(f:Field)
            WHERE SIZE(TRIM(f.Field_Code)) = 2
            AND toInteger(p.Start_Year) >= {min_year}
            AND toInteger(p.Start_Year) <= {max_year}
            RETURN f.Name, COUNT(p) AS Conteggio
            ORDER BY Conteggio DESC
        """
query_results = conn.query(query)

# Creare un dataframe dai risultati della query
df = pd.DataFrame(query_results, columns=['Field_Name', 'Conteggio'])

# Creare il grafico a barre orizzontale
count_anno_chart = px.bar(df, y='Field_Name', x='Conteggio', color='Conteggio', color_continuous_scale='Turbo', orientation='h', labels={'Conteggio': 'Numero Progetti', 'Field_Name' : 'Macro-settore'})

# etichette del grafico
count_anno_chart.update_layout(
    xaxis_title='Numero Progetti',
    yaxis_title='Macro-settore',
    yaxis={'categoryorder': 'total ascending'}
)

st.plotly_chart(count_anno_chart, use_container_width=True)

#Select box per filtrare per macrocategoria
query = """MATCH (f:Field) WHERE SIZE(TRIM(f.Field_Code)) = 2 AND f.Name <> 'NaN'
            RETURN toInteger(f.Field_Code) AS Field_Code, f.Name AS Name
            """
query_results = conn.query(query)
macro_fields_results = [(record['Field_Code'], record['Name']) for record in query_results]
selected_macro_name = st.selectbox('Seleziona il macro-settore:', [name[1] for name in macro_fields_results])

# Trova il Field_Code corrispondente al campo selezionato
selected_macro_code = None
for record in macro_fields_results:
    if record[1] == selected_macro_name:
        selected_macro_code = record[0]

# Query per conteggio micro-categorie
query = f"""MATCH (p:Project)-[r]-(f:Field)
            WHERE f.Field_Code =~ '^{selected_macro_code}\\d{{2}}$'
            AND toInteger(p.Start_Year) >= {min_year}
            AND toInteger(p.Start_Year) <= {max_year}
            RETURN f.Name, COUNT(p) AS Conteggio
            ORDER BY Conteggio DESC
        """
query_results = conn.query(query)

# Creare un dataframe dai risultati della query
df = pd.DataFrame(query_results, columns=['Field_Name', 'Conteggio'])

# Creare il grafico a barre
count_anno_chart = px.bar(df, y='Field_Name', x='Conteggio', color='Conteggio', color_continuous_scale='Turbo', orientation='h', labels={'Conteggio': 'Numero Progetti', 'Field_Name' : 'Settore'})

# ordine del grafico
count_anno_chart.update_layout(
    yaxis={'categoryorder': 'total ascending'}
)

st.plotly_chart(count_anno_chart, use_container_width=True)

st.write("----------------------------------------------------")
st.header("Trend nel tempo")
st.write("Piccola presentazione")


# Query per conteggio macro-categorie nel tempo
query = """MATCH (p:Project)-[r]-(f:Field)
            WHERE SIZE(TRIM(f.Field_Code)) = 2
            RETURN f.Name, COUNT(p) AS projectCount, toInteger(p.Start_Year) AS Anno
            ORDER BY Anno DESC
        """

# Esecuzione della query
query_results = conn.query(query)

# Creare un dataframe dai risultati della query
df = pd.DataFrame(query_results, columns=['Field_Name', 'Conteggio', 'Anno'])

# Creare un grafico vuoto
fig = go.Figure()

# Ottenere i nomi unici dei campi
field_names = df['Field_Name'].unique()

# Aggiungere una linea per ogni campo
for field in field_names:
    df_field = df[df['Field_Name'] == field]

    # Togliere la visibilità di alcune macro-categorie per una migliore visualizzazione
    if field in ['Mathematical Sciences', 'Law and Legal Studies', 'Health Sciences', 'Psychology', 'Economics', 'Education', 'Philosophy and Religious Studies', 'Language, Communication and Culture', 'Creative Arts and Writing', 'Commerce, Management, Tourism and Services' ,'Agricultural, Veterinary and Food Sciences', 'Built Environment and Design', 'Earth Sciences', 'Environmental Sciences' ,'History, Heritage and Archaeology' ]:
        fig.add_trace(go.Scatter(x=df_field['Anno'], y=df_field['Conteggio'], name=field, visible='legendonly'))
    else:
        fig.add_trace(go.Scatter(x=df_field['Anno'], y=df_field['Conteggio'], name=field))

# Mostra il grafico in Streamlit
st.plotly_chart(fig, use_container_width=True)

st.write("-------------------------------------------------------")
st.header("Healthcare")
st.write("Introduzione ai progetti Healthcare")

# Esecuzione della query per numero totale progetti
numeroTotProgetti = conn.query("""MATCH (p:Project) RETURN count(*)""")
numeroTotProgetti = numeroTotProgetti[0][0]

# Esecuzione della query per numero totale progetti in campo medico
query = """MATCH (p:Project)
            WHERE p.HRCS_HC_Categories <> 'NaN'
            OR  p.HRCS_RAC_Categories <> 'NaN'
            OR  p.CSO_Categories <> 'NaN'
            OR  p.Cancer_Types <> 'NaN'
            RETURN count(*)
        """
numeroTotProgettiCampoMedico = conn.query(query)
numeroTotProgettiCampoMedico = numeroTotProgettiCampoMedico[0][0]

# Esecuzione della query per numero totale progetti sul cancro
numeroProgettiSulCancro = conn.query("""MATCH (p:Project) WHERE p.Cancer_Types <> 'NaN' RETURN count(*)""")
numeroProgettiSulCancro = numeroProgettiSulCancro[0][0]

# Layout a due colonne
col1, col2 = st.columns([1, 1])

with col1:
    labels = ['Altri Progetti', 'Progetti Healthcare']
    values = [numeroTotProgetti-numeroTotProgettiCampoMedico, numeroTotProgettiCampoMedico]
    explode1 = (0, 0.1)  # Esplosione della seconda fetta (Campo Medico)

    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.5)])
    fig.update_layout(title="Percentuale Progetti Healthcare")

    st.plotly_chart(fig, use_container_width=True)

with col2:
    labels = ['Progetti Healthcare', 'Progetti specifici sul Cancro']
    values = [numeroTotProgettiCampoMedico-numeroProgettiSulCancro, numeroProgettiSulCancro]
    explode2 = (0, 0.1)  # Esplosione della seconda fetta (Progetti sul Cancro)

    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.5)])
    fig.update_layout(title="Percentuale Progetti sul Cancro")

    st.plotly_chart(fig, use_container_width=True)


query = """MATCH (p:Project)
            WHERE p.Cancer_Types <> 'NaN' AND p.Cancer_Types <> 'Not Site-Specific Cancer'
            WITH split(p.Cancer_Types, ';') AS cancerTypes
            UNWIND cancerTypes AS cancerType
            RETURN cancerType, COUNT(*) AS Conteggio
            ORDER BY Conteggio DESC
        """

# Esecuzione della query
query_results = conn.query(query)

# Creare un dataframe dai risultati della query
df = pd.DataFrame(query_results, columns=['cancerType', 'Conteggio'])

# Creare il grafico a barre
count_anno_chart = px.bar(df, y='cancerType', x='Conteggio', color='Conteggio', color_continuous_scale='Turbo', orientation='h', labels={'Conteggio': 'Numero Progetti', 'cancerType' : 'Tipologia Cancro'})

# ordine del grafico
count_anno_chart.update_layout(
    yaxis={'categoryorder': 'total ascending'}
)


st.plotly_chart(count_anno_chart, use_container_width=True)

query = """MATCH (p:Project)
            WHERE p.Cancer_Types <> 'NaN' AND p.Cancer_Types <> 'Not Site-Specific Cancer'
            WITH split(p.Cancer_Types, ';') AS cancerTypes, p
            UNWIND cancerTypes AS cancerType
            RETURN cancerType, sum(toInteger(p.Funding)) AS TotalFunding
            ORDER BY TotalFunding DESC
        """

# Esecuzione della query
query_results = conn.query(query)

# Creare un dataframe dai risultati della query
df = pd.DataFrame(query_results, columns=['cancerType', 'TotalFunding'])

# Creare il grafico a barre
count_anno_chart = px.bar(df, y='cancerType', x='TotalFunding', color='TotalFunding', color_continuous_scale='Turbo', orientation='h', labels={'TotalFunding': 'Fondi Investiti (€)', 'cancerType' : 'Tipologia Cancro'})

# ordine del grafico
count_anno_chart.update_layout(
    yaxis={'categoryorder': 'total ascending'}
)

# Aggiunta del simbolo dell'euro alle etichette sui fondi
count_anno_chart.update_xaxes(ticksuffix='€')

# Modifiche etichette visualizzabili passando col mouse sulla barra
count_anno_chart.update_traces(
    customdata=df['TotalFunding'],
    hovertemplate=' %{customdata} €',
)

# Aggiunta del simbolo dell'euro alle etichette del riferimento dei colori
count_anno_chart.update_coloraxes(colorbar=dict(ticksuffix='€'))

st.plotly_chart(count_anno_chart, use_container_width=True)

st.write("-------------------------------------------------------")
st.header("Progetti sulla sostenibilità ambientale")
st.write("Intro")

query = """MATCH (p:Project)
            WHERE p.Sustainable_Goals <> 'NaN'
            WITH split(p.Sustainable_Goals, '; ') AS goals
            UNWIND goals AS goal
            WITH split(goal, ' ') AS goalParts, goal
            WITH substring(goal, size(goalParts[0]) + 1) AS Etichetta
            RETURN Etichetta, COUNT(*) AS Conteggio
            ORDER BY Conteggio DESC
        """

# Esecuzione della query
query_results = conn.query(query)

# Creare un dataframe dai risultati della query
df = pd.DataFrame(query_results, columns=['Etichetta', 'Conteggio'])

# Creare il grafico a barre
count_anno_chart = px.bar(df, y='Etichetta', x='Conteggio', color='Conteggio', color_continuous_scale='Turbo', orientation='h',labels={'Conteggio': 'Numero Progetti', 'Etichetta' : 'Campo di Interesse'})

# ordine del grafico
count_anno_chart.update_layout(
    yaxis={'categoryorder': 'total ascending'}
)

st.plotly_chart(count_anno_chart, use_container_width=True)

query = """MATCH (p:Project)
            WHERE p.Sustainable_Goals <> 'NaN'
            WITH split(p.Sustainable_Goals, '; ') AS goals, p
            UNWIND goals AS goal
            WITH split(goal, ' ') AS goalParts, goal, p
            WITH substring(goal, size(goalParts[0]) + 1) AS Etichetta, p
            RETURN Etichetta, SUM(toInteger(p.Funding)) AS Totale_Fondi
            ORDER BY Totale_Fondi DESC
        """

# Esecuzione della query
query_results = conn.query(query)

# Creare un dataframe dai risultati della query
df = pd.DataFrame(query_results, columns=['Etichetta', 'Totale_Fondi'])

# Creare il grafico a barre
count_anno_chart = px.bar(df, y='Etichetta', x='Totale_Fondi', color='Totale_Fondi', color_continuous_scale='Turbo', orientation='h', labels={'Totale_Fondi': 'Fondi Investiti (€)', 'Etichetta' : 'Campo di Interesse'})

# Aggiunta del simbolo dell'euro alle etichette sui fondi
count_anno_chart.update_xaxes(ticksuffix='€')

# Modifiche etichette visualizzabili passando col mouse sulla barra
count_anno_chart.update_traces(
    customdata=df['Totale_Fondi'],
    hovertemplate=' %{customdata} €',
)

# ordine del grafico
count_anno_chart.update_layout(
    yaxis={'categoryorder': 'total ascending'}
)

# Aggiunta del simbolo dell'euro alle etichette del riferimento dei colori
count_anno_chart.update_coloraxes(colorbar=dict(ticksuffix='€'))

st.plotly_chart(count_anno_chart, use_container_width=True)

# Chiudiamo la connessione al DB
conn.close()
