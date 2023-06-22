from utils import st, conn
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Eccellenze",
    page_icon="️📊",
    layout="wide"
)

st.title("Eccellenze")
st.write("Piccola introduzione")

# Definisci il valore minimo e massimo dell'intervallo per lo slider
year_range = st.slider("Seleziona l'intervallo di anni", min_value=1970, max_value=2023, value=(2010, 2023))

# Estrai i valori minimo e massimo selezionati dall'intervallo
min_year = year_range[0]
max_year = year_range[1]

# Query per conteggio macro-categorie
query = f"""MATCH (p:Project)-[r]-(f:Field)
where SIZE(TRIM(f.Field_Code)) = 2 AND toInteger(p.Start_Year) >= {min_year} AND toInteger(p.Start_Year) <= {max_year}
RETURN f.Name, COUNT(p) AS Conteggio
ORDER BY Conteggio DESC
        """
query_results = conn.query(query)

# Creare un dataframe dai risultati della query
df = pd.DataFrame(query_results, columns=['Field_Name', 'Conteggio'])

# Creare il grafico a barre
count_anno_chart = px.bar(df, x='Field_Name', y='Conteggio', color='Conteggio', color_continuous_scale='Viridis')

st.plotly_chart(count_anno_chart, use_container_width=True)


# Query per conteggio micro-categorie
query = """MATCH (p:Project)-[r]-(f:Field)
where SIZE(TRIM(f.Field_Code)) > 2
RETURN f.Name, COUNT(p) AS Conteggio
ORDER BY Conteggio DESC
        """
query_results = conn.query(query)

# Creare un dataframe dai risultati della query
df = pd.DataFrame(query_results, columns=['Field_Name', 'Conteggio'])

# Creare il grafico a barre
count_anno_chart = px.bar(df, x='Field_Name', y='Conteggio', color='Conteggio', color_continuous_scale='Viridis')

st.plotly_chart(count_anno_chart, use_container_width=True)

st.write("----------------------------------------------------")

st.header("Trend nel tempo")
st.write("Piccola presentazione")


# Query per conteggio micro-categorie
query = """MATCH (p:Project)-[r]-(f:Field)
where SIZE(TRIM(f.Field_Code)) = 2
RETURN f.Name, COUNT(p) AS projectCount, toInteger(p.Start_Year) AS Anno
ORDER BY Anno DESC"""

# Esegui la query per ottenere i risultati
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

    # Se il campo è uno di quelli che vuoi nascondere all'inizio, imposta visible='legendonly'
    if field in ['Mathematical Sciences', 'Law and Legal Studies', 'Health Sciences', 'Psychology', 'Economics', 'Education', 'Philosophy and Religious Studies', 'Language, Communication and Culture', 'Creative Arts and Writing', 'Commerce, Management, Tourism and Services' ,'Agricultural, Veterinary and Food Sciences', 'Built Environment and Design', 'Earth Sciences', 'Environmental Sciences' ,'History, Heritage and Archaeology' ]:
        fig.add_trace(go.Scatter(x=df_field['Anno'], y=df_field['Conteggio'], name=field, visible='legendonly'))
    else:
        fig.add_trace(go.Scatter(x=df_field['Anno'], y=df_field['Conteggio'], name=field))

# Mostra il grafico in Streamlit
st.plotly_chart(fig, use_container_width=True)

st.write("-------------------------------------------------------")
st.header("Progetti sul cancro")
st.write("Intro")


# Esegui la query per numero totale progetti
numeroTotProgetti = conn.query("""MATCH (p:Project) RETURN count(*)""")
numeroTotProgetti = numeroTotProgetti[0][0]

# Esegui la query per numero totale progetti in campo medico
query = """MATCH (p:Project)
WHERE p.HRCS_HC_Categories <> 'NaN' OR  p.HRCS_RAC_Categories <> 'NaN' OR  p.CSO_Categories <> 'NaN' OR  p.Cancer_Types <> 'NaN'
RETURN count(*)"""
numeroTotProgettiCampoMedico = conn.query(query)
numeroTotProgettiCampoMedico = numeroTotProgettiCampoMedico[0][0]

# Esegui la query per numero totale progetti sul cancro
numeroProgettiSulCancro = conn.query("""MATCH (p:Project) WHERE p.Cancer_Types <> 'NaN' RETURN count(*)""")
numeroProgettiSulCancro = numeroProgettiSulCancro[0][0]

# Layout a due colonne
col1, col2 = st.columns([1, 1])

with col1:
    labels = ['Altro', 'Campo Medico']
    values = [numeroTotProgetti-numeroTotProgettiCampoMedico, numeroTotProgettiCampoMedico]
    explode = (0, 0.1)  # Esplosione della seconda fetta (Campo Medico)

    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.5)])
    fig.update_layout(title="Percentuale Progetti Medici")

    st.plotly_chart(fig, use_container_width=True)

with col2:
    labels = ['Campo Medico', 'Specifici sul Cancro']
    values = [numeroTotProgettiCampoMedico-numeroProgettiSulCancro, numeroProgettiSulCancro]
    explode = (0, 0.1)  # Esplosione della seconda fetta (Progetti sul Cancro)

    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.5)])
    fig.update_layout(title="Percentuale Progetti sul Cancro")

    st.plotly_chart(fig, use_container_width=True)


query = """MATCH (p:Project)
WHERE p.Cancer_Types <> 'NaN' AND p.Cancer_Types <> 'Not Site-Specific Cancer'
WITH split(p.Cancer_Types, ';') AS cancerTypes
UNWIND cancerTypes AS cancerType
RETURN cancerType, COUNT(*) AS Conteggio
ORDER BY Conteggio DESC"""

# Esegui la query per ottenere i risultati
query_results = conn.query(query)

# Creare un dataframe dai risultati della query
df = pd.DataFrame(query_results, columns=['cancerType', 'Conteggio'])

# Creare il grafico a barre
count_anno_chart = px.bar(df, x='cancerType', y='Conteggio', color='Conteggio', color_continuous_scale='Viridis')

st.plotly_chart(count_anno_chart, use_container_width=True)


query = """MATCH (p:Project)
WHERE p.Cancer_Types <> 'NaN' AND p.Cancer_Types <> 'Not Site-Specific Cancer'
WITH split(p.Cancer_Types, ';') AS cancerTypes, p
UNWIND cancerTypes AS cancerType
RETURN cancerType, sum(toInteger(p.Funding)) AS TotalFunding
ORDER BY TotalFunding DESC"""

# Esegui la query per ottenere i risultati
query_results = conn.query(query)

# Creare un dataframe dai risultati della query
df = pd.DataFrame(query_results, columns=['cancerType', 'TotalFunding'])

# Creare il grafico a barre
count_anno_chart = px.bar(df, x='cancerType', y='TotalFunding', color='TotalFunding', color_continuous_scale='Viridis')

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
ORDER BY Conteggio DESC"""

# Esegui la query per ottenere i risultati
query_results = conn.query(query)

# Creare un dataframe dai risultati della query
df = pd.DataFrame(query_results, columns=['Etichetta', 'Conteggio'])

# Creare il grafico a barre
count_anno_chart = px.bar(df, x='Etichetta', y='Conteggio', color='Conteggio', color_continuous_scale='Viridis')

st.plotly_chart(count_anno_chart, use_container_width=True)

query = """MATCH (p:Project)
WHERE p.Sustainable_Goals <> 'NaN'
WITH split(p.Sustainable_Goals, '; ') AS goals, p
UNWIND goals AS goal
WITH split(goal, ' ') AS goalParts, goal, p
WITH substring(goal, size(goalParts[0]) + 1) AS Etichetta, p
RETURN Etichetta, SUM(toInteger(p.Funding)) AS Totale_Fondi
ORDER BY Totale_Fondi DESC"""

# Esegui la query per ottenere i risultati
query_results = conn.query(query)

# Creare un dataframe dai risultati della query
df = pd.DataFrame(query_results, columns=['Etichetta', 'Totale_Fondi'])

# Creare il grafico a barre
count_anno_chart = px.bar(df, x='Etichetta', y='Totale_Fondi', color='Totale_Fondi', color_continuous_scale='Viridis')

st.plotly_chart(count_anno_chart, use_container_width=True)
