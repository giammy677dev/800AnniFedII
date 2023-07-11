from utils import st

st.set_page_config(
    page_title="Homepage",
    page_icon="🏠",
    layout="wide"
)


def elenco_bullet(testo_grassetto, testo_normale):
    st.markdown(f"- <span style='color:#3e8ad2'><b>{testo_grassetto}</b></span>: {testo_normale}",
                unsafe_allow_html=True)

# Immagine di copertina
st.image('utility/FrontoneUniBordo.jpg', use_column_width=True)

st.title("HomePage - 800 anni Federico II")
st.write("""Ottocento anni di formazione, tradizione, innovazione, crescita. In questa dashboard, realizzata a seguito del
            corso di Big Data, cerchiamo di sintetizzare il contributo in termini di progetti di ricerca che l'Università
            degli Studi di Napoli Federico II ha finanziato o co-finanziato in collaborazione con altre Università da
            tutto il mondo.
        """)

st.write("""La dashboard è organizzata nelle seguenti tre sezioni:""")

# Elenco con bullet list e link ai collegamenti delle sezioni
elenco_bullet("Eccellenze", """in questa sezione riportiamo tutti i campi di ricerca in cui l'Università Federico II si
                            è distinta. Vengono riportati diversi grafici relativi al numero di progetti ed ai fondi
                            investiti, suddivisi per macro-settori disciplinari. Inoltre, viene riportato anche un
                            approfondimento riguardo due tematiche molto attuali quali la ricerca sul cancro e la ricerca
                            riguardo la sostenibilità ambientale;
                            """)
elenco_bullet("Campi di Ricerca", """in questa sezione riportiamo una serie di informazioni rispetto ai micro-settori in
                                    cui l'Università Federico II ha finanziato o co-finanziato progetti di ricerca. Inoltre,
                                    viene riportata anche una mappa interattiva che mostra le collaborazioni effettuate
                                    dall'Università Federico II con Università da tutto il mondo;
                                    """)
elenco_bullet("Mappa di Sintesi", """in questa sezione riportiamo un grafo in cui viene mostrato visivamente un riassunto
                                  riguardo uno specifico macro-settore, evidenziando i collegamenti che intercorrono 
                                  con i micro-settori ed i progetti relativi.
                                  """)

st.divider()

# Layout a due colonne
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown(f"<span style='color:#3e8ad2'><b>Ing.</b></span> Orlando Gian Marco", unsafe_allow_html=True)
    st.markdown(f"<span style='color:#3e8ad2'><b>Ing.</b></span> Perillo Marco", unsafe_allow_html=True)
    st.markdown(f"<span style='color:#3e8ad2'><b>Ing.</b></span> Russo Diego", unsafe_allow_html=True)
with col2:
    st.markdown(f"<span style='color:#3e8ad2; text-align:right'><b>Prof.</b></span> Moscato Vincenzo",
                unsafe_allow_html=True)
    st.markdown(f"<span style='color:#3e8ad2; text-align:right'><b>Dott.</b></span> Ferrara Antonino",
                unsafe_allow_html=True)
    st.markdown(f"<span style='color:#3e8ad2; text-align:right'><b>Dott.</b></span> Galli Antonio",
                unsafe_allow_html=True)
    st.markdown(f"<span style='color:#3e8ad2; text-align:right'><b>Dott.</b></span> La Gatta Valerio",
                unsafe_allow_html=True)
    st.markdown(f"<span style='color:#3e8ad2; text-align:right'><b>Dott.</b></span> Postiglione Marco",
                unsafe_allow_html=True)
