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
elenco_bullet("Campi di Ricerca", """in questa sezione riportiamo una serie di informazioni rispetto ai settori in
                                    cui l'Università Federico II ha finanziato o co-finanziato progetti di ricerca. Inoltre,
                                    viene riportata anche una mappa interattiva che mostra le collaborazioni effettuate
                                    dall'Università Federico II con Università da tutto il mondo. Infine, viene riportato
                                    anche un approfondimento grafico relativo ai singoli progetti del settore
                                    selezionato;
                                    """)
elenco_bullet("Mappa di Sintesi", """in questa sezione riportiamo un grafo in cui viene mostrato visivamente un riassunto
                                  riguardo uno specifico macro-settore, evidenziando i collegamenti che intercorrono 
                                  con i settori ed i progetti relativi.
                                  """)

st.divider()

# Primo div con testo all'estrema sinistra
st.markdown("<div style='text-align: left; width: 45%; float: left; margin-left: 25px'>"
            "<p><b style='color: #3e8ad2;'>Ing.</b> Orlando Gian Marco</p>"
            "<p><b style='color: #3e8ad2;'>Ing.</b> Perillo Marco</p>"
            "<p><b style='color: #3e8ad2;'>Ing.</b> Russo Diego</p>"
            "</div>"
            "<div style='text-align: right; width: 45%; float: right; margin-right: 25px'>"
            "<p><b style='color: #3e8ad2;'>Prof.</b> Moscato Vincenzo</p>"
            "<p><b style='color: #3e8ad2;'>Dott.</b> Ferrara Antonino</p>"
            "<p><b style='color: #3e8ad2;'>Dott.</b> Galli Antonio</p>"
            "<p><b style='color: #3e8ad2;'>Dott.</b> La Gatta Valerio</p>"
            "<p><b style='color: #3e8ad2;'>Dott.</b> Postiglione Marco</p>"
            "</div>",
            unsafe_allow_html=True)


