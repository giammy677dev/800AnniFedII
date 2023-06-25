from utils import st, conn

st.set_page_config(
    page_title="Homepage",
    page_icon="️🏠",
    layout="wide"
)

def elenco_bullet(testo_grassetto, testo_normale):
    st.markdown(f"- <span style='color:#3e8ad2'><b>{testo_grassetto}</b></span>: {testo_normale}",
                unsafe_allow_html=True)

# Immagine di copertina
st.image('LogoUniRemakeWhite.png', use_column_width=True)

st.title("HomePage - 800 anni Federico II")
st.header("Dashboard di analitiche sui progetti ️")
st.write("""bla bla blaa""")

st.write("""Le analitiche effettuate che ritroviamo in questa dashboard sono le seguenti:""")

# Elenco con bullet list e link ai collegamenti delle sezioni
elenco_bullet("Prima Analitica", "Presentazione pagina analitiche")
elenco_bullet("Seconda Analitica", "Presentazione pagina analitiche")
elenco_bullet("Terza Analitica", "Presentazione pagina analitiche")
