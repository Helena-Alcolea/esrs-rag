"""
ESRS-RAG — interactive showcase.

ONE SECTION ONLY: LIVE RETRIEVAL. It is lightweight (only the question is
encoded; the corpus vectors are pre-computed) -> it fits in a free deployment,
and it is the part a visitor can actually play with.

The LLM (Ollama, local) does NOT run here: hosting it would break the "100%
on-premise" guarantee the project is built on. The complete system — corpus,
embeddings and LLM, all local — is the one you run on your own machine, and it
is the one every evaluation number comes from.

DESIGN: "Broadsheet" (light, primary) and "Nocturne" (dark), the two systems in
the design document. Both palettes live in a SINGLE CSS block written as
`light-dark(light, dark)`: Streamlit sets `color-scheme` on the app container,
so the browser resolves each pair on its own and EVERYTHING switches at once
when the theme changes (⋮ -> Settings).
    ⚠️ Do NOT read the theme from Python (`st.context.theme`): Streamlit's own
    documentation warns the value is wrong precisely while the user is changing
    the theme in the menu -> half the page stayed light.

Run locally:  uv run streamlit run app/streamlit_app.py
"""

import html
import json
import time
from pathlib import Path

import numpy as np
import streamlit as st

DATA_DIR = Path(__file__).resolve().parent / "datos"
MODEL = "intfloat/multilingual-e5-base"
DEFAULT_K = 8  # the system's frozen configuration (k sweep, inverted U)
K_OPTIONS = [3, 5, 8, 10, 15, 20]
VISIBLE = 4  # paragraphs shown expanded; the rest go into ONE expander

# ── When to say "no reliable source" ────────────────────────────────────────
# Retrieval NEVER abstains: it always returns the k nearest paragraphs. Deciding
# that none of them is worth showing needs a signal, and the obvious one does
# not work: measured over 28 queries, a keyboard mash scores HIGHER (83.2%) than
# a perfectly formed question the corpus cannot answer (73.7%). So there are two
# guards, each aimed at what it can actually detect:
#   1. IS THIS EVEN LANGUAGE? The tokenizer answers for free: gibberish shatters
#      into subword pieces ("jdfgsdnfgkjgnkfgdufg" -> 17 pieces for one word).
#      ⚠️ MEASURED LIMITATION — this statistic is a MEAN per word, so it only
#      behaves on MULTI-WORD queries. On a single word nothing dilutes the
#      count, and the corpus's own long terms break it: "biodiversidad" and
#      "biodiversity" 3.0, "sostenibilidad" 3.0, "taxonomía" /
#      "descarbonización" / "microplásticos" 4.0. The first version cut at 2.5
#      and flagged every one of them. (The claim it rested on — "no real query
#      exceeded 2.0" — was true of SENTENCES only: the 28-query dev sample
#      contained no single-word queries at all.) The cut-off is now 6.0: clear
#      of real language (max 4.0), well under the long mash (17.0).
#      → So this guard catches LONG mashes ONLY, and that is declared, not
#      hidden. Short ones ("aaaa" 2.0, "qwerty" 3.0) are NOT separable from the
#      norm's own codes, which are legitimate queries ("E1" 2.0, "S1-14" 3.0);
#      a characters-per-piece statistic does not rescue it either — it fails on
#      exactly those codes instead (dev sample of 33 real + 15 gibberish
#      queries; the two classes overlap under both statistics). Two heuristics
#      measured, both rejected — the same verdict the similarity threshold got
#      when it was tried as an abstention signal.
#   2. IS ANYTHING CLOSE? Among queries that ARE language, in-corpus ones scored
#      83.5-91.9% and out-of-corpus ones 73.7-80.1% — separable, unlike the
#      gibberish case.
# ⚠️ Both cut-offs were tuned on small dev samples, NOT validated on the
# evaluation set. They soften the display; they never delete a result:
# the paragraphs stay one click away.
MAX_PIECES_PER_WORD = 6.0
MIN_TOP_SIMILARITY = 0.82

st.set_page_config(page_title="ESRS·RAG", page_icon="🌿", layout="wide")


# ─────────────────────── UI strings (one source per language) ───────────────
T = {
    "es": dict(
        nav_search="Buscar", nav_corpus="Corpus",
        dl_reg="Reglamento Delegado (UE) 2023/2772", dl_onprem="En local",
        dl_priv="Nada sale de la máquina",
        hero_a="Pregunta a la norma.", hero_b="Obtén el párrafo.",
        hero_sub=("Cada resultado se rastrea hasta un párrafo numerado del Anexo ESRS, "
                  "con la similitud que lo situó ahí. Pregunta en español o en inglés — "
                  "el modelo de embeddings es multilingüe."),
        placeholder="p. ej. ¿Qué debo declarar sobre la contaminación del aire?",
        retrieve="Recuperar", or_try="O prueba", depth="Profundidad de recuperación",
        examples=["Límites del alcance 3", "¿Cómo se evalúa la doble materialidad?",
                  "Uso de agua en zonas con estrés hídrico", "Formación anticorrupción"],
        brief_kicker="Resumen de recuperación",
        m_par="párrafos", m_src="fuentes", m_top="similitud máxima", m_cpu="en CPU",
        coverage="Cobertura", coverage_none="sin resultados",
        para_kicker="Los párrafos · literales, ordenados por similitud coseno",
        export="Exportar párrafos (.md)",
        more="{n} párrafos más — {span}",
        corpus_kicker="El corpus · un fragmento = un párrafo numerado",
        col_std="Norma", col_share="Peso", total="Total indexado", hits="resultados",
        weak_kicker="Sin fuente fiable",
        weak_gibberish="Eso no parece una consulta.",
        weak_far="Nada del corpus se acerca lo suficiente.",
        weak_body=("La recuperación **nunca se abstiene**: siempre devuelve los párrafos más "
                   "cercanos, haya algo relevante o no. Por eso el porcentaje **ordena**, no "
                   "califica — con una consulta sin sentido los 1.700 párrafos se agolpan en una "
                   "banda estrecha y el primero no significa nada. El corpus es solo el Anexo "
                   "ESRS: no contiene datos de empresa, ni legislación nacional, ni guías."),
        weak_show="Ver de todos modos los párrafos más cercanos",
        empty_kicker="Vacío · primer uso", empty_title="Aún no se ha recuperado nada.",
        empty_body=("1.700 párrafos del Anexo ESRS están indexados y esperando. "
                    "Formula una pregunta de reporting en cualquiera de los dos idiomas."),
        r_step1="Pregunta codificada · e5-base, 768d",
        r_step2="Puntuando 1.700 vectores", r_step3="Componiendo extractos",
        r_meta="solo CPU · sin llamadas de red",
        ar="Requisito de aplicación", glossary="Glosario",
        disclaimer=("Proyecto en curso — no constituye asesoramiento jurídico. "
                    "El texto auténtico es el del Diario Oficial."),
        tech="ⓘ  Ficha técnica",
        tech_body=(
            "**Corpus** — Anexo del ESRS (12 normas), indexado como **1.700** párrafos "
            "numerados, entradas de glosario y filas de apéndice. Un fragmento = un párrafo.\n\n"
            "**Recuperación** — `intfloat/multilingual-e5-base` (768 dimensiones, ventana de "
            "512 tokens), en local sobre CPU. Similitud coseno exacta sobre vectores "
            "precalculados.\n\n"
            "**Generación** — `llama3.2:3b` mediante **Ollama**, `temperature=0`, sin conexión. "
            "El modelo **no** se hospeda aquí: hacerlo rompería la garantía de que nada sale "
            "de la máquina.\n\n"
            "⚠️ **Esto es el demo hospedado, y solo ejecuta la RECUPERACIÓN.** El sistema "
            "completo — corpus, embeddings y LLM, todo en local, sin que ningún documento "
            "salga de la máquina — es el que se ejecuta en tu propio equipo, y es del que "
            "salen todos los números de la evaluación.\n\n"
            "La recuperación y la generación se evalúan **por separado**, contra un baseline "
            "aleatorio y otro de BM25, sobre 45 preguntas verificadas a mano contra el párrafo "
            "de origen.\n\n"
            "_Tema claro / oscuro: menú ⋮ → Settings._"
        ),
    ),
    "en": dict(
        nav_search="Search", nav_corpus="Corpus",
        dl_reg="Delegated Regulation (EU) 2023/2772", dl_onprem="On-premise",
        dl_priv="Nothing leaves the machine",
        hero_a="Ask the standard.", hero_b="Get the paragraph.",
        hero_sub=("Every result is traced to a numbered paragraph of the ESRS Annex, "
                  "with the similarity score that put it there. Ask in Spanish or "
                  "English — the embedding model is multilingual."),
        placeholder="e.g. What must I disclose about air pollution?",
        retrieve="Retrieve", or_try="Or try", depth="Retrieval depth",
        examples=["Scope 3 boundary rules", "How is double materiality assessed?",
                  "Water use in stressed areas", "Anti-corruption training coverage"],
        brief_kicker="Retrieval brief",
        m_par="paragraphs", m_src="sources", m_top="top similarity", m_cpu="on CPU",
        coverage="Coverage", coverage_none="not touched",
        para_kicker="The paragraphs · verbatim, ranked by cosine similarity",
        export="Export paragraphs (.md)",
        more="{n} more paragraphs — {span}",
        corpus_kicker="The corpus · one chunk = one numbered paragraph",
        col_std="Standard", col_share="Share", total="Total indexed", hits="hits",
        weak_kicker="No reliable source",
        weak_gibberish="That does not look like a query.",
        weak_far="Nothing in the corpus comes close enough.",
        weak_body=("Retrieval **never abstains**: it always returns the nearest paragraphs, "
                   "whether or not anything is relevant. That is why the percentage **ranks**, it "
                   "does not grade — for a nonsense query all 1,700 paragraphs crowd into a narrow "
                   "band and the top one means nothing. The corpus is the ESRS Annex only: no "
                   "company data, no national law, no guidance."),
        weak_show="Show the nearest paragraphs anyway",
        empty_kicker="Empty · first run", empty_title="Nothing retrieved yet.",
        empty_body=("1,700 paragraphs of the ESRS Annex are indexed and waiting. "
                    "Ask a reporting question in either language."),
        r_step1="Question encoded · e5-base, 768d",
        r_step2="Scoring 1,700 vectors", r_step3="Assembling excerpts",
        r_meta="CPU only · no network call",
        ar="Application requirement", glossary="Glossary",
        disclaimer=("Work in progress — a portfolio project, not legal advice. "
                    "The authoritative text is the Official Journal."),
        tech="ⓘ  Technical details",
        tech_body=(
            "**Corpus** — ESRS Annex (12 standards), indexed as **1,700** numbered "
            "paragraphs, glossary entries and appendix rows. One chunk = one paragraph.\n\n"
            "**Retrieval** — `intfloat/multilingual-e5-base` (768 dims, 512-token window), "
            "running locally on CPU. Exact cosine similarity over pre-computed vectors.\n\n"
            "**Generation** — `llama3.2:3b` via **Ollama**, `temperature=0`, offline. "
            "The model is *not* hosted here: that would break the on-premise guarantee.\n\n"
            "⚠️ **This is the hosted demo, and it runs RETRIEVAL only.** The complete system "
            "— corpus, embeddings and LLM, all local, with no document ever leaving the "
            "machine — is the one you run on your own box, and it is the one every "
            "evaluation number comes from.\n\n"
            "Retrieval and generation are evaluated **separately**, against random and "
            "BM25 baselines, on 45 questions hand-verified against the source paragraphs.\n\n"
            "_Light / dark theme: ⋮ menu → Settings._"
        ),
    ),
}

# Document order, not size order (a reader looks for "E2", not "the 4th largest").
STANDARDS = [
    ("1",  "NEIS 1 Requisitos generales",           "ESRS 1 General requirements"),
    ("2",  "NEIS 2 Información general",            "ESRS 2 General disclosures"),
    ("E1", "E1 Cambio climático",                   "E1 Climate change"),
    ("E2", "E2 Contaminación",                      "E2 Pollution"),
    ("E3", "E3 Agua y recursos marinos",            "E3 Water and marine resources"),
    ("E4", "E4 Biodiversidad y ecosistemas",        "E4 Biodiversity and ecosystems"),
    ("E5", "E5 Uso de recursos y circularidad",     "E5 Resource use and circularity"),
    ("S1", "S1 Personal propio",                    "S1 Own workforce"),
    ("S2", "S2 Trabajadores de la cadena de valor", "S2 Workers in the value chain"),
    ("S3", "S3 Colectivos afectados",               "S3 Affected communities"),
    ("S4", "S4 Consumidores y usuarios finales",    "S4 Consumers and end-users"),
    ("G1", "G1 Conducta empresarial",               "G1 Business conduct"),
    ("GLOSARIO", "Glosario y apéndices",            "Glossary & appendices"),
]


# ─────────────────────────── loading (cached) ───────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL)


@st.cache_data(show_spinner=False)
def load_corpus():
    vectors = np.load(DATA_DIR / "corpus_vectores.npz")["E"]
    meta = json.loads((DATA_DIR / "corpus_textos.json").read_text(encoding="utf-8"))
    return vectors, meta


@st.cache_data(show_spinner=False)
def count_by_standard():
    _, meta = load_corpus()
    counts = {}
    for code in meta["estandares"]:
        counts[code] = counts.get(code, 0) + 1
    return counts


if "lang" not in st.session_state:
    st.session_state.lang = "es"
t = T[st.session_state.lang]


# ──────────────────── CSS: both design systems, one block ───────────────────
# Every token is light-dark(Broadsheet, Nocturne). SYSTEM typeface on purpose:
# the design calls for Inter from Google Fonts, which would be an external
# request on every visit — exactly what this app claims never to make.
st.markdown("""<style>
  :root {
    --e-bg:          light-dark(#f3f2f2, #161826);
    --e-surface:     light-dark(#eae9e9, #232532);
    --e-text:        light-dark(#201e1d, #e9e9ed);
    --e-accent:      light-dark(#1B7F5A, #7FC9A3);
    --e-accent-deep: light-dark(#115C41, #A8DCC2);
    --e-accent-ink:  light-dark(#0B3F2C, #EAF7F0);
    --e-tint:        light-dark(#E3F5EC, #2A4A3B);
    --e-spot-bg:     light-dark(#fff1f4, #2A4A3B);
    --e-spot-fg:     light-dark(#790e3d, #EAF7F0);
    --e-divider:     light-dark(rgba(32,30,29,.16), rgba(233,233,237,.16));
    /* The design's "muted" (#605d5d / #9397ab) only reaches ~5.8:1 and ~6.1:1,
       and these are 11px ALL-CAPS labels (kickers, metadata, ¶). Raised until
       both themes clear 10:1, including over the card surface (10.3:1 and
       8.9:1). Measured, not eyeballed. */
    --e-muted:       light-dark(#403d3d, #c3c6d4);
    --e-body:        light-dark(#2d2b2b, #cfd3e5);
    --e-track:       light-dark(#d7d3d3, #3f424d);
    --e-rule:        light-dark(#201e1d, rgba(233,233,237,.28));
    --e-rule-soft:   light-dark(rgba(32,30,29,.10), rgba(233,233,237,.10));
    --e-card-ring:   light-dark(transparent, #3f424d);
    --e-row:         light-dark(rgba(27,127,90,.09), rgba(127,201,163,.10));
    --e-glow:        light-dark(transparent, rgba(127,201,163,.13));
    --e-label-ls: .13em;
  }
  /* Streamlit's fixed header was covering the masthead: make it transparent
     and leave room at the top. */
  [data-testid="stHeader"] { background: transparent; }
  .block-container { max-width: 1440px; padding: 4.4rem 2.5rem 3rem; }
  /* ⚠️ Set the typeface ONLY on the container and let it inherit. A broad
     selector (e.g. [class*="st-"]) also reaches Streamlit's icons, which are
     LIGATURES of "Material Symbols Rounded": change their font and the
     ligature renders as its literal name ("expand_more") on top of the
     button's label. */
  .stApp, .stDialog, [data-testid="stMainMenuPopover"] {
      font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  }
  [data-testid="stIconMaterial"] { font-family: "Material Symbols Rounded" !important; }
  .e-kicker { font-size:11px; letter-spacing:var(--e-label-ls); text-transform:uppercase;
      color:var(--e-muted); }
  .e-rule { height:3px; background:var(--e-rule); }
  .e-hair { height:1px; background:var(--e-rule); }
  .e-soft { height:1px; background:var(--e-rule-soft); }
  .e-card { display:grid; grid-template-columns:36px 1fr; gap:20px; padding:20px;
      background:var(--e-surface); border-radius:2px; margin-bottom:10px;
      box-shadow:0 0 0 1px var(--e-card-ring); }
  .e-rank { font-weight:600; font-size:27px; line-height:1; color:var(--e-accent);
      font-variant-numeric:tabular-nums; }
  .e-src { font-size:11.5px; letter-spacing:.1em; text-transform:uppercase;
      color:var(--e-accent-deep); }
  .e-badge { font-size:11px; padding:2px 7px; background:var(--e-spot-bg);
      color:var(--e-spot-fg); border-radius:1px; }
  .e-badge-o { font-size:11px; padding:2px 7px; border:1px solid var(--e-divider);
      color:var(--e-muted); border-radius:1px; }
  .e-chip { font-size:11.5px; padding:3px 10px; background:var(--e-tint);
      color:var(--e-accent-ink); border-radius:1px; font-variant-numeric:tabular-nums; }
  .e-chip-o { font-size:11.5px; padding:3px 10px; border:1px solid var(--e-divider);
      color:var(--e-muted); border-radius:1px; }
  .e-body { font-size:16.5px; line-height:1.58; text-wrap:pretty; color:var(--e-body);
      margin:0; }
  .e-track { display:inline-block; width:64px; height:4px; background:var(--e-track);
      vertical-align:middle; }
  .e-fill { display:block; height:4px; background:var(--e-accent); }
  .e-meta { font-size:12.5px; color:var(--e-muted); font-variant-numeric:tabular-nums; }
  /* Corpus table: hand-built grid, as in the design (share bar + ¶ count). */
  .e-row { display:grid; grid-template-columns:1fr 76px 58px; align-items:center;
      gap:15px; padding:9px 0; font-size:14.5px; color:var(--e-text);
      background-image:linear-gradient(var(--e-rule-soft),var(--e-rule-soft));
      background-repeat:no-repeat; background-position:bottom; background-size:100% 1px; }
  .e-row-hit { background-color:var(--e-row); font-weight:600; }
  .e-row-head { padding:7px 0; font-size:10.5px; letter-spacing:var(--e-label-ls);
      text-transform:uppercase; color:var(--e-muted);
      background-image:linear-gradient(var(--e-rule),var(--e-rule)); }
  .e-bar { display:block; height:5px; background:var(--e-track); }
  .e-num { text-align:right; padding-right:10px; font-variant-numeric:tabular-nums;
      color:var(--e-muted); }
  /* The design's search box: surface background, accent edge. */
  [data-testid="stTextInput"] input { background:var(--e-surface) !important;
      border:1px solid var(--e-accent) !important; border-radius:2px !important;
      font-size:17px !important; padding:13px 14px !important; color:var(--e-text) !important; }
</style>""", unsafe_allow_html=True)


# ─────────────────────────── masthead ───────────────────────────
LEAF_SVG = ('<svg viewBox="0 0 256 256" width="22" height="22" fill="var(--e-accent)" '
            'style="display:block;flex:none"><path d="M228 40a12 12 0 0 0-12-12c-52 0-96 '
            '12-124 34C68 82 56 110 56 140c0 17 4 32 11 45l-24 24a12 12 0 0 0 17 17l24-24c13 '
            '7 28 11 45 11 30 0 58-12 78-36 22-28 34-72 34-124-.2-4-1-9-13-13Zm-40 122c-16 '
            '19-38 29-62 29-10 0-19-2-27-5l86-86a12 12 0 0 0-17-17l-86 86c-3-8-5-17-5-27 '
            '0-24 10-46 29-62 23-19 60-30 105-31-1 45-12 82-31 105Z"/></svg>')
LOCK_SVG = ('<svg viewBox="0 0 256 256" width="12" height="12" fill="currentColor" '
            'style="display:block;flex:none"><path d="M200 84h-36V56a36 36 0 0 0-72 0v28H56a20 '
            '20 0 0 0-20 20v96a20 20 0 0 0 20 20h144a20 20 0 0 0 20-20v-96a20 20 0 0 '
            '0-20-20ZM116 56a12 12 0 0 1 24 0v28h-24Zm80 140H60v-88h136Z"/></svg>')

left, right = st.columns([6, 1], vertical_alignment="center")
with left:
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:30px">'
        f'  <div style="display:flex;align-items:center;gap:10px">{LEAF_SVG}'
        f'    <span style="font-weight:600;font-size:18px;letter-spacing:-.01em;'
        f'color:var(--e-text)">ESRS<span style="color:var(--e-accent)">·</span>RAG</span></div>'
        f'  <nav style="display:flex;gap:30px;font-size:14.5px">'
        f'    <span style="color:var(--e-accent-deep);font-weight:600">{t["nav_search"]}</span>'
        f'    <a href="#corpus" style="text-decoration:none;color:var(--e-muted)">{t["nav_corpus"]}</a>'
        f'  </nav></div>', unsafe_allow_html=True)
with right:
    st.segmented_control("lang", ["es", "en"], key="lang", label_visibility="collapsed",
                         format_func=str.upper)

# Identity rail ("dateline"): ONLY three facts, as in the final design.
st.markdown(
    f'<div class="e-rule"></div>'
    f'<div style="display:flex;justify-content:center;gap:56px;padding:9px 0;font-size:11px;'
    f'letter-spacing:var(--e-label-ls);text-transform:uppercase;color:var(--e-muted)">'
    f'  <span>{t["dl_reg"]}</span>'
    f'  <span style="display:flex;align-items:center;gap:5px">{LOCK_SVG}{t["dl_onprem"]}</span>'
    f'  <span style="color:var(--e-accent-deep)">{t["dl_priv"]}</span></div>'
    f'<div class="e-hair"></div>', unsafe_allow_html=True)

# Hero
st.markdown(
    f'<section style="padding:40px 0 0;background:radial-gradient(120% 150% at 12% -30%,'
    f'var(--e-glow) 0%, transparent 60%)">'
    f'<h1 style="margin:0 0 15px;font-weight:600;font-size:46px;line-height:1.08;'
    f'letter-spacing:-.018em;color:var(--e-text)">{t["hero_a"]}<br>{t["hero_b"]}</h1>'
    f'<p style="margin:0 0 6px;font-size:17.5px;line-height:1.5;max-width:35em;'
    f'color:var(--e-body)">{t["hero_sub"]}</p></section>', unsafe_allow_html=True)


# ─────────────────────────── search ───────────────────────────
def use_example():
    """Clicking an example pill writes it into the search box."""
    if st.session_state.get("example"):
        st.session_state.query = st.session_state.example


box, button = st.columns([5, 1], vertical_alignment="bottom")
with box:
    st.text_input("q", key="query", placeholder=t["placeholder"],
                  label_visibility="collapsed")
with button:
    st.button(t["retrieve"], type="primary", width="stretch")

st.markdown(f'<span class="e-kicker">{t["or_try"]}</span>', unsafe_allow_html=True)
st.pills("ex", t["examples"], key="example", label_visibility="collapsed",
         on_change=use_example)

st.markdown(f'<span class="e-kicker">{t["depth"]}</span>', unsafe_allow_html=True)
k = st.segmented_control(
    "k", K_OPTIONS, default=DEFAULT_K, key="k", label_visibility="collapsed",
    help="k=8 is the deployed configuration, chosen by a sweep: below it the answer "
         "often falls outside the context; far above it the model drowns in noise."
) or DEFAULT_K

query = st.session_state.get("query", "").strip()


# ─────────────────────────── render helpers ───────────────────────────
def pieces_per_word(text, tokenizer):
    """Average subword pieces per word — the tokenizer's own gibberish detector."""
    words = [w for w in text.split() if any(c.isalpha() for c in w)]
    if not words:
        return float("inf")          # digits or symbols only
    return sum(len(tokenizer.tokenize(w)) for w in words) / len(words)


def standard_name(code):
    column = 1 if st.session_state.lang == "es" else 2
    for row in STANDARDS:
        if row[0] == code:
            return row[column]
    return code


def source_label(anchor):
    """'E2§AR 12' -> 'E2 Pollution · ¶ AR 12'"""
    code, _, rest = anchor.partition("§")
    if code == "GLOSARIO":
        return f'{t["glossary"]} · {rest}'
    return f'{standard_name(code)} · ¶ {rest}'


def card(rank, i, meta, similarity):
    req = meta["requisitos"][i]
    if req and req not in ("término", "acrónimo"):
        badge = f'<span class="e-badge">{html.escape(req)}</span>'
    elif "§AR" in meta["anclas"][i]:
        badge = f'<span class="e-badge-o">{t["ar"]}</span>'
    else:
        badge = ""
    pct = max(0, min(100, round(similarity * 100)))
    return (
        f'<article class="e-card"><div class="e-rank">{rank}</div><div>'
        f'<div style="display:flex;align-items:center;gap:15px;flex-wrap:wrap;margin-bottom:8px">'
        f'<span class="e-src">{html.escape(source_label(meta["anclas"][i]))}</span>{badge}'
        f'<span class="e-meta" style="margin-left:auto">{similarity*100:.1f}% '
        f'<span class="e-track"><span class="e-fill" style="width:{pct}%"></span></span>'
        f'</span></div>'
        f'<p class="e-body">{html.escape(meta["textos"][i])}</p></div></article>')


def corpus_column(codes, counts, hits, top_count):
    parts = [f'<div class="e-row e-row-head"><span>{t["col_std"]}</span>'
             f'<span>{t["col_share"]}</span><span style="text-align:right">¶</span></div>']
    for code in codes:
        n = counts.get(code, 0)
        n_hits = hits.get(code, 0)
        mark = (f'<span style="margin-left:10px;font-size:10.5px;font-weight:400;'
                f'letter-spacing:.08em;text-transform:uppercase;color:var(--e-accent-deep)">'
                f'{n_hits} {t["hits"]}</span>') if n_hits else ""
        parts.append(
            f'<div class="e-row{" e-row-hit" if n_hits else ""}">'
            f'<span style="padding-left:10px">{html.escape(standard_name(code))}{mark}</span>'
            f'<span class="e-bar"><span style="display:block;height:5px;'
            f'background:var(--e-accent);width:{round(100 * n / top_count)}%"></span></span>'
            f'<span class="e-num">{n}</span></div>')
    return "".join(parts)


def corpus_section(counts, hits):
    codes = [c for c, _, _ in STANDARDS]
    top_count = max(counts.values())
    total = f"{sum(counts.values()):,}".replace(
        ",", "." if st.session_state.lang == "es" else ",")
    st.markdown(f'<div id="corpus"></div><div class="e-soft" style="margin:34px 0 18px"></div>'
                f'<span class="e-kicker">{t["corpus_kicker"]}</span>', unsafe_allow_html=True)
    a, b = st.columns(2, gap="large")
    with a:
        st.markdown(corpus_column(codes[:7], counts, hits, top_count), unsafe_allow_html=True)
    with b:
        st.markdown(corpus_column(codes[7:], counts, hits, top_count) +
                    f'<div class="e-row" style="background-image:none">'
                    f'<span class="e-kicker" style="padding-left:10px">{t["total"]}</span>'
                    f'<span></span><span class="e-num" style="font-weight:600;'
                    f'color:var(--e-text)">{total}</span></div>', unsafe_allow_html=True)


# ─────────────────────────── results ───────────────────────────
counts = count_by_standard()

if not query:
    st.markdown(
        f'<div class="e-soft" style="margin:34px 0 18px"></div>'
        f'<div style="padding:20px;background:var(--e-surface);border-radius:2px;'
        f'box-shadow:0 0 0 1px var(--e-card-ring);max-width:44em">'
        f'<div class="e-kicker" style="color:var(--e-accent-deep)">{t["empty_kicker"]}</div>'
        f'<p style="margin:15px 0 10px;font-weight:600;font-size:19px;line-height:1.25;'
        f'color:var(--e-text)">{t["empty_title"]}</p>'
        f'<p style="margin:0;font-size:13.5px;line-height:1.5;color:var(--e-body)">'
        f'{t["empty_body"]}</p></div>', unsafe_allow_html=True)
    corpus_section(counts, {})
else:
    vectors, meta = load_corpus()
    with st.status(t["r_step2"], expanded=False) as status:
        model = load_model()             # outside the timer: it only loads once
        started = time.perf_counter()    # what is timed is the SEARCH, not the boot
        st.write(t["r_step1"])
        q = model.encode([f"query: {query}"], normalize_embeddings=True)[0]
        st.write(t["r_step2"])
        scores = vectors @ q.astype(np.float32)
        top = np.argsort(-scores, kind="stable")[:k]
        seconds = time.perf_counter() - started
        st.write(t["r_step3"])
        status.update(label=f'{t["r_step3"]} · {t["r_meta"]}', state="complete")

    # Two guards before showing anything as a result (see the constants above).
    not_language = pieces_per_word(query, model.tokenizer) > MAX_PIECES_PER_WORD
    nothing_close = float(scores[top[0]]) < MIN_TOP_SIMILARITY

    per_standard = {}
    for i in top:
        code = meta["estandares"][i]
        per_standard[code] = per_standard.get(code, 0) + 1
    touched = [c for c, _, _ in STANDARDS if c in per_standard]
    untouched = [c for c, _, _ in STANDARDS if c not in per_standard]
    chips = "".join(f'<span class="e-chip">{html.escape(standard_name(c))} · '
                    f'{per_standard[c]}</span>' for c in touched)
    if untouched:
        chips += (f'<span class="e-chip-o">{" · ".join(untouched)} — '
                  f'{t["coverage_none"]}</span>')

    if not_language or nothing_close:
        # The design's third state: "no reliable source". The paragraphs are NOT
        # deleted — they stay one click away, because hiding them would hide the
        # very behaviour this page exists to show.
        st.markdown(
            f'<div class="e-soft" style="margin:34px 0 18px"></div>'
            f'<div style="padding:20px;background:var(--e-surface);border-radius:2px;'
            f'box-shadow:0 0 0 1px var(--e-card-ring);max-width:44em">'
            f'<div class="e-kicker" style="color:var(--e-accent-deep)">{t["weak_kicker"]}</div>'
            f'<p style="margin:15px 0 10px;font-weight:600;font-size:19px;line-height:1.25;'
            f'color:var(--e-text)">'
            f'{t["weak_gibberish"] if not_language else t["weak_far"]}</p>'
            f'<p style="margin:0;font-size:13.5px;line-height:1.5;color:var(--e-body)">'
            f'{t["weak_body"]}</p></div>', unsafe_allow_html=True)
        with st.expander(f'{t["weak_show"]}  ·  {scores[top[0]]*100:.1f}% → '
                         f'{scores[top[-1]]*100:.1f}%'):
            for rank, i in enumerate(top, start=1):
                st.markdown(card(rank, i, meta, float(scores[i])), unsafe_allow_html=True)
        corpus_section(counts, {})
    else:

        st.markdown(
            f'<div class="e-soft" style="margin:34px 0 18px"></div>'
            f'<div style="display:flex;align-items:baseline;gap:20px;margin-bottom:10px">'
            f'<span class="e-kicker" style="color:var(--e-accent-deep)">{t["brief_kicker"]}</span>'
            f'<span class="e-meta">{k} {t["m_par"]} · {len(touched)} {t["m_src"]} · '
            f'{t["m_top"]} {scores[top[0]]*100:.1f}% · {seconds:.1f} s {t["m_cpu"]}</span></div>'
            f'<p style="margin:0 0 20px;font-weight:600;font-size:25px;line-height:1.25;'
            f'letter-spacing:-.018em;max-width:28em;color:var(--e-text)">'
            f'«{html.escape(query)}»</p>'
            f'<div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap">'
            f'<span class="e-kicker">{t["coverage"]}</span>{chips}</div>',
            unsafe_allow_html=True)

        st.markdown(f'<div style="margin:34px 0 12px"><span class="e-kicker">'
                    f'{t["para_kicker"]}</span></div>', unsafe_allow_html=True)
        for rank, i in enumerate(top[:VISIBLE], start=1):
            st.markdown(card(rank, i, meta, float(scores[i])), unsafe_allow_html=True)

        rest = list(top[VISIBLE:])
        if rest:
            span = f'{scores[rest[0]]*100:.1f}% → {scores[rest[-1]]*100:.1f}%'
            with st.expander(t["more"].format(n=len(rest), span=span)):
                for rank, i in enumerate(rest, start=VISIBLE + 1):
                    st.markdown(card(rank, i, meta, float(scores[i])), unsafe_allow_html=True)

        export_md = f"# {query}\n\n" + "\n\n".join(
            f'## {rank}. {source_label(meta["anclas"][i])} — {scores[i]*100:.1f}%\n\n'
            f'{meta["textos"][i]}' for rank, i in enumerate(top, start=1))
        st.download_button(t["export"], export_md, file_name="esrs-rag.md", mime="text/markdown")

        corpus_section(counts, per_standard)


# ─────────────────────────── footer ───────────────────────────
st.markdown('<div class="e-soft" style="margin:40px 0 15px"></div>', unsafe_allow_html=True)
footer, tech = st.columns([4, 1], vertical_alignment="center")
with footer:
    st.markdown(f'<span class="e-meta">⚠️ {t["disclaimer"]}</span>', unsafe_allow_html=True)
with tech:
    with st.popover(t["tech"], width="stretch"):
        st.markdown(t["tech_body"])
