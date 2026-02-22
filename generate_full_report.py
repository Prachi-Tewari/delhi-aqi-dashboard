"""
Comprehensive Technical Report Generator
==========================================
Generates a full PDF covering every module, feature, algorithm,
debug story, and implementation detail of the Delhi AQI Intelligence Platform.
"""

from fpdf import FPDF
import json, os, textwrap

PROJECT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT, "forecasting", "models")


class Report(FPDF):
    """Custom PDF with professional formatting."""

    BLUE = (30, 64, 175)
    DARK = (30, 41, 59)
    GRAY = (100, 116, 139)
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    LIGHT_BG = (248, 250, 252)
    ACCENT = (59, 130, 246)

    def __init__(self):
        super().__init__("P", "mm", "A4")
        self.set_auto_page_break(auto=True, margin=20)
        self.add_page()

    # ── Helpers ──────────────────────────────────────────────────────
    def _safe(self, text):
        """Replace Unicode chars that latin-1 can't handle."""
        return (str(text)
                .replace("\u2013", "-").replace("\u2014", "--")
                .replace("\u2018", "'").replace("\u2019", "'")
                .replace("\u201c", '"').replace("\u201d", '"')
                .replace("\u2026", "...").replace("\u2265", ">=")
                .replace("\u2264", "<=").replace("\u00b2", "2")
                .replace("\u00b3", "3").replace("\u2248", "~")
                .replace("\u00d7", "x").replace("\u2192", "->")
                .replace("\u03b1", "alpha").replace("\u03bb", "lambda")
                .replace("\u03c0", "pi").replace("\u03c3", "sigma")
                .replace("\u03bc", "u").replace("\u2080", "0")
                .replace("\u2081", "1").replace("\u2082", "2")
                .replace("\u2083", "3").replace("\u2084", "4")
                .replace("\u2085", "5").replace("\u2086", "6")
                .replace("\u00b5", "u")
                .encode("latin-1", "replace").decode("latin-1"))

    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(*self.GRAY)
            self.cell(0, 8, "Delhi AQI Intelligence Platform -- Technical Report", align="L")
            self.cell(0, 8, f"Page {self.page_no()}", align="R", new_x="LMARGIN", new_y="NEXT")
            self.set_draw_color(*self.ACCENT)
            self.line(10, 14, 200, 14)
            self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(*self.GRAY)
        self.cell(0, 10, "Confidential -- Prachi Tewari", align="C")

    def title_page(self):
        self.ln(50)
        self.set_font("Helvetica", "B", 28)
        self.set_text_color(*self.BLUE)
        self.cell(0, 14, "Delhi AQI Intelligence Platform", align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(4)
        self.set_font("Helvetica", "", 14)
        self.set_text_color(*self.DARK)
        self.cell(0, 10, "Comprehensive Technical Report", align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(8)
        self.set_draw_color(*self.ACCENT)
        self.line(60, self.get_y(), 150, self.get_y())
        self.ln(12)
        self.set_font("Helvetica", "", 11)
        self.set_text_color(*self.GRAY)
        lines = [
            "Real-Time Air Quality Monitoring, ML Forecasting,",
            "Dynamic Insights & RAG-Powered AI Advisory System",
            "",
            "Author: Prachi Tewari",
            "Repository: github.com/Prachi-Tewari/delhi-aqi-dashboard",
            "",
            "February 2026",
        ]
        for l in lines:
            self.cell(0, 7, l, align="C", new_x="LMARGIN", new_y="NEXT")

    def chapter(self, num, title):
        self.add_page()
        self.set_font("Helvetica", "B", 18)
        self.set_text_color(*self.BLUE)
        self.cell(0, 12, self._safe(f"Chapter {num}: {title}"),
                  new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*self.ACCENT)
        self.line(10, self.get_y(), 100, self.get_y())
        self.ln(6)

    def section(self, title):
        self.ln(4)
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(*self.DARK)
        self.cell(0, 9, self._safe(title), new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def subsection(self, title):
        self.ln(2)
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(71, 85, 105)
        self.cell(0, 7, self._safe(title), new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def body(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*self.BLACK)
        self.multi_cell(0, 5.5, self._safe(text))
        self.ln(1)

    def bullet(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*self.BLACK)
        x = self.get_x()
        self.cell(6, 5.5, "-")
        self.multi_cell(0, 5.5, self._safe(text))

    def code_block(self, text, width=190):
        self.set_fill_color(*self.LIGHT_BG)
        self.set_font("Courier", "", 8)
        self.set_text_color(30, 41, 59)
        for line in text.split("\n"):
            self.cell(width, 4.5, self._safe(line), fill=True,
                      new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def table(self, headers, rows, col_widths=None):
        if not col_widths:
            w = 190 // len(headers)
            col_widths = [w] * len(headers)
        # Header
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(*self.BLUE)
        self.set_text_color(*self.WHITE)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 7, self._safe(h), border=1, fill=True,
                      align="C")
        self.ln()
        # Rows
        self.set_font("Helvetica", "", 9)
        self.set_text_color(*self.BLACK)
        for ri, row in enumerate(rows):
            if ri % 2 == 0:
                self.set_fill_color(241, 245, 249)
            else:
                self.set_fill_color(*self.WHITE)
            for i, cell in enumerate(row):
                self.cell(col_widths[i], 6.5, self._safe(str(cell)),
                          border=1, fill=True, align="C")
            self.ln()
        self.ln(3)


def build_report():
    pdf = Report()
    pdf.title_page()

    # ══════════════════════════════════════════════════════════════════
    # TABLE OF CONTENTS
    # ══════════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(*Report.BLUE)
    pdf.cell(0, 12, "Table of Contents", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    toc = [
        ("1", "Project Overview & Architecture"),
        ("2", "Data Ingestion Layer (OpenAQ API Client)"),
        ("3", "AQI Computation Engine (India NAQI Standard)"),
        ("4", "ML Forecasting System"),
        ("5", "RAG-Powered AI Assistant"),
        ("6", "Dynamic Insights Engine"),
        ("7", "Visualization Module"),
        ("8", "Dashboard Application (app.py)"),
        ("9", "Debugging & Problem Resolution"),
        ("10", "Deployment & Configuration"),
    ]
    for num, title in toc:
        pdf.set_font("Helvetica", "", 11)
        pdf.set_text_color(*Report.DARK)
        pdf.cell(0, 7, f"  Chapter {num}:  {title}", new_x="LMARGIN", new_y="NEXT")

    # ══════════════════════════════════════════════════════════════════
    # CHAPTER 1: PROJECT OVERVIEW
    # ══════════════════════════════════════════════════════════════════
    pdf.chapter(1, "Project Overview & Architecture")

    pdf.section("1.1 Problem Statement")
    pdf.body(
        "Delhi, India consistently ranks among the world's most polluted cities. "
        "PM2.5 concentrations routinely exceed the WHO 24-hour guideline of 15 ug/m3 "
        "by 10-20x, contributing to an estimated 6-10 year reduction in life expectancy "
        "for Delhi residents. Existing air quality information systems suffer from: "
        "(i) delayed or infrequent updates, (ii) no short-term forecasting, "
        "(iii) raw numeric outputs without health context, and (iv) no conversational "
        "interface for public engagement."
    )

    pdf.section("1.2 Solution: Delhi AQI Intelligence Platform")
    pdf.body(
        "An end-to-end system with six integrated modules:\n\n"
        "1. Data Ingestion -- Live data from 15+ CPCB/DPCC stations via OpenAQ v3 API\n"
        "2. AQI Computation -- India NAQI-compliant index with per-pollutant sub-indices\n"
        "3. ML Forecasting -- Regime-aware LightGBM ensemble for 6-hour recursive AQI prediction\n"
        "4. Dynamic Insights -- Rule-based trend alerts, health advisories, anomaly detection\n"
        "5. RAG AI Assistant -- Conversational AI (Llama 3.3 70B) with hybrid retrieval and hallucination guards\n"
        "6. Streamlit Dashboard -- Professional UI with Plotly visualizations"
    )

    pdf.section("1.3 Technology Stack")
    pdf.table(
        ["Component", "Technology", "Version/Detail"],
        [
            ["Language", "Python", "3.14"],
            ["Web Framework", "Streamlit", "1.54.0"],
            ["ML: Gradient Boosting", "LightGBM", "4.6.0"],
            ["ML: Comparison Model", "XGBoost", "3.2.0"],
            ["Explainability", "SHAP", "0.50.0"],
            ["Vector Search", "FAISS", "CPU"],
            ["Embedding Model", "BAAI/bge-base-en-v1.5", "768-dim"],
            ["Reranker", "ms-marco-MiniLM-L-6-v2", "Cross-encoder"],
            ["LLM", "Llama 3.3 70B", "via Groq API"],
            ["Visualization", "Plotly", "Interactive charts"],
            ["Data Source", "OpenAQ v3 API", "REST, API key auth"],
            ["PDF Generation", "fpdf2", "Report module"],
        ],
        col_widths=[50, 70, 70],
    )

    pdf.section("1.4 System Architecture")
    pdf.body(
        "The system follows a pipeline architecture with six layers:\n\n"
        "OpenAQ v3 API (15+ Stations)\n"
        "    |\n"
        "    v\n"
        "[1. Data Ingestion] -> Station Discovery -> Sensor Deduplication -> Hourly Aggregation\n"
        "    |\n"
        "    v\n"
        "[2. AQI Engine] -> Sub-Index Calculation (6 pollutants) -> AQI = max(sub-indices) -> Category\n"
        "    |\n"
        "    +---> [3. ML Forecasting] -> 58 Features -> Regime Clustering (K=3) -> LightGBM -> 6h Forecast\n"
        "    +---> [4. Dynamic Insights] -> Trend Alerts, Health Advisories, Anomaly Detection\n"
        "    +---> [5. RAG Assistant] -> FAISS+BM25 -> Reranker -> Llama 3.3 70B -> Hallucination Guard\n"
        "    |\n"
        "    v\n"
        "[6. Streamlit Dashboard] -> Live AQI | Forecast | Stations | AI Chat | Visualizations\n"
        "    |\n"
        "    v\n"
        "End User"
    )

    pdf.section("1.5 Directory Structure")
    pdf.code_block(
        "aqi_rag_system/\n"
        "  app.py                    # Main Streamlit dashboard (1204 lines)\n"
        "  api/\n"
        "    openaq_client.py        # OpenAQ v3 API client (592 lines)\n"
        "  forecasting/\n"
        "    features.py             # Feature engineering (196 lines)\n"
        "    train.py                # Training pipeline (578 lines)\n"
        "    inference.py            # Live inference (446 lines)\n"
        "    models/                 # Saved model artifacts\n"
        "  rag/\n"
        "    retriever.py            # Hybrid retrieval (190 lines)\n"
        "    reranker.py             # Cross-encoder reranker (142 lines)\n"
        "    llm_pipeline.py         # Groq LLM interface (175 lines)\n"
        "    hallucination.py        # Hallucination detection (100 lines)\n"
        "    confidence.py           # Response confidence scoring (212 lines)\n"
        "    query_classifier.py     # Intent classification (169 lines)\n"
        "    prompt_template.py      # Prompt construction (84 lines)\n"
        "    memory.py               # Conversation memory (175 lines)\n"
        "    tool_calling.py         # Live data tool calls (201 lines)\n"
        "  insights/\n"
        "    engine.py               # Dynamic insights generator (432 lines)\n"
        "  visualization/\n"
        "    plots.py                # Plotly charts & AQI logic (1179 lines)\n"
        "  embeddings/\n"
        "    vector_store.py         # FAISS + BM25 hybrid store (226 lines)\n"
        "    cache.py                # Embedding cache\n"
        "  ingestion/\n"
        "    load_pdfs.py            # PDF loader\n"
        "    chunk_docs.py           # Text chunking\n"
        "    embed_docs.py           # Document embedding"
    )

    # ══════════════════════════════════════════════════════════════════
    # CHAPTER 2: DATA INGESTION
    # ══════════════════════════════════════════════════════════════════
    pdf.chapter(2, "Data Ingestion Layer (OpenAQ API Client)")

    pdf.section("2.1 OpenAQ v3 API Overview")
    pdf.body(
        "OpenAQ is an open-source platform aggregating government air quality data from 65+ countries. "
        "Our client (api/openaq_client.py, 592 lines) interfaces with the v3 REST API using an API key "
        "passed via the X-API-Key header. Base URL: https://api.openaq.org/v3"
    )

    pdf.section("2.2 Location Discovery")
    pdf.body(
        "The list_locations() function performs a geo-radius search centered on India Gate "
        "(28.6139N, 77.2090E) with a 25 km radius. It filters for stations active within "
        "the last 30 days by checking each location's datetimeLast field. Up to 100 locations "
        "are returned, each with a sensors list containing {id, parameter, units}."
    )
    pdf.code_block(
        "params = {\n"
        '    "coordinates": "28.6139,77.209",\n'
        '    "radius": 25000,\n'
        '    "limit": 100,\n'
        '    "iso": "IN"\n'
        "}"
    )

    pdf.section("2.3 Sensor Deduplication")
    pdf.body(
        "Critical discovery: Many Delhi stations have both legacy (defunct) and current sensors "
        "for the same pollutant parameter. Legacy sensors have lower IDs and return empty results, "
        "wasting API quota. Our deduplication strategy keeps only the sensor with the highest ID "
        "per parameter per location, ensuring we query the newest, active hardware."
    )
    pdf.code_block(
        "# Deduplication logic:\n"
        "best_sensor = {}\n"
        "for sensor in loc['sensors']:\n"
        "    param = sensor['parameter']\n"
        "    sid = sensor['id']\n"
        "    if param not in best_sensor or sid > best_sensor[param]['id']:\n"
        "        best_sensor[param] = sensor"
    )

    pdf.section("2.4 Measurements Endpoint (Critical Fix)")
    pdf.body(
        "We discovered that the /sensors/{id}/hours endpoint returns pre-computed hourly "
        "aggregates that are often stale (data from 2016!) and IGNORES date_from/date_to parameters. "
        "The fix was to switch to /sensors/{id}/measurements with datetime_from filtering, "
        "which returns raw measurements with correct date filtering. We then aggregate to hourly "
        "averages client-side.\n\n"
        "Endpoint: /sensors/{id}/measurements?datetime_from=<ISO>&limit=1000\n\n"
        "Client-side aggregation: group by floor(timestamp, 1h) and parameter, compute mean."
    )

    pdf.section("2.5 Fallback Synthesis")
    pdf.body(
        "When the API fails or returns empty data, _demo_hourly() generates synthetic data "
        "anchored to the last known live readings. It uses a diurnal pattern with morning and "
        "evening rush-hour peaks, Gaussian noise (~8% of base), and shifts the final value to "
        "match the actual live reading. This ensures the dashboard never shows a blank state "
        "and the forecast uses consistent values."
    )

    pdf.section("2.6 Rate Limiting & Safety")
    pdf.body(
        "- Maximum 50 API calls per refresh cycle\n"
        "- Process up to 10 locations (most relevant by proximity)\n"
        "- Skip parameters once we have 4x the needed hourly data\n"
        "- 25-second timeout per request\n"
        "- Safety timestamp filter: discard data older than (hours + 1) from now"
    )

    pdf.section("2.7 Other Endpoints")
    pdf.body(
        "get_latest_city_measurements() -- Most-recent value from each sensor. Uses /locations/{id}/latest "
        "which returns {sensorsId, value, datetime} but NO parameter name. We build a sensorId->param map "
        "from the location's sensors list.\n\n"
        "get_historical_city() -- Daily aggregated data via /sensors/{id}/days for trend analysis.\n\n"
        "get_station_latest() -- Per-station readings for the station detail view.\n\n"
        "list_city_stations() -- Simplified station list for UI selectors."
    )

    # ══════════════════════════════════════════════════════════════════
    # CHAPTER 3: AQI COMPUTATION
    # ══════════════════════════════════════════════════════════════════
    pdf.chapter(3, "AQI Computation Engine (India NAQI Standard)")

    pdf.section("3.1 India National Air Quality Index")
    pdf.body(
        "The AQI is defined by CPCB (Central Pollution Control Board) as the maximum sub-index "
        "across six criteria pollutants:\n\n"
        "AQI = max(I_PM2.5, I_PM10, I_NO2, I_SO2, I_CO, I_O3)\n\n"
        "Each sub-index I_p is computed via piecewise linear interpolation on CPCB breakpoint tables:\n\n"
        "I_p = I_lo + ((I_hi - I_lo) / (C_hi - C_lo)) * (C_p - C_lo)\n\n"
        "where C_p is the pollutant concentration and (C_lo, C_hi), (I_lo, I_hi) are the enclosing "
        "breakpoint pair."
    )

    pdf.section("3.2 NAQI Breakpoint Tables")
    pdf.table(
        ["AQI Range", "PM2.5", "PM10", "NO2", "SO2", "CO (mg/m3)", "O3"],
        [
            ["0-50",     "0-30",   "0-50",   "0-40",   "0-40",   "0-1.0",   "0-50"],
            ["51-100",   "31-60",  "51-100", "41-80",  "41-80",  "1.1-2.0", "51-100"],
            ["101-200",  "61-90",  "101-250","81-180", "81-380", "2.1-10",  "101-168"],
            ["201-300",  "91-120", "251-350","181-280","381-800","10.1-17",  "169-208"],
            ["301-400",  "121-250","351-430","281-400","801-1600","17.1-34", "209-748"],
            ["401-500",  "251-380","431-510","401-520","1601-2100","34.1-46","749-940"],
        ],
        col_widths=[25, 25, 28, 27, 30, 30, 25],
    )

    pdf.section("3.3 Unit Conversion Logic")
    pdf.body(
        "OpenAQ sensors report in various units (ug/m3, ppb, ppm, mg/m3). The _convert_to_ugm3() "
        "function handles all conversions:\n\n"
        "- CO is unique: NAQI breakpoints are in mg/m3 (not ug/m3)\n"
        "- ppb to ug/m3 conversion factors: NO2=1.88, SO2=2.62, CO=1.145e-3, O3=1.96\n"
        "- Special CO handling: OpenAQ often labels CO as 'ppb' when it's actually ppm. "
        "If CO 'ppb' value < 100, we treat it as ppm (real ppb would be 500-10000 for urban sites)\n"
        "- ppm to mg/m3: CO_PPM_TO_MGM3 = 1.145"
    )

    pdf.section("3.4 24-Hour Average Computation")
    pdf.body(
        "India NAQI uses 24-hour average concentrations. The _compute_24h_averages() function:\n"
        "1. Prefers hourly data (24h median across stations) when available\n"
        "2. Falls back to median of latest readings when hourly data is absent\n"
        "3. Median is used instead of mean for robustness against outlier stations"
    )

    pdf.section("3.5 AQI Categories & Health Advisories")
    pdf.table(
        ["AQI", "Category", "Health Guidance"],
        [
            ["0-50", "Good", "Safe for outdoor activities"],
            ["51-100", "Satisfactory", "Sensitive groups limit exertion"],
            ["101-200", "Moderate", "Reduce heavy outdoor exercise"],
            ["201-300", "Poor", "Everyone reduce outdoor activity"],
            ["301-400", "Very Poor", "Avoid outdoor, use air purifier"],
            ["401-500", "Severe", "Stay indoors, N95 mask, seek help"],
        ],
        col_widths=[20, 35, 135],
    )

    # ══════════════════════════════════════════════════════════════════
    # CHAPTER 4: ML FORECASTING
    # ══════════════════════════════════════════════════════════════════
    pdf.chapter(4, "ML Forecasting System")

    pdf.section("4.1 Dataset")
    pdf.body(
        "Delhi AQI Combined 2020-2024 dataset: 43,848 hourly observations across 42 columns "
        "from CPCB continuous ambient air quality monitoring stations.\n\n"
        "Variables:\n"
        "- 6 criteria pollutants: PM2.5, PM10, NO2, SO2, CO, O3\n"
        "- 3 auxiliary pollutants: NO, NOx, NH3\n"
        "- 7 meteorological parameters: temperature, humidity, wind speed, wind direction, "
        "rainfall, solar radiation, barometric pressure\n"
        "- AQI and category labels\n\n"
        "Preprocessing:\n"
        "- Column standardization via RAW_COL_MAP dictionary\n"
        "- Hourly resampling using median aggregation\n"
        "- Linear time-interpolation for gaps <= 6 hours\n"
        "- Winsorization at 0.1% and 99.9% quantiles\n"
        "- Duplicate timestamp removal (keep last)"
    )

    pdf.section("4.2 Feature Engineering (58 Features)")
    pdf.body(
        "All features are strictly causal -- no future information leakage. "
        "The build_features() function applies transformations in a fixed order."
    )

    pdf.subsection("4.2.1 AQI Lag Features (7 features)")
    pdf.body(
        "Lag horizons: h in {1, 2, 3, 6, 12, 24, 168} hours.\n"
        "aqi_lag1 captures immediate autoregressive dynamics.\n"
        "aqi_lag24 captures daily periodicity.\n"
        "aqi_lag168 captures weekly patterns."
    )

    pdf.subsection("4.2.2 Pollutant Lag Features (14 features)")
    pdf.body(
        "Lag-1 and lag-3 for each of 7 key pollutants (pm25, pm10, no2, so2, nh3, co, o3). "
        "These capture multi-pollutant temporal dynamics beyond the aggregate AQI."
    )

    pdf.subsection("4.2.3 Rolling Statistics (6 features)")
    pdf.body(
        "For windows w in {3, 6, 24} hours, compute rolling mean and standard deviation "
        "of AQI. Windows are shifted by 1 step (df[target].shift(1).rolling(...)) to prevent "
        "target leakage. The 24h rolling std captures volatility."
    )

    pdf.subsection("4.2.4 Rate-of-Change Features (4 features)")
    pdf.body(
        "First-order differences: delta1 = AQI(t) - AQI(t-1), delta3, delta6.\n"
        "Second-order acceleration: accel = delta1(t) - delta1(t-1).\n"
        "These capture momentum and inflection points."
    )

    pdf.subsection("4.2.5 Temporal Features (10 features)")
    pdf.body(
        "Calendar: hour, day-of-week, month, is_weekend.\n"
        "Cyclical encoding using sin/cos to preserve periodicity:\n"
        "hour_sin = sin(2*pi*hour/24), hour_cos = cos(2*pi*hour/24)\n"
        "dow_sin/cos (period=7), month_sin/cos (period=12)."
    )

    pdf.subsection("4.2.6 Meteorological Features (17 features)")
    pdf.body(
        "Raw: temperature, humidity, wind_speed, wind_dir, rainfall, solar_rad, pressure.\n"
        "Raw pollutants: pm25, pm10, no2, so2, nh3, co, o3, no, nox.\n"
        "Derived:\n"
        "- Wind vector decomposition: u = speed*sin(dir), v = speed*cos(dir)\n"
        "- Temperature-humidity interaction: T*RH/100 (atmospheric stability proxy)\n"
        "- Solar-temperature product: SR*T/1000 (photochemical activity for O3)"
    )

    pdf.section("4.3 Pollution Regime Clustering")
    pdf.body(
        "A key innovation: We identify distinct pollution regimes using unsupervised learning.\n\n"
        "Process:\n"
        "1. Aggregate hourly data to daily features: AQI mean/std/max/min/range, PM2.5/PM10 means, "
        "temperature/humidity/wind_speed means\n"
        "2. StandardScaler normalization\n"
        "3. KMeans clustering (k=3, n_init=10, random_state=42)\n"
        "4. Map daily regime labels back to hourly via forward-fill\n\n"
        "Result: Three regimes emerge naturally:"
    )

    pdf.table(
        ["Regime", "Hourly Observations", "Mean AQI", "Characterization"],
        [
            ["0 (Clean)", "12,984", "~106", "Monsoon, post-rain, good dispersion"],
            ["1 (Moderate)", "7,808", "~231", "Transition, dust storms, post-Diwali"],
            ["2 (Severe)", "5,280", "~349", "Winter inversions, stubble burning"],
        ],
        col_widths=[30, 40, 25, 95],
    )

    pdf.section("4.4 Model Architecture")
    pdf.subsection("4.4.1 LightGBM Configuration")
    pdf.body(
        "LightGBM is a gradient boosted decision tree (GBDT) framework. Unlike neural networks, "
        "GBDTs do not use epochs -- they build trees sequentially. Each tree corrects the residual "
        "errors of all previous trees."
    )
    pdf.table(
        ["Parameter", "Value", "Purpose"],
        [
            ["boosting_type", "GBDT", "Standard gradient boosting"],
            ["n_estimators", "2000", "Max trees (early stopped at ~50 patience)"],
            ["num_leaves", "127", "Complexity per tree (2^7 - 1)"],
            ["learning_rate", "0.05", "Step size per tree"],
            ["feature_fraction", "0.8", "Random 80% features per tree (regularization)"],
            ["bagging_fraction", "0.8", "Random 80% samples per tree"],
            ["bagging_freq", "5", "Subsample every 5 iterations"],
            ["min_child_samples", "20", "Min samples per leaf (prevents overfitting)"],
            ["reg_alpha", "0.1", "L1 regularization"],
            ["reg_lambda", "0.1", "L2 regularization"],
        ],
        col_widths=[40, 30, 120],
    )

    pdf.subsection("4.4.2 Regime-Aware Ensemble")
    pdf.body(
        "The final model is an ensemble of per-regime LightGBM models:\n\n"
        "1. Training: Separate LightGBM is trained on data from each regime\n"
        "2. Inference: Input data is classified into a regime via the KMeans model, "
        "then the corresponding specialized model generates the prediction\n"
        "3. Fallback: If a regime has <100 training or <20 validation samples, "
        "the global model is used instead\n\n"
        "This architecture lets each model learn distinct feature-target relationships "
        "for each atmospheric condition -- the relationship between wind speed and AQI "
        "is fundamentally different during winter inversions vs. monsoon."
    )

    pdf.subsection("4.4.3 Training Protocol")
    pdf.body(
        "Strict time-based split (no random shuffling, prevents temporal leakage):\n"
        "- Train: 2020-2022 (26,072 hours)\n"
        "- Validation: 2023 (8,605 hours) -- used for early stopping only\n"
        "- Test: 2024 (8,784 hours) -- completely held out, never seen during training\n\n"
        "The validation set determines when to stop adding trees (early stopping patience=50). "
        "If validation error hasn't improved for 50 consecutive trees, training halts. "
        "This prevents overfitting while keeping model complexity optimal."
    )

    pdf.section("4.5 Results")
    pdf.subsection("4.5.1 One-Step Prediction Performance (Test Set, 2024)")
    # Load actual results
    try:
        with open(os.path.join(MODELS_DIR, "training_results.json")) as f:
            results = json.load(f)
    except:
        results = {}

    mc = results.get("model_comparison", {})
    rows = []
    for model, metrics in mc.items():
        mae = metrics.get("MAE", metrics.get("Persist_MAE", "N/A"))
        rmse = metrics.get("RMSE", metrics.get("Persist_RMSE", "N/A"))
        r2 = metrics.get("R2", metrics.get("Persist_R2", "N/A"))
        spike = metrics.get("Spike_MAE", metrics.get("Persist_Spike_MAE", "N/A"))
        rows.append([model, str(mae), str(rmse), str(r2), str(spike)])

    if rows:
        pdf.table(
            ["Model", "MAE", "RMSE", "R2", "Spike MAE"],
            rows,
            col_widths=[50, 30, 30, 30, 50],
        )
    pdf.body(
        "The regime-aware LightGBM achieves the best overall MAE (0.49) and dramatically "
        "outperforms all baselines on spike events (Spike MAE = 0.66 vs 1.22 for XGBoost "
        "and 2.09 for persistence). This is the primary motivation for regime clustering: "
        "high-AQI events follow fundamentally different dynamics."
    )

    pdf.subsection("4.5.2 Six-Hour Recursive Forecast Performance")
    rec = results.get("recursive_metrics", {})
    rec_rows = []
    for h, m in rec.items():
        rec_rows.append([h, str(m.get("MAE")), str(m.get("RMSE")),
                         str(m.get("R2")), str(m.get("Spike_MAE"))])
    if rec_rows:
        pdf.table(
            ["Horizon", "MAE", "RMSE", "R2", "Spike MAE"],
            rec_rows,
            col_widths=[30, 35, 35, 35, 55],
        )
    pdf.body(
        "Error growth is approximately linear (~1.3 MAE per hour). Even at the 6-hour horizon, "
        "R2 > 0.986. For a city where AQI ranges 50-500, a 6-hour MAE of 8.57 represents "
        "a relative error of <3% -- sufficient for actionable public health decisions."
    )

    pdf.subsection("4.5.3 SHAP Feature Importance")
    try:
        with open(os.path.join(MODELS_DIR, "shap_importance.json")) as f:
            shap_data = json.load(f)
        top10 = list(shap_data.items())[:10]
        shap_rows = [[str(i+1), name, f"{val:.4f}"]
                     for i, (name, val) in enumerate(top10)]
        pdf.table(
            ["Rank", "Feature", "Mean |SHAP|"],
            shap_rows,
            col_widths=[20, 90, 80],
        )
    except:
        pass
    pdf.body(
        "aqi_lag1 dominates with a mean |SHAP| of 78.14, confirming strong autoregressive behavior. "
        "The 3-hour rolling mean (11.16) and aqi_lag2 (8.49) provide significant complementary signal. "
        "Rate-of-change features (delta1=1.70) capture momentum. Raw PM2.5 at rank 9 (0.11) shows "
        "the model also leverages current pollutant composition beyond AQI."
    )

    pdf.section("4.6 Recursive Multi-Step Forecasting Logic")
    pdf.body(
        "The recursive_forecast() function predicts AQI for +1h to +6h:\n\n"
        "For each step h = 1, 2, ..., 6:\n"
        "  1. Extract feature vector from the last row of the working DataFrame\n"
        "  2. Predict: y_hat = model.predict(X_input)\n"
        "  3. Create a new row for timestamp t+1:\n"
        "     - Set AQI(t+1) = y_hat\n"
        "     - Update all lag features: lag1 = AQI(t), lag2 = AQI(t-1), ...\n"
        "     - Recompute rolling mean/std over the extended window\n"
        "     - Update delta1 = y_hat - AQI(t)\n"
        "     - Advance temporal features (hour, day, cyclical encodings)\n"
        "  4. Append the new row to the working DataFrame\n"
        "  5. Repeat from step 1 with updated data\n\n"
        "Each step feeds predictions from prior steps, so errors compound -- "
        "but our linear error growth profile (+1.3 MAE/hour) shows this is well-controlled."
    )

    pdf.section("4.7 Live Inference Pipeline")
    pdf.body(
        "The inference module (forecasting/inference.py) bridges offline-trained models "
        "with real-time API data:\n\n"
        "1. Format detection: Auto-detect long (OpenAQ) vs wide format\n"
        "2. Parameter normalization: Map 'pm2.5'->'pm25', 'nitrogen dioxide'->'no2', etc.\n"
        "3. Pivot: Long to wide format with hourly averaging across stations\n"
        "4. AQI computation: Per-hour AQI from NAQI breakpoints. Current AQI overrides last hour.\n"
        "5. Feature construction: Full 58-feature pipeline. Missing features zero-filled.\n"
        "6. Recursive prediction: 6-hour forecast with feature updates at each step\n"
        "7. Post-processing: Trend classification (worsening/stable/improving based on 10% threshold), "
        "confidence (high/medium/low based on data hours), uncertainty bands (15% at +1h, 40% at +6h), "
        "natural language summary"
    )

    pdf.section("4.8 Uncertainty Quantification")
    pdf.body(
        "Prediction intervals: sigma_h = (0.10 + 0.05*h) * y_hat\n"
        "This produces 15% bands at +1h growing to 40% at +6h. The heuristic was calibrated "
        "against recursive evaluation residual distributions. Future work: conformal prediction "
        "or quantile regression for rigorous statistical coverage guarantees."
    )

    # ══════════════════════════════════════════════════════════════════
    # CHAPTER 5: RAG AI ASSISTANT
    # ══════════════════════════════════════════════════════════════════
    pdf.chapter(5, "RAG-Powered AI Assistant")

    pdf.section("5.1 Architecture Overview")
    pdf.body(
        "The AI assistant combines Retrieval-Augmented Generation with live data injection:\n\n"
        "User Query\n"
        "  -> Query Classifier (intent detection)\n"
        "  -> Hybrid Retrieval (FAISS dense + BM25 sparse)\n"
        "  -> Cross-Encoder Reranking (ms-marco-MiniLM-L-6-v2)\n"
        "  -> Prompt Construction (question + RAG context + live data + forecast)\n"
        "  -> Llama 3.3 70B (via Groq API)\n"
        "  -> Hallucination Detection\n"
        "  -> Confidence Scoring\n"
        "  -> Response with badges and citations"
    )

    pdf.section("5.2 Query Classification (query_classifier.py)")
    pdf.body(
        "Keyword-based intent classifier with 7 categories:\n\n"
        "- live_data: 'current', 'right now', 'today' -> needs API, skip RAG\n"
        "- health_advice: 'safe', 'mask', 'exercise' -> needs both API + RAG\n"
        "- historical: 'trend', 'past year' -> needs RAG, skip API\n"
        "- comparison: 'compare', 'vs', 'WHO' -> needs both\n"
        "- factual: 'what is', 'explain', 'how does' -> needs RAG\n"
        "- policy: 'government', 'GRAP', 'ban' -> needs RAG\n"
        "- general: catch-all\n\n"
        "Each intent maps to a config dict specifying: needs_api, needs_rag, top_k, "
        "preferred_topics (for retrieval boosting)."
    )

    pdf.section("5.3 Embedding & Vector Store")
    pdf.body(
        "Embedding Model: BAAI/bge-base-en-v1.5 (768 dimensions)\n"
        "Query Prefix: 'Represent this sentence for searching relevant passages: '\n"
        "Embedding Cache: In-memory LRU (4096 entries) with NumPy NPZ disk persistence.\n\n"
        "Vector Store (vector_store.py):\n"
        "- FAISS IndexFlatL2 for dense retrieval (GPU-free, exact search)\n"
        "- Custom BM25 implementation (Okapi BM25, k1=1.5, b=0.75) for sparse retrieval\n"
        "- No external BM25 dependency -- 60-line pure Python implementation\n"
        "- Metadata stored as JSON (text, topic, year, credibility_score per chunk)"
    )

    pdf.section("5.4 Hybrid Search")
    pdf.body(
        "The retrieve() method combines dense and sparse retrieval:\n\n"
        "1. Embed query using BGE model\n"
        "2. FAISS search: top-20 by L2 distance (dense_weight=0.6)\n"
        "3. BM25 search: top-20 by keyword relevance (sparse_weight=0.4)\n"
        "4. Merge and deduplicate candidates\n"
        "5. Optional topic boost: if intent prefers certain topics (e.g., 'health'), "
        "boost matching candidates by 1.3x"
    )

    pdf.section("5.5 Cross-Encoder Reranking (reranker.py)")
    pdf.body(
        "Cross-encoder model: cross-encoder/ms-marco-MiniLM-L-6-v2\n\n"
        "Process:\n"
        "1. Form (query, passage) pairs for all candidates\n"
        "2. Score each pair with the cross-encoder (produces a logit)\n"
        "3. Normalize scores to [0,1] using sigmoid\n"
        "4. Combine with source reliability: final = 0.85*ce_score + 0.15*credibility\n"
        "5. Adaptive top-k: return results above min_score=0.15, up to top_k\n\n"
        "This two-stage retrieve-then-rerank architecture is computationally efficient -- "
        "the expensive cross-encoder only scores 20 candidates instead of the entire corpus."
    )

    pdf.section("5.6 LLM Generation (llm_pipeline.py)")
    pdf.body(
        "Model: Llama 3.3 70B Versatile (via Groq Cloud API)\n"
        "Temperature: 0.4 (balanced creativity/accuracy)\n"
        "top_p: 0.9\n"
        "Max tokens: 1500\n\n"
        "System prompt (SYSTEM_MESSAGE) establishes the AI as a world-class air quality "
        "intelligence analyst with expertise in: India NAQI standard, Delhi pollution sources "
        "(vehicular 40-50%, stubble burning, industry), seasonal patterns, health science "
        "(PM2.5 alveolar penetration, COPD, IHD), policy context (GRAP stages, NCAP targets), "
        "WHO guidelines, and measurement science.\n\n"
        "10 response rules enforce: answer only what's asked, use markdown, include hyperlinks "
        "to authoritative sources (WHO, CPCB, IQAir, PubMed), cite [Ref N] for knowledge base, "
        "match tone to question complexity."
    )

    pdf.section("5.7 Live Data Context Injection")
    pdf.body(
        "The system prompt is augmented at runtime with:\n\n"
        "1. Current AQI data: All pollutant values, sub-indices, category, dominant pollutant, "
        "cigarette equivalent\n"
        "2. Forecast data: 6-hour predictions with trend, confidence, per-hour AQI and category\n"
        "3. Dynamic insights: Top 4 current insights (trend alerts, health advisories)\n\n"
        "This ensures the LLM always has current factual context, enabling accurate answers "
        "to questions like 'Is it safe to jog right now?' without hallucinating numbers."
    )

    pdf.section("5.8 Hallucination Detection (hallucination.py)")
    pdf.body(
        "Post-generation analysis checks:\n"
        "1. Citation check: If RAG context was provided, does the response contain [Ref N] citations?\n"
        "2. Numeric claims: More than 3 specific numbers without source attribution -> warning\n"
        "3. Hedging language: 'I think', 'probably', 'maybe' -> uncertainty signal\n"
        "4. Combined risk: low/medium/high\n\n"
        "If risk is HIGH and RAG context was available, the system automatically regenerates "
        "the response with a stricter prompt that forces source citation. This prevents the LLM "
        "from making up statistics about Delhi pollution."
    )

    pdf.section("5.9 Confidence Scoring (confidence.py)")
    pdf.body(
        "Each response gets a 0-1 confidence score from four dimensions:\n\n"
        "1. Source quality: Average reranker scores + credibility of retrieved sources\n"
        "2. Citation coverage: How well the response cites available sources\n"
        "3. Relevance: Does the response address the query? (keyword overlap heuristic)\n"
        "4. Response quality: Length, structure, specificity\n\n"
        "Weights vary by intent (e.g., factual queries weight source quality higher). "
        "Grade: high (>=0.7), medium (>=0.4), low (<0.4). Displayed as a badge under each response."
    )

    pdf.section("5.10 Conversation Memory (memory.py)")
    pdf.body(
        "The ConversationMemory class manages multi-turn context:\n\n"
        "- Max token budget: 6,000 tokens\n"
        "- Keep recent: 6 messages verbatim\n"
        "- Older messages: Automatically summarized using the LLM itself\n"
        "- Summarization prompt asks for key facts, AQI values, and user preferences\n\n"
        "This prevents the conversation from exceeding the LLM context window while "
        "preserving important information from earlier turns."
    )

    pdf.section("5.11 Prompt Template (prompt_template.py)")
    pdf.body(
        "The build_prompt() function assembles the user message:\n\n"
        "1. User's question (front and centre)\n"
        "2. Intent hint for calibrating response style\n"
        "3. Tool-call results (if live API data was fetched)\n"
        "4. Live data snapshot (current AQI readings)\n"
        "5. RAG knowledge-base excerpts with [Ref N] labels, sources, and relevance scores\n"
        "6. Lightweight instructions: answer what's asked, use citations, include links"
    )

    pdf.section("5.12 Tool Calling (tool_calling.py)")
    pdf.body(
        "When the query classifier detects 'live_data' intent, the auto_tool_call() function:\n\n"
        "1. Checks if a cached snapshot exists (from app.py's current computation)\n"
        "2. If yes, packages it as a tool result (avoids redundant API calls)\n"
        "3. If no, calls get_current_aqi() to fetch fresh data from OpenAQ\n\n"
        "The tool result is formatted as structured context for the LLM prompt."
    )

    # ══════════════════════════════════════════════════════════════════
    # CHAPTER 6: INSIGHTS ENGINE
    # ══════════════════════════════════════════════════════════════════
    pdf.chapter(6, "Dynamic Insights Engine")

    pdf.section("6.1 Overview")
    pdf.body(
        "The insights engine (insights/engine.py, 432 lines) generates real-time, "
        "context-aware analyses WITHOUT LLM calls -- ensuring low latency and deterministic outputs. "
        "Each insight is a dataclass with: category, severity (info/warning/critical), title, body, "
        "and priority score for display ordering."
    )

    pdf.section("6.2 Insight Categories")
    pdf.body(
        "1. Trend Alerts: Detects >25% pollutant concentration changes in 3-hour windows. "
        "Compares recent 3h mean vs earlier 3h mean for PM2.5, PM10, NO2, O3.\n\n"
        "2. Diurnal Patterns: Time-of-day contextualization based on IST hour:\n"
        "   - 7-10 AM: Morning rush hour warning\n"
        "   - 17-21: Evening pollution peak (boundary layer collapse)\n"
        "   - 23-04: Nighttime inversion alert\n"
        "   - 11-15: Midday mixing (best outdoor window)\n\n"
        "3. WHO Comparisons: Current readings vs WHO 2021 guidelines. Highlights 5x/10x exceedances. "
        "Includes cigarette equivalent (PM2.5 / 22 = daily cigarettes).\n\n"
        "4. Forecast Notes: Extracts highlights from ML forecast -- worsening/improving predictions, "
        "category transitions, pollutant-specific rising trends.\n\n"
        "5. Station Insights: Identifies spatial variation -- when worst station reads 2x+ the best "
        "station for PM2.5 or PM10, flags hyperlocal sources.\n\n"
        "6. Health Context: Population-specific advisories for respiratory patients, cardiac risk, "
        "outdoor exercise, sensitive groups. Severity escalates with AQI.\n\n"
        "7. Anomaly Detection: (via station comparisons) Stations deviating significantly from city mean."
    )

    pdf.section("6.3 Priority System")
    pdf.body(
        "Each insight gets a priority score (0-100+):\n"
        "- AQI emergency (>300): priority 100\n"
        "- WHO 10x exceedance: priority 95\n"
        "- AQI worsening forecast: priority 85\n"
        "- Trend spike (>25%): priority 80 + pct_change\n"
        "- WHO 5x exceedance: priority 75\n"
        "- Respiratory risk: priority 70-90\n"
        "- Evening peak: priority 60\n\n"
        "Insights are sorted by priority descending and capped at max_insights (default 6)."
    )

    # ══════════════════════════════════════════════════════════════════
    # CHAPTER 7: VISUALIZATION
    # ══════════════════════════════════════════════════════════════════
    pdf.chapter(7, "Visualization Module")

    pdf.section("7.1 Chart Functions (plots.py, 1179 lines)")
    pdf.body(
        "All charts are interactive Plotly figures with a consistent professional theme:\n\n"
        "1. aqi_gauge() -- Semicircular gauge with AQI value, colored by category\n"
        "2. sub_index_chart() -- Horizontal bar chart of all 6 sub-indices\n"
        "3. pollutant_vs_who() -- Grouped bar: current value vs WHO guideline (red highlight for exceedance)\n"
        "4. timeseries_plot() -- Time series for any pollutant with date range\n"
        "5. pollutant_radar() -- 6-axis radar chart of normalized sub-indices\n"
        "6. aqi_scale_bar() -- Reference color bar with category labels\n"
        "7. station_comparison_chart() -- Bar chart comparing stations for a selected pollutant\n"
        "8. station_aqi_heatmap() -- Heatmap: stations x pollutants, colored by sub-index severity\n"
        "9. station_detail_chart() -- Per-station Plotly figure with all parameters\n"
        "10. forecast_chart() -- 6-hour forecast line with AQI category background bands and confidence intervals"
    )

    pdf.section("7.2 Forecast Chart Design")
    pdf.body(
        "The forecast_chart() function creates a Plotly figure with:\n"
        "- AQI category background bands (Good=green through Severe=red) as horizontal shapes\n"
        "- Confidence interval as a filled area (upper/lower bounds)\n"
        "- Forecast line: dashed blue with circle markers\n"
        "- Current AQI: diamond marker at hour 0\n"
        "- Hover data showing AQI value, category, and confidence range"
    )

    # ══════════════════════════════════════════════════════════════════
    # CHAPTER 8: DASHBOARD APPLICATION
    # ══════════════════════════════════════════════════════════════════
    pdf.chapter(8, "Dashboard Application (app.py)")

    pdf.section("8.1 Overview")
    pdf.body(
        "The main application (app.py, 1204 lines) is a Streamlit wide-layout dashboard that "
        "orchestrates all modules. Key design principles:\n"
        "- Professional analytics aesthetic (no emojis)\n"
        "- Inter + JetBrains Mono fonts\n"
        "- CSS animations: fadeInUp, pulse, shimmer, barGrow\n"
        "- Responsive design (mobile breakpoints at 768px)"
    )

    pdf.section("8.2 Data Flow")
    pdf.body(
        "1. Sidebar: API key inputs, city/country selection, history days slider\n"
        "2. Data fetch: Three cached (@st.cache_data, TTL=600s) functions fetch latest, hourly, and station data\n"
        "3. AQI computation: _compute_24h_averages() -> compute_aqi() -> classify\n"
        "4. Snapshot: Build key-value dict of all readings for chat context\n"
        "5. Insights: generate_insights() with current AQI, pollutants, hourly data\n"
        "6. Forecast: forecast_next_6_hours() with hourly data, current AQI, pollutant values"
    )

    pdf.section("8.3 UI Sections (Top to Bottom)")
    pdf.body(
        "1. Navigation Bar: Title, live dot, station count, timestamp\n"
        "2. Hero Banner: Large AQI number, gradient background colored by category, "
        "LIVE badge, PM2.5 value, dominant pollutant\n"
        "3. AQI Scale Strip: 6-segment color bar with category labels\n"
        "4. Pollutant Cards: 6 cards showing value, sub-index, animated progress bar\n"
        "5. Key Metrics: Cigarettes/day equivalent, AQI with category, WHO exceedance multiplier\n"
        "6. Health Advisory: Category-specific guidance + condition-specific cards "
        "(asthma, cardiac, allergies, COPD) + recommended actions pills\n"
        "7. Dynamic Insights: Grid of insight cards sorted by priority\n"
        "8. 6-Hour AQI Forecast: 3 KPI cards (trend, next-hour, confidence) + Plotly forecast chart + details table\n"
        "9. Station-Wise Monitoring: 3 tabs (comparison bar chart, heatmap, station detail drill-down)\n"
        "10. Detailed Analysis: 4 tabs (gauge+radar overview, WHO comparison, trend charts, raw data table)\n"
        "11. AI Chat: Full conversational interface with RAG, citations, confidence badges, response details\n"
        "12. Footer: Data source credits and technology attributions"
    )

    pdf.section("8.4 Chat Integration")
    pdf.body(
        "The chat system integrates all components:\n"
        "1. User types a question\n"
        "2. _retrieve_rag(): Classify intent -> retrieve from vector store (if needs_rag)\n"
        "3. _execute_tool_call(): Check if live data fetch needed (if needs_api)\n"
        "4. Build context: System prompt + live snapshot + forecast + insights\n"
        "5. Build user message: Question + RAG refs + tool results\n"
        "6. Conversation memory: Trim/summarize history to fit token budget\n"
        "7. LLM call: chat_with_guard() -> generate + hallucination check + confidence score\n"
        "8. Display: Markdown answer + confidence badge + expandable details (intent, chunks, hallucination risk)\n"
        "9. Source citations: If RAG was used, show expandable list of referenced documents\n"
        "10. Fallback: If LLM fails, _rule_answer() generates a template-based response from live data"
    )

    # ══════════════════════════════════════════════════════════════════
    # CHAPTER 9: DEBUGGING & PROBLEM RESOLUTION
    # ══════════════════════════════════════════════════════════════════
    pdf.chapter(9, "Debugging & Problem Resolution")

    pdf.section("9.1 API Date Filtering Bug (Critical)")
    pdf.body(
        "Problem: Predictions were 'a bit off' -- forecast showed AQI ~300 when live reading was ~105.\n\n"
        "Root Cause Investigation:\n"
        "1. Wrote _test_api.py to examine raw API responses\n"
        "2. Discovered get_hourly_data() used /sensors/{id}/hours endpoint\n"
        "3. This endpoint returns pre-computed aggregates from 2016-2025(!)\n"
        "4. It completely IGNORES date_from and date_to parameters\n"
        "5. The forecast model was trained on recent data but receiving 8-year-old stale data\n\n"
        "Fix:\n"
        "- Switched to /sensors/{id}/measurements endpoint\n"
        "- This endpoint supports datetime_from filtering correctly\n"
        "- Added client-side hourly aggregation (group by floor(timestamp, 1h), compute mean)\n"
        "- Added safety cutoff filter: discard data older than requested window\n\n"
        "Verification: After fix, all 6 pollutants present, correct date range (last 14-24h), "
        "sensible forecast (AQI 99-114 from current 105)."
    )

    pdf.section("9.2 Missing PM10 Bug")
    pdf.body(
        "Problem: PM10 data was missing from API results despite stations having PM10 sensors.\n\n"
        "Root Cause: Sensor deduplication was not implemented, so the API client queried "
        "legacy (defunct) sensors first, consuming the 50-call rate limit before reaching "
        "active PM10 sensors.\n\n"
        "Fix: Implemented per-parameter deduplication -- for each location, keep only the sensor "
        "with the highest ID (newest hardware) per parameter. This ensured active sensors were "
        "queried first, and PM10 data appeared in results."
    )

    pdf.section("9.3 Column Name Mismatch")
    pdf.body(
        "Problem: features.py expected 'timestamp' but the CSV had 'Timestamp' (capital T).\n\n"
        "Fix: Added case-insensitive handling in load_and_clean():\n"
        "  if 'Timestamp' in df.columns:\n"
        "      df.rename(columns={'Timestamp': 'timestamp'}, inplace=True)"
    )

    pdf.section("9.4 LightGBM libomp Dependency (macOS)")
    pdf.body(
        "Problem: LightGBM import failed with 'libomp not found' on macOS.\n\n"
        "Fix: brew install libomp\n\n"
        "LightGBM uses OpenMP for parallel tree building. macOS does not ship with libomp "
        "by default (unlike Linux which includes it in most distros)."
    )

    pdf.section("9.5 Unicode in PDF Generation")
    pdf.body(
        "Problem: generate_report.py crashed with FPDFUnicodeEncodingException on em dashes.\n\n"
        "Root Cause: fpdf2's built-in Helvetica/Courier fonts use latin-1 encoding, which does "
        "not support Unicode characters like em dash (--), curly quotes, or Greek letters.\n\n"
        "Fix: Created _safe() helper that replaces all Unicode characters with ASCII equivalents "
        "before rendering. Maps: -- -> --, curly quotes -> straight quotes, Greek letters -> names, "
        "subscript numbers -> regular numbers."
    )

    pdf.section("9.6 CO Unit Confusion")
    pdf.body(
        "Problem: CO sub-index was wildly wrong for some stations.\n\n"
        "Root Cause: OpenAQ v3 reports CO as 'ppb' but the values (0.5-5.0) are clearly in ppm. "
        "True CO ppb for urban Delhi would be 500-10,000.\n\n"
        "Fix: Added heuristic threshold: if CO 'ppb' value < 100, treat as ppm and convert "
        "using factor 1.145 (ppm to mg/m3). Otherwise, use standard ppb conversion."
    )

    pdf.section("9.7 Target Leakage Prevention")
    pdf.body(
        "Problem: Initial rolling features used current row, leaking the target into inputs.\n\n"
        "Fix: All rolling windows shifted by 1 step:\n"
        "  df[target].shift(1).rolling(window=w, min_periods=1).mean()\n\n"
        "This ensures only past information is used. Same principle applied to all features: "
        "lags use shift(h), deltas use diff(h), no future data anywhere."
    )

    pdf.section("9.8 Stale Streamlit Cache")
    pdf.body(
        "Problem: Dashboard showed old data after API fix.\n\n"
        "Fix: @st.cache_data has TTL=600s (10 minutes). During development, the old cached "
        "results persisted. Killing the Streamlit process and restarting cleared the cache. "
        "Added 'Clear cache' button in sidebar for production use."
    )

    # ══════════════════════════════════════════════════════════════════
    # CHAPTER 10: DEPLOYMENT & CONFIGURATION
    # ══════════════════════════════════════════════════════════════════
    pdf.chapter(10, "Deployment & Configuration")

    pdf.section("10.1 Environment Setup")
    pdf.body(
        "Python 3.14 virtual environment:\n"
        "  python -m venv .venv\n"
        "  source .venv/bin/activate\n"
        "  pip install -r requirements.txt\n\n"
        "macOS dependency: brew install libomp (for LightGBM)"
    )

    pdf.section("10.2 Required API Keys")
    pdf.body(
        "1. OpenAQ API Key: Free at https://explore.openaq.org -> Sign up -> API Keys\n"
        "   Used for: X-API-Key header on all OpenAQ v3 requests\n\n"
        "2. Groq API Key: Free at https://console.groq.com\n"
        "   Used for: Llama 3.3 70B chat completions\n\n"
        "Keys can be provided via:\n"
        "- Environment variables: OPENAQ_API_KEY, GROQ_API_KEY\n"
        "- Streamlit sidebar text inputs\n"
        "- Streamlit Cloud secrets (.streamlit/secrets.toml)"
    )

    pdf.section("10.3 Running the Dashboard")
    pdf.code_block(
        "# Launch with environment variables:\n"
        'OPENAQ_API_KEY="<key>" GROQ_API_KEY="<key>" \\\n'
        "  .venv/bin/streamlit run app.py --server.port 8501"
    )

    pdf.section("10.4 Training the Forecasting Model")
    pdf.code_block(
        "# Run the training pipeline:\n"
        "python -m forecasting.train \\\n"
        "  --data /path/to/Delhi_AQI_Combined_2020_2024.csv\n"
        "\n"
        "# Outputs saved to forecasting/models/:\n"
        "#   best_model.pkl, feature_cols.json,\n"
        "#   regime_scaler.pkl, regime_kmeans.pkl, regime_models.pkl,\n"
        "#   training_results.json, shap_importance.json/png"
    )

    pdf.section("10.5 Key Dependencies (requirements.txt)")
    pdf.table(
        ["Package", "Purpose"],
        [
            ["streamlit", "Web dashboard framework"],
            ["plotly", "Interactive visualizations"],
            ["pandas, numpy", "Data manipulation"],
            ["lightgbm", "Gradient boosting (primary model)"],
            ["xgboost", "Gradient boosting (comparison)"],
            ["shap", "Model interpretability"],
            ["scikit-learn", "KMeans, metrics, preprocessing"],
            ["faiss-cpu", "Dense vector search"],
            ["sentence-transformers", "BGE embeddings + cross-encoder"],
            ["requests", "HTTP client for APIs"],
            ["fpdf2", "PDF report generation"],
            ["joblib", "Model serialization"],
        ],
        col_widths=[55, 135],
    )

    pdf.section("10.6 GitHub Repository")
    pdf.body(
        "Repository: https://github.com/Prachi-Tewari/delhi-aqi-dashboard\n"
        "Branch: main\n"
        "License: Open source\n\n"
        "The repository includes all source code, trained model artifacts, "
        "SHAP analysis outputs, and this documentation."
    )

    # ── Save ─────────────────────────────────────────────────────────
    out = os.path.join(PROJECT, "Delhi_AQI_Full_Technical_Report.pdf")
    pdf.output(out)
    print(f"\nReport generated: {out}")
    print(f"Pages: {pdf.page_no()}")
    return out


if __name__ == "__main__":
    build_report()
