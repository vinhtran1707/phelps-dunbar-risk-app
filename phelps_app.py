# ╔══════════════════════════════════════════════════════════════════════╗
# ║   PHELPS DUNBAR — ATTORNEY RISK ASSESSMENT APP                     ║
# ║   Run: streamlit run phelps_app.py                                  ║
# ╚══════════════════════════════════════════════════════════════════════╝

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.pipeline      import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model  import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Phelps Dunbar | Attorney Risk Assessment",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Merriweather:wght@400;700&family=Source+Sans+3:wght@300;400;500;600&display=swap');

html, body, [data-testid="stAppViewContainer"],
[data-testid="stMain"], .main, .block-container,
[class*="css"] {
    background-color: #F7F9FC !important;
    color: #1a2a3a !important;
    font-family: 'Source Sans 3', sans-serif !important;
}

input, textarea, select,
.stTextInput input, .stNumberInput input,
div[data-baseweb="input"] input,
div[data-baseweb="select"] div,
div[data-baseweb="select"] span,
[data-testid="stNumberInput"] input {
    background-color: #FFFFFF !important;
    color: #1a2a3a !important;
    border: 1px solid #D0DAEA !important;
    font-family: 'Source Sans 3', sans-serif !important;
    font-size: 0.88rem !important;
}

div[data-baseweb="select"] * { color: #1a2a3a !important; background-color: #FFFFFF !important; }
ul[data-baseweb="menu"] li, ul[role="listbox"] li { color: #1a2a3a !important; background-color: #FFFFFF !important; }
ul[data-baseweb="menu"] li:hover { background-color: #EBF1FB !important; }

label, .stSelectbox label, .stNumberInput label,
.stTextInput label, .stSlider label, .stCheckbox label,
[data-testid="stWidgetLabel"] p {
    color: #1F3864 !important;
    font-size: 0.80rem !important;
    font-weight: 600 !important;
    font-family: 'Source Sans 3', sans-serif !important;
}

.stCheckbox span { color: #1a2a3a !important; }
.stSlider [data-testid="stTickBarMin"], .stSlider [data-testid="stTickBarMax"], .stSlider p { color: #6B7C93 !important; }
#MainMenu, footer, header, [data-testid="collapsedControl"] { visibility: hidden; }

.block-container { padding: 1rem 2rem 2rem 2rem !important; max-width: 1400px !important; }

.app-header {
    background: #0047bb url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='120' height='104'%3E%3Cpolygon points='60,0 120,104 0,104' fill='none' stroke='rgba(255,255,255,0.12)' stroke-width='1'/%3E%3Cpolygon points='0,0 60,104 120,0' fill='none' stroke='rgba(255,255,255,0.12)' stroke-width='1'/%3E%3Cpolygon points='0,0 120,0 60,104' fill='none' stroke='rgba(255,255,255,0.08)' stroke-width='1'/%3E%3C/svg%3E") repeat;
    background-size: 120px 104px;
    border-radius: 8px;
    padding: 1.4rem 2.2rem;
    margin-bottom: 1.6rem;
    display: flex;
    align-items: center;
    gap: 1.5rem;
    border-bottom: 4px solid #C8993A;
    box-shadow: 0 2px 12px rgba(0,71,187,0.25);
}
.firm-name { font-family: 'Helvetica', serif; font-size: 1.8rem; font-weight: 700; color: #FFFFFF !important; margin: 0; }
.hdr-sub { font-size:0.76rem; color:#E8B85A !important; letter-spacing:0.14em; text-transform:uppercase; margin:0 0 4px 0; font-weight:600; }
.hdr-meta { font-size:0.80rem; color:rgba(255,255,255,0.60) !important; margin:0; }
.hdr-proto { font-size:0.70rem; color:rgba(255,255,255,0.40); margin:4px 0 0 0; font-style:italic; }
.gold-bar { width:2px; height:48px; background:#C8993A; opacity:0.7; border-radius:1px; }

.step-label {
    font-size:0.73rem; font-weight:700; color:#1F3864; letter-spacing:0.12em; text-transform:uppercase;
    background:#EBF1FB; border-left:4px solid #1F3864; padding:0.35rem 0.9rem;
    margin-bottom:1rem; margin-top:0.4rem; border-radius:0 4px 4px 0;
}

.input-divider { border:none; border-top:1px solid #D0DAEA; margin:0.8rem 0; }

.risk-card { border-radius:10px; padding:1.6rem 1.2rem; text-align:center; border:2px solid; box-shadow:0 2px 10px rgba(0,0,0,0.08); }
.risk-card.high { background:#FFF0EF; border-color:#C0392B; }
.risk-card.low  { background:#F0FBF5; border-color:#1E7A4A; }
.risk-label { font-family:'Merriweather',serif; font-size:1.6rem; font-weight:700; margin-bottom:0.2rem; }
.risk-label.high { color:#C0392B; }
.risk-label.low  { color:#1E7A4A; }
.risk-pct  { font-size:3rem; font-weight:300; line-height:1; margin-bottom:0.2rem; }
.risk-pct.high { color:#C0392B; }
.risk-pct.low  { color:#1E7A4A; }
.risk-sub  { font-size:0.72rem; color:#6B7C93; text-transform:uppercase; letter-spacing:0.1em; }

.sumrow { display:flex; justify-content:space-between; align-items:center; padding:0.32rem 0.5rem; border-bottom:1px solid #EEF1F5; font-size:0.83rem; }

.drow { display:flex; align-items:center; gap:0.7rem; padding:0.45rem 0.7rem; border-radius:5px; margin-bottom:0.3rem; background:#F7F9FC; border:1px solid #D0DAEA; }
.dname { font-size:0.83rem; color:#1a2a3a; flex:1; }
.dval  { font-size:0.78rem; color:#6B7C93; }
.dpos  { color:#1E7A4A; font-weight:700; font-size:0.85rem; }
.dneg  { color:#C0392B; font-weight:700; font-size:0.85rem; }

.recbox { background:#FFFFFF; border:1px solid #D0DAEA; border-left:3px solid #2E4F8A; border-radius:5px; padding:0.7rem 0.9rem; font-size:0.84rem; margin-bottom:0.6rem; line-height:1.55; }
.recbox.warn { border-left-color:#C0392B; background:#FDECEA; }
.recbox.ok   { border-left-color:#1E7A4A; background:#EBF7F1; }
.recbox strong { color:#1F3864; }

.stButton > button {
    background: linear-gradient(135deg, #1F3864 0%, #2E4F8A 100%) !important;
    color: #FFFFFF !important; border: none !important; border-radius: 6px !important;
    font-weight: 600 !important; font-size: 0.92rem !important;
    letter-spacing: 0.06em !important; text-transform: uppercase !important;
    width: 100% !important; border-bottom: 3px solid #C8993A !important; padding: 0.7rem 1rem !important;
}

.footer-strip { background:#1F3864; border-radius:8px; padding:0.8rem 1.5rem; margin-top:1rem; display:flex; justify-content:space-around; align-items:center; flex-wrap:wrap; gap:0.5rem; }
.fi { text-align:center; }
.fl { font-size:0.65rem; color:rgba(255,255,255,0.45); text-transform:uppercase; letter-spacing:0.08em; }
.fv { font-size:0.85rem; color:#E8B85A; font-weight:600; }
/* Phelps blue sliders — full override */
[data-testid="stSlider"] [role="slider"] {
    background: #0047bb !important;
    border-color: #0047bb !important;
}
[data-testid="stSlider"] div[data-baseweb="slider"] > div > div > div {
    background: #0047bb !important;
}
[data-testid="stSlider"] div[data-baseweb="slider"] > div > div:first-child > div:first-child {
    background: #0047bb !important;
}
/* Override the red filled track */
[data-testid="stSlider"] [class*="Track"] > div {
    background-color: #0047bb !important;
}
/* Override the thumb dot */
[data-testid="stSlider"] [class*="Thumb"] {
    background-color: #0047bb !important;
    border-color: #0047bb !important;
    box-shadow: 0 0 0 3px rgba(0,71,187,0.2) !important;
}
/* Light unfilled track */
[data-testid="stSlider"] [class*="Track"] > div:last-child {
    background-color: #99bbee !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL — loads real trained LR model
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    import pickle, os
    model_path = 'best_model_lr90.pkl'
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        st.stop()
    with open(model_path, 'rb') as f:
        pipe = pickle.load(f)
    # Exact feature names from the real trained model
    # Verified by: pipe.named_steps['scaler'].feature_names_in_.tolist()
    feature_cols = [
        'billable_hours', 'worked_hours', 'worked_amounts', 'wip_hours',
        'matters_count', 'billable_ratio', 'daily_charge_hours_budget',
        'daily_bill_hours_budget', 'daily_bill_fees_budget',
        'daily_fees_coll_budget', 'daily_budget_completion', 'location_id',
        'practice_group_id', 'service_years', 'day_of_week', 'week_of_year',
        'quarter', 'day_of_month', 'is_monday', 'is_friday', 'is_month_end',
        'is_year_end', 'is_qtr_end', 'is_weekend', 'seniority_enc',
        'roll90_matters', 'roll90_worked_hrs', 'roll90_bill_ratio',
        'roll7_bill_ratio_cmp'
    ]
    return pipe, feature_cols

model, FCOLS = load_model()


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
LOC_ID = {
    "Location 101 — HQ (Reference)": 101, "Location 102": 102,
    "Location 103": 103, "Location 104": 104, "Location 105": 105,
    "Location 106": 106, "Location 107 — Top Performer": 107,
    "Location 109": 109, "Location 112": 112,
    "Location 117 — Structural Risk": 117, "Location 199 — Junior Pool": 199,
}
PG_ID = {
    "PG 1 (Reference)": 1, "PG 2 — Outperformer": 2, "PG 3": 3,
    "PG 4 — Underperformer": 4, "PG 6 — Top Performer": 6,
    "PG 11 — Structural Risk": 11,
}
SEN = {"Junior (0–2 yrs)":0,"Mid-Career (2–5 yrs)":1,"Senior (5–10 yrs)":2,"Partner (10+ yrs)":3}
DAY = {"Monday":0,"Tuesday":1,"Wednesday":2,"Thursday":3,"Friday":4}
QTR = {"Q1 — Jan–Mar":1,"Q2 — Apr–Jun":2,"Q3 — Jul–Sep":3,"Q4 — Oct–Dec":4}

FN = {
    'billable_hours':"Today's billable hrs", 'worked_hours':"Today's worked hrs",
    'worked_amounts':'Billed amounts ($)', 'wip_hours':'WIP hours',
    'matters_count':'Active matters today', 'billable_ratio':'90-day billing ratio',
    'daily_charge_hours_budget':'Daily charge budget',
    'daily_budget_completion':'Budget completion %',
    'location_id':'Office location', 'practice_group_id':'Practice group',
    'service_years':'Service years', 'seniority_enc':'Seniority',
    'roll90_matters':'90-day matter avg', 'roll90_worked_hrs':'90-day worked hrs avg',
    'roll90_bill_ratio':'90-day billing ratio', 'roll7_bill_ratio_cmp':'7-day billing ratio',
    'quarter':'Quarter', 'day_of_week':'Day of week',
    'is_month_end':'Month-end flag', 'is_year_end':'Year-end flag',
    'is_qtr_end':'Quarter-end flag',
}

# Mid-quarter week estimates
WEEK_EST = {1: 8, 2: 21, 3: 34, 4: 47}


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE BUILDER — matches exact model training columns
# ─────────────────────────────────────────────────────────────────────────────
def bfeat(inp):
    r = {col: 0.0 for col in FCOLS}
    r['billable_hours']            = inp['billable_hours']
    r['worked_hours']              = inp['worked_hours']
    r['worked_amounts']            = inp['billable_hours'] * 350
    r['wip_hours']                 = inp['wip_hours']
    r['matters_count']             = inp['roll90_matters']
    r['billable_ratio']            = inp['roll90_bill_ratio']
    r['daily_charge_hours_budget'] = inp['daily_charge_hours_budget']
    r['daily_bill_hours_budget']   = inp['daily_charge_hours_budget'] * 0.9
    r['daily_bill_fees_budget']    = inp['daily_charge_hours_budget'] * 350
    r['daily_fees_coll_budget']    = inp['daily_charge_hours_budget'] * 300
    r['daily_budget_completion']   = (
        inp['billable_hours'] / inp['daily_charge_hours_budget']
        if inp['daily_charge_hours_budget'] > 0 else 0.0
    )
    r['location_id']               = inp['location_id']
    r['practice_group_id']         = inp['practice_group_id']
    r['service_years']             = inp['service_years']
    r['day_of_week']               = inp['day_of_week']
    r['week_of_year']              = inp['week_of_year']
    r['quarter']                   = inp['quarter']
    r['day_of_month']              = inp['day_of_month']
    r['is_monday']                 = 1.0 if inp['day_of_week'] == 0 else 0.0
    r['is_friday']                 = 1.0 if inp['day_of_week'] == 4 else 0.0
    r['is_month_end']              = 1.0 if inp['is_month_end'] else 0.0
    r['is_year_end']               = 1.0 if inp['is_year_end'] else 0.0
    r['is_qtr_end']                = 1.0 if inp['is_qtr_end'] else 0.0
    r['is_weekend']                = 0.0
    r['seniority_enc']             = inp['seniority_enc']
    r['roll90_matters']            = inp['roll90_matters']
    r['roll90_worked_hrs']         = inp['roll90_worked_hrs']
    r['roll90_bill_ratio']         = inp['roll90_bill_ratio']
    r['roll7_bill_ratio_cmp']      = inp['roll7_bill_ratio']
    return pd.DataFrame([r])[FCOLS]

def pred(inp):
    X  = bfeat(inp)
    ph = model.predict_proba(X)[0][1]
    return ph, 1-ph, ("HIGH RISK" if ph < 0.5 else "LOW RISK")

def drivers(inp):
    X  = bfeat(inp)
    sc = model.named_steps.get('scaler') or model.named_steps.get('sc')
    lr = model.named_steps.get('model')  or model.named_steps.get('lr')
    co = lr.coef_[0]
    Xs = sc.transform(X)
    out = [{'f':f,'v':X.iloc[0][f],'c':float(Xs[0][i]*c),
             'd':'pos' if Xs[0][i]*c>0 else 'neg'}
           for i,(f,c) in enumerate(zip(FCOLS,co)) if abs(Xs[0][i]*c)>0.01]
    return sorted(out, key=lambda x: abs(x['c']), reverse=True)[:8]


# ─────────────────────────────────────────────────────────────────────────────
# PAGE HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <div>
    <p class="hdr-sub">HR Analytics Platform</p>
    <h1 class="firm-name">Phelps Dunbar LLP</h1>
  </div>
  <div class="gold-bar"></div>
  <div>
    <p class="hdr-sub">Attorney Risk Assessment &nbsp;(Concept by Vinh Tran &amp; Reese Hathaway)</p>
    <p class="hdr-meta">90-day Logistic Regression &nbsp;·&nbsp; 75.7% Accuracy &nbsp;·&nbsp; AUC 0.842</p>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="step-label">⚖️  Step 1 — Enter Attorney Details</div>',
            unsafe_allow_html=True)

# Row 1
a, b, c, d, e = st.columns([1.6, 1.8, 1.5, 1.2, 1.0])
atty  = a.text_input("Attorney Name (optional)", placeholder="e.g. J. Smith")
loc_l = b.selectbox("Office Location",  list(LOC_ID.keys()))
pg_l  = c.selectbox("Practice Group",   list(PG_ID.keys()))
sen_l = d.selectbox("Seniority Level",  list(SEN.keys()))
svc   = e.number_input("Service Years", 0.0, 40.0, 5.0, 0.5)

st.markdown('<hr class="input-divider">', unsafe_allow_html=True)


# Row 2
f2, g2, h2, i2, j2 = st.columns([1.4, 1.4, 0.9, 0.9, 0.9])
day_l = f2.selectbox("Day of Week", list(DAY.keys()))
qtr_l = g2.selectbox("Quarter",    list(QTR.keys()))

me = h2.checkbox("Month-End")
h2.markdown("<small style='color:#2E4F8A;font-size:0.72rem'>📅 Last 3 days of month.<br>Slightly positive signal.</small>", unsafe_allow_html=True)

ye = i2.checkbox("Year-End")
i2.markdown("<small style='color:#2E4F8A;font-size:0.72rem'>📅 December only.<br>Strongest month (50.7%).</small>", unsafe_allow_html=True)

qe = j2.checkbox("Qtr-End")
j2.markdown("<small style='color:#2E4F8A;font-size:0.72rem'>📅 Last days of Mar/<br>Jun/Sep/Dec.</small>", unsafe_allow_html=True)

st.markdown('<hr class="input-divider">', unsafe_allow_html=True)

# Row 3
k3, l3, m3, n3 = st.columns(4)
wh = k3.number_input("Hours Worked Today",       0.0, 24.0,  8.5, 0.5)
bh = l3.number_input("Billable Hours Today",      0.0, 24.0,  7.0, 0.5)
wp = m3.number_input("WIP Hours (unbilled)", 0.0, 100.0, 10.0, 0.5,
       help="Billable hours completed but not yet invoiced. "
            "High WIP = backlog ready to convert. Healthy: 8+ hrs.")
db = n3.number_input("Daily Charge Hours Budget", 0.0, 15.0,  8.0, 0.5)

st.markdown('<hr class="input-divider">', unsafe_allow_html=True)

# Row 4 — number input + slider for each field
o4, p4, q4, r4 = st.columns(4)

with o4:
    rw = st.number_input("90-Day Avg Worked Hrs/Day", 0.0, 14.0, 9.5, 0.1,
                         help="Avg total hours worked per day over 90 days. "
                              "The #1 predictor in the model. Healthy: 9.0+ hrs/day.")
    rw = st.slider("##rw", 0.0, 14.0, rw, 0.1, label_visibility="collapsed")

with p4:
    rb = st.number_input("90-Day Avg Billing Ratio", 0.0, 1.0, 0.78, 0.01, format="%.2f",
                         help="% of worked hours billed to clients, averaged over 90 days. "
                              "Healthy: 70%+. Below 70% triggers a recommendation.")
    rb = st.slider("##rb", 0.0, 1.0, rb, 0.01, label_visibility="collapsed")

with q4:
    rm = st.number_input("90-Day Avg Matter Count/Day", 0.0, 15.0, 5.0, 0.1,
                         help="Avg active client matters billed per day over 90 days. "
                              "Below 4.5 is the critical threshold — thin pipeline drives most misses.")
    rm = st.slider("##rm", 0.0, 15.0, rm, 0.1, label_visibility="collapsed")

with r4:
    r7 = st.number_input("7-Day Avg Billing Ratio", 0.0, 1.0, 0.75, 0.01, format="%.2f",
                         help="Short-term billing trend over the past 7 days. "
                              "A drop here vs the 90-day avg is an early warning sign.")
    r7 = st.slider("##r7", 0.0, 1.0, r7, 0.01, label_visibility="collapsed")

st.markdown('<hr class="input-divider">', unsafe_allow_html=True)

_, bc, _ = st.columns([2.5, 1, 2.5])
with bc:
    st.button("⚖️  ASSESS RISK NOW")


# ─────────────────────────────────────────────────────────────────────────────
# BUILD INPUT DICT
# ─────────────────────────────────────────────────────────────────────────────
inp = dict(
    billable_hours          = bh,
    worked_hours            = wh,
    wip_hours               = wp,
    roll90_matters          = rm,
    roll90_worked_hrs       = rw,
    roll90_bill_ratio       = rb,
    roll7_bill_ratio        = r7,
    daily_charge_hours_budget = db,
    service_years           = svc,
    seniority_enc           = SEN[sen_l],
    day_of_week             = DAY[day_l],
    quarter                 = QTR[qtr_l],
    week_of_year            = WEEK_EST[QTR[qtr_l]],
    day_of_month            = 15,
    is_month_end            = me,
    is_year_end             = ye,
    is_qtr_end              = qe,
    location_id             = LOC_ID[loc_l],
    practice_group_id       = PG_ID[pg_l],
)


# ─────────────────────────────────────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────────────────────────────────────
ph, pm, rk = pred(inp)
dvs        = drivers(inp)
hi         = (rk == "HIGH RISK")
cls        = "high" if hi else "low"
pct        = f"{pm:.0%}" if hi else f"{ph:.0%}"
sub        = "probability of missing goal" if hi else "probability of hitting goal"
nh         = (f"<br><span style='font-size:0.80rem;color:#6B7C93'>{atty}</span>"
              if atty else "")

st.markdown('<div class="step-label">📊  Step 2 — Risk Assessment Results</div>',
            unsafe_allow_html=True)

r1, r2, r3 = st.columns([1.0, 1.1, 1.9])

with r1:
    lbl = "HIGH RISK" if hi else "LOW RISK"
    st.markdown(
        f'<div class="risk-card {cls}">'
        f'<div class="risk-label {cls}">{lbl}</div>{nh}'
        f'<div class="risk-pct {cls}">{pct}</div>'
        f'<div class="risk-sub">{sub}</div></div>',
        unsafe_allow_html=True
    )

with r2:
    gc  = "#C0392B" if hi else "#1E7A4A"
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=round(pm*100, 1),
        number={'suffix':'%','font':{'size':26,'color':gc,'family':'Source Sans 3'}},
        gauge={
            'axis':{'range':[0,100],'tickfont':{'color':'#6B7C93','size':9}},
            'bar':{'color':gc,'thickness':0.28},
            'bgcolor':'#F7F9FC','bordercolor':'#D0DAEA',
            'steps':[{'range':[0,35],'color':'#EBF7F1'},
                     {'range':[35,65],'color':'#FFFBE6'},
                     {'range':[65,100],'color':'#FDECEA'}],
            'threshold':{'line':{'color':gc,'width':3},'thickness':0.8,'value':pm*100}
        },
        title={'text':'Goal Miss Probability',
               'font':{'color':'#6B7C93','size':10,'family':'Source Sans 3'}}
    ))
    fig.update_layout(height=200, margin=dict(t=35,b=0,l=5,r=5),
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

with r3:
    st.markdown('<p style="font-size:0.72rem;font-weight:700;color:#1F3864;'
                'letter-spacing:0.1em;text-transform:uppercase;'
                'border-bottom:2px solid #C8993A;padding-bottom:0.3rem;'
                'margin-bottom:0.6rem">📋 Input Summary</p>',
                unsafe_allow_html=True)
    bt = bh/wh if wh > 0 else 0
    bg = wh - db
    for lbl2, val, ok in [
        ("90-Day Worked Avg",     f"{rw:.1f} hrs/day", rw >= 9.0),
        ("90-Day Billing Ratio",  f"{rb:.0%}",          rb >= 0.75),
        ("90-Day Matter Avg",     f"{rm:.1f}/day",      rm >= 4.5),
        ("Today's Billing Ratio", f"{bt:.0%}",          bt >= 0.75),
        ("WIP Backlog",           f"{wp:.1f} hrs",      wp >= 8.0),
        ("vs Daily Budget",       f"{bg:+.1f} hrs",     bg >= 0),
    ]:
        col = "#1E7A4A" if ok else "#C0392B"
        ic  = "✓" if ok else "✗"
        st.markdown(
            f'<div class="sumrow"><span style="color:#6B7C93">{lbl2}</span>'
            f'<span style="color:{col};font-weight:600">{ic} {val}</span></div>',
            unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

d1, d2 = st.columns([1.4, 1.6])

with d1:
    st.markdown('<p style="font-size:0.72rem;font-weight:700;color:#1F3864;'
                'letter-spacing:0.1em;text-transform:uppercase;'
                'border-bottom:2px solid #C8993A;padding-bottom:0.3rem;'
                'margin-bottom:0.5rem">🔍 Key Risk Drivers</p>',
                unsafe_allow_html=True)
    st.caption("+ helps goal  /  − hurts goal")
    for d in dvs:
        sign = "+" if d['d'] == 'pos' else "−"
        cls2 = "dpos" if d['d'] == 'pos' else "dneg"
        ic   = "📈" if d['d'] == 'pos' else "📉"
        fn   = FN.get(d['f'], d['f'].replace('_', ' ').title())
        vs   = f"{d['v']:.2f}" if isinstance(d['v'], float) else str(d['v'])
        st.markdown(
            f'<div class="drow"><span>{ic}</span>'
            f'<span class="dname">{fn}</span>'
            f'<span class="dval">{vs}</span>'
            f'<span class="{cls2}">{sign}{abs(d["c"]):.2f}</span></div>',
            unsafe_allow_html=True)

with d2:
    st.markdown('<p style="font-size:0.72rem;font-weight:700;color:#1F3864;'
                'letter-spacing:0.1em;text-transform:uppercase;'
                'border-bottom:2px solid #C8993A;padding-bottom:0.3rem;'
                'margin-bottom:0.5rem">💡 HR Recommendations</p>',
                unsafe_allow_html=True)
    recs = []
    if rm < 4.0:
        recs.append(("⚠️","Matter Pipeline Thin",
            f"90-day avg {rm:.1f} matters/day is below the 4.5 threshold. "
            "Review matter allocation and cross-office staffing.","warn"))
    if rb < 0.70:
        recs.append(("⚠️","Billing Efficiency Below Target",
            f"90-day billing ratio {rb:.0%} is below 70%. Schedule a billing "
            "review and check for unbilled WIP accumulation.","warn"))
    if rw < 8.0:
        recs.append(("⚠️","Hours Below Firm Average",
            f"90-day worked avg {rw:.1f} hrs/day is below average. Investigate "
            "whether this reflects insufficient matter flow.","warn"))
    if wp < 5.0:
        recs.append(("⚠️","Low WIP Backlog",
            f"Only {wp:.1f} hrs of unbilled WIP. Prioritize billing cadence "
            "to build a healthier backlog.","warn"))
    if LOC_ID[loc_l] == 117:
        recs.append(("📍","Location 117 Structural Flag",
            "This office carries a structural penalty (LR coef -0.316) "
            "independent of matter volume. Recommend qualitative review.","warn"))
    if PG_ID[pg_l] in [4, 11]:
        recs.append(("⚖️","Practice Group Structural Risk",
            f"{pg_l} shows consistent underperformance. Consider quarterly "
            "goal-setting for irregular matter flow.","warn"))
    if QTR[qtr_l] == 1:
        recs.append(("📅","Q1 Intervention Window",
            "Q1 has the lowest firm-wide attainment (42.8%). Highest-ROI "
            "period for proactive check-ins.","warn"))
    if not recs:
        recs.append(("✅","No Immediate Concerns",
            "All inputs are within healthy ranges. Continue monitoring "
            "90-day rolling averages monthly.","ok"))
    for ic2, tt, bb, st2 in recs[:3]:
        st.markdown(
            f'<div class="recbox {st2}"><strong>{ic2} {tt}</strong><br>'
            f'<span style="font-size:0.82rem;color:#333">{bb}</span></div>',
            unsafe_allow_html=True)

st.markdown("""
<div class="footer-strip">
  <div class="fi"><div class="fl">Model</div><div class="fv">Logistic Regression</div></div>
  <div class="fi"><div class="fl">Window</div><div class="fv">90-day rolling</div></div>
  <div class="fi"><div class="fl">Accuracy</div><div class="fv">75.7%</div></div>
  <div class="fi"><div class="fl">AUC</div><div class="fv">0.842</div></div>
  <div class="fi"><div class="fl">Features</div><div class="fv">29 VIF-cleaned</div></div>
  <div class="fi"><div class="fl">Training</div><div class="fv">2015–2022</div></div>
</div>
<p style="text-align:center;font-size:0.68rem;color:#9AAABB;margin-top:0.5rem">
  Decision-support tool only. Trained on synthesized Phelps Dunbar dataset.
</p>
""", unsafe_allow_html=True)
