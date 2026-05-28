# District Coverage Agent — GTM FY27

An agentic RAG bot that answers questions about RPG Enterprises' district coverage data. Runs on Slack, deployed on Railway, powered by GPT-4.1.

---

## What It Can Answer

**Zone & Region level**
- Zone wise DSE / DT / Feeder addition plans
- Top zones or regions by any metric
- Region wise breakdown within a zone
- Compare zones by SDs, DSEs, feeders, DTs

**District level**
- Coverage status of a specific district
- Which districts have DT / Feeder / DSE appointment plans
- Which month are additions planned for a district
- Metro DT and Super DT plans by district

**Coverage gaps**
- Districts with no DTs
- Districts with no SDs in FY26
- Districts with zero DSEs
- Districts with no appointment plan at all

**Custom queries**
- Any combination of the above filtered by zone, region, or month
- Example: *"Which districts in East Zone are adding 1 DSE in April?"*

---

## How to Use in Slack

**Ask in a channel** — mention the bot with your question:
```
@gtm_bot What is the zone wise DSE addition plan?
@gtm_bot Show coverage gaps in North Zone 1
@gtm_bot Which districts in South Zone 1 have a DT appointment plan?
@gtm_bot Top 5 regions by feeder additions
```

**Ask via DM** — open a direct message with the bot and type your question directly (no need to @mention).

**Post answer to channel** — add any of these phrases to share the response with everyone:
```
@gtm_bot Zone wise DT addition plan — share with the team
@gtm_bot Feeder appointment plan, post in channel
@gtm_bot Top regions by DSE additions, let everyone see this
```

**Follow-up questions** — the bot remembers context within a conversation:
```
@gtm_bot Tell me region wise DSE addition plan in East Zone
> [bot replies]
@gtm_bot What about Asansol?
> [bot understands you mean East Zone > Asansol]
```

---

## Project Structure

```
GTM/
├── agents/
│   ├── district_coverage_agent.py   # Core ReAct agent + 10 tools
│   ├── slack_bot.py                 # Flask webhook server for Slack
│   └── __init__.py
├── data/
│   └── district_coverage/
│       └── district_coverage.parquet
├── requirements.txt
├── Procfile                         # Railway start command
└── .gitignore
```

---

## Agent Tools

| Tool | Purpose |
|---|---|
| `think` | ReAct reasoning step before every action |
| `load_data` | Schema + zones/regions overview |
| `get_summary` | High-level stats across all 618 districts |
| `filter_by_geography` | Filter by zone / region / district name |
| `get_dt_appointment_plan` | DT plans with month, Metro/Super flags |
| `get_feeder_plan` | Feeder addition plans with zone & region summaries |
| `get_dse_plan` | DSE addition plans with zone & region summaries |
| `aggregate_by_zone_or_region` | Rollup of all key metrics by zone or region |
| `search_district` | Free-text search on district name or DT code |
| `get_coverage_gaps` | Districts with zero DTs / SDs / DSEs / no plan |
| `run_pandas` | Arbitrary pandas queries for complex questions |

---

## Local Setup

**1. Clone the repo**
```bash
git clone git@github.com:anupriyomandal/gtm_bot.git
cd gtm_bot
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Create `agents/.env`**
```
OPENAI_API_KEY=sk-...
SLACK_BOT_TOKEN=xoxb-...
SLACK_SIGNING_SECRET=...
```

**4. Run locally**
```bash
python agents/district_coverage_agent.py
```
Type questions directly in the terminal to test without Slack.

---

## Slack App Setup

1. Go to [api.slack.com/apps](https://api.slack.com/apps) → Create New App
2. **OAuth & Permissions** → add Bot Token Scopes:
   - `app_mentions:read`, `chat:write`, `im:history`, `reactions:write`, `reactions:read`
3. Install to Workspace → copy `xoxb-...` Bot Token
4. **Basic Information** → copy Signing Secret
5. **App Home** → enable Messages Tab → allow users to send messages
6. **Event Subscriptions** → enable → set Request URL:
   ```
   https://your-railway-url.up.railway.app/slack/events
   ```
7. Subscribe to bot events: `app_mention`, `message.im`
8. Save Changes → reinstall app when prompted

---

## Railway Deployment

1. Go to [railway.app](https://railway.app) → New Project → Deploy from GitHub
2. Select `gtm_bot` repo
3. Add environment variables under **Variables**:
   ```
   OPENAI_API_KEY
   SLACK_BOT_TOKEN
   SLACK_SIGNING_SECRET
   ```
4. Railway auto-detects the `Procfile` and deploys
5. Copy the generated URL and paste into Slack Event Subscriptions

---

## Updating the Data

To refresh the district coverage data:

1. Replace `data/district_coverage.xlsx` with the updated file
2. Run the conversion script:
```bash
python -c "
import pandas as pd, re
from datetime import datetime, timedelta

MONTH_MAP = {
    'jan':'Jan','feb':'Feb','mar':'Mar','apr':'Apr','may':'May','jun':'Jun',
    'jul':'Jul','aug':'Aug','sep':'Sep','oct':'Oct','nov':'Nov','dec':'Dec',
    'january':'Jan','february':'Feb','march':'Mar','april':'Apr',
    'june':'Jun','july':'Jul','august':'Aug','september':'Sep',
    'october':'Oct','november':'Nov','december':'Dec',
}

def clean_month(val):
    if pd.isna(val): return None
    v = str(val).strip()
    if v in ('nan','-','0','N',''): return None
    if re.fullmatch(r'\d{5}', v):
        return (datetime(1899,12,30) + __import__('datetime').timedelta(days=int(v))).strftime('%b-%y')
    if re.match(r'^\d{4}-\d{2}-\d{2}', v):
        return pd.to_datetime(v).strftime('%b-%y')
    m = re.match(r\"^([A-Za-z]+)['.\\-\\s]+'?(\\d{2,4})$\", v)
    if m:
        abbr = MONTH_MAP.get(m.group(1).lower())
        yr = m.group(2)[-2:]
        if abbr: return f'{abbr}-{yr}'
    if re.fullmatch(r'[A-Za-z]+', v):
        abbr = MONTH_MAP.get(v.lower())
        if abbr: return f'{abbr}-26'
    return None

df = pd.read_excel('data/district_coverage.xlsx', sheet_name=0)
df = df.drop(columns=[c for c in ['Unnamed: 21','District.1','FY25','FY26'] if c in df.columns])
df.columns = [c.strip() for c in df.columns]
for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].astype(str)
df['Current No of DSEs'] = pd.to_numeric(df['Current No of DSEs'].str.strip(), errors='coerce')
df['No of DSEs Addition Plan (1,2,3 etc)'] = pd.to_numeric(df['No of DSEs Addition Plan (1,2,3 etc)'].replace('-', None), errors='coerce')
for col in ['DT Appointment Plan (Y/N)','Metro DT Appointment Plan (Y/N)','Super DT Appointment Plan (Y/N)']:
    df[col] = df[col].apply(lambda v: 'Y' if str(v).strip().upper() in ('Y','YES','1') else ('N' if str(v).strip().upper() in ('N','NO','0') else None))
for col in ['Month of DSE Appoinment Plan','Month of DT Appoinment Plan','Month of Appoinment Plan']:
    df[col] = df[col].apply(clean_month)
df.to_parquet('data/district_coverage/district_coverage.parquet', index=False, engine='pyarrow')
print('Done:', df.shape)
"
```
3. Commit and push — Railway redeploys automatically

---

## Tech Stack

- **LLM**: GPT-4.1 (agent), GPT-4.1-mini (intent classifier)
- **Agent pattern**: ReAct (Reasoning + Acting) with 11 tools
- **Data**: Parquet via pandas + pyarrow
- **Web server**: Flask + Gunicorn
- **Slack SDK**: slack-bolt
- **Deployment**: Railway (auto-deploy from GitHub)
