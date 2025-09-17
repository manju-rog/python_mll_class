# Deterministic, ruthless synthetic data generator (no LLM required).
# Produces thousands of labeled texts with typos, code-switching, emojis, and noise.
import csv, random, itertools, argparse, os, json, re
random.seed(42)

EMPLOYEES = [
  "Manju","Rahul","Priya","Aisha","Vikram","Kiran","Anil","Deepa","Rohit","Neha","Arjun","Meera",
  "Sanjay","Nitin","Pooja","Raj","Sara","Aman","Isha","Ravi","Sneha","Gaurav"
]
DATE_PHRASES = [
  "today","yesterday","tomorrow","on Monday","on Friday","last week","this week","next week","last month","next month",
  "between 1st and 7th","between 2 Jan and 7 Jan","in August","for Q3","for the past 2 weeks","for the last 10 days","on 2025-09-01"
]

# Template library
ABSENCE_TPL = [
  "who was absent {when}",
  "show absences {when}",
  "who is on leave {when}",
  "attendance summary {when}",
  "absent list {when}",
  "who was out {when}",
  "leave status for {when}",
  "who took PTO {when}",
  "who all were out {when}",
  "anyone on sick leave {when}",
  "attendance report {when}",
  "who had PTO {when}"
]
ABSENCE_ACT_TPL = [
  "mark {name} absent {when}",
  "put {name} on leave {when}",
  "set {name} as absent {when}",
  "record {name} as on leave {when}",
  "log {name} as absent {when}",
  "flag {name} on PTO {when}",
]

TDO_TPL = [
  "draft a TDO for {topic}",
  "create a technical design outline for {topic}",
  "prepare an architecture doc for {topic}",
  "draft spec for {topic}",
  "draft HLD for {topic}",
  "sketch technical design for {topic}",
  "outline architecture for {topic}"
]
COST_TPL = [
  "estimate cost for {topic}",
  "cost prediction for {topic}",
  "forecast budget for {topic}",
  "predict infra costs for {topic}",
  "what is the budget for {topic}",
  "ballpark costs for {topic}",
  "rough budget for {topic}",
  "TCO estimate for {topic}",
  "capex/opex for {topic}"
]

TOPICS = [
  "invoice microservice","absence API","payroll service","React front-end","Kubernetes migration",
  "data pipeline","authentication service","notification system","reporting module","mobile app",
  "ETL pipeline","chat service","billing gateway","inventory system","reservation API","analytics dashboard"
]

OOS = [
  # General
  "what's the weather in Paris","tell me a joke","solve 2+2","play music","book a flight",
  "who won the cricket match","translate hola to english","open youtube","how tall is Mount Everest",
  "random gibberish zqxv kjh","😀😀😀","/help /start","<script>alert(1)</script>",
  # Absence decoys (negatives)
  "leave the page","please leave me alone","left join sql example","how to leave a review","paid leave policy pdf",
  "maternity leave policy pdf","bereavement leave policy doc","absent minded professor movie","absent minded today","absentee ballot",
  "absenteeism rate by department","absenteeism policy pdf","absenteeism dashboard metrics",
  # Deeper decoys
  "leave-one-out cross validation","loo cv example","l1o validation",
  # Cost decoys (negatives)
  "costco membership price","costume ideas for halloween","cost of living in NYC","iPhone cost in india","absorption cost accounting",
  "cost center codes","cost center mapping",
  "absorption costing example","absorption costing vs variable costing",
  # TDO decoys (negatives)
  "todo list for tomorrow","to-do checklist app","what does TDO stand for","to-do vs TDO difference",
  # Account/Other
  "logout please","sign out","delete my account","cancel subscription"
]

def inject_typos(text, prob=0.15):
  def typo(word):
    if len(word) < 4: return word
    i = random.randint(1, len(word)-2)
    return word[:i] + word[i+1] + word[i] + word[i+2:]
  out = []
  for w in text.split():
    if random.random() < prob:
      out.append(typo(w))
    else:
      out.append(w)
  return " ".join(out)

def code_switch(text):
  repl = {
    "who": ["kaun"],
    "absent": ["absent", "chutti"],
    "leave": ["chutti"],
    "cost": ["kharcha"],
    "estimate": ["andaza"],
    "draft": ["banado"],
  }
  words = text.split()
  for i,w in enumerate(words):
    lw = w.lower().strip(",.?")
    if lw in repl and random.random() < 0.3:
      words[i] = random.choice(repl[lw])
  return " ".join(words)

def add_noise(text):
  prefixes = ["pls","hey","bro","yo","assistant","hi"]
  suffixes = ["thanks","ok","urgent","asap","🚀","🙏"]
  s = text
  if random.random() < 0.4: s = random.choice(prefixes) + " " + s
  if random.random() < 0.4: s = s + " " + random.choice(suffixes)
  if random.random() < 0.2: s = s.capitalize()
  if random.random() < 0.2: s = s.upper()
  return s

def gen():
  rows = []
  # Absence (query)
  for tpl in ABSENCE_TPL:
    for when in DATE_PHRASES:
      q = tpl.format(when=when)
      rows.append((q, "absence"))
  # Absence (action-like)
  for tpl in ABSENCE_ACT_TPL:
    for name in EMPLOYEES:
      for when in ["today","tomorrow","on 2025-09-01"]:
        rows.append((tpl.format(name=name, when=when), "absence"))
  # TDO
  for tpl in TDO_TPL:
    for topic in TOPICS:
      rows.append((tpl.format(topic=topic), "tdo_drafting"))
  # Cost
  for tpl in COST_TPL:
    for topic in TOPICS:
      rows.append((tpl.format(topic=topic), "cost_estimation"))
  # OOS
  for t in OOS:
    rows.append((t, "OUT_OF_SCOPE"))

  # Augment
  aug = []
  for (text, label) in rows:
    for _ in range(random.randint(2,5)):
      t = text
      if random.random() < 0.5: t = inject_typos(t)
      if random.random() < 0.5: t = code_switch(t)
      if random.random() < 0.8: t = add_noise(t)
      aug.append((t, label))
  rows += aug
  random.shuffle(rows)
  return rows

if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("--out", default="data/synth_dataset.csv")
  rows = gen()
  os.makedirs(os.path.dirname(ap.parse_args().out), exist_ok=True)
  with open(ap.parse_args().out, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["text","label"])
    w.writerows(rows)
  print(f"wrote {len(rows)} examples to {ap.parse_args().out}")
