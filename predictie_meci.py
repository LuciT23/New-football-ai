import pandas as pd 
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

# 1. Încarcă datele istorice
df = pd.read_csv("ROU.csv")
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
df = df.sort_values('Date').reset_index(drop=True)
df = df.dropna(subset=['Res', 'AvgCH', 'AvgCD', 'AvgCA', 'HG', 'AG']).copy()

# Probabilități implicite
sum_prob = (1/df['AvgCH']) + (1/df['AvgCD']) + (1/df['AvgCA'])
df['prob_bookie_1'] = (1/df['AvgCH']) / sum_prob
df['prob_bookie_X'] = (1/df['AvgCD']) / sum_prob
df['prob_bookie_2'] = (1/df['AvgCA']) / sum_prob

# Calcul istoric
echipa_meciuri, h2h_dict = {}, {}
forma_gazda_gen, forma_oaspete_gen = [], []
forma_gazda_acasa, forma_oaspete_dep = [], []
gm_gazda, gp_gazda, gm_oaspete, gp_oaspete = [], [], [], []
h2h_g, h2h_o = [], []

for idx, row in df.iterrows():
    home, away = row['Home'], row['Away']
    hist_h = echipa_meciuri.get(home, [])[-5:]
    hist_a = echipa_meciuri.get(away, [])[-5:]
    hist_h_home = [m for m in echipa_meciuri.get(home, []) if m['locatie'] == 'H'][-5:]
    hist_a_away = [m for m in echipa_meciuri.get(away, []) if m['locatie'] == 'A'][-5:]
    pair = tuple(sorted([home, away]))
    hist_h2h = h2h_dict.get(pair, [])[-5:]
    
    forma_gazda_gen.append(sum([m['pts'] for m in hist_h])/len(hist_h) if hist_h else 1.0)
    forma_oaspete_gen.append(sum([m['pts'] for m in hist_a])/len(hist_a) if hist_a else 1.0)
    forma_gazda_acasa.append(sum([m['pts'] for m in hist_h_home])/len(hist_h_home) if hist_h_home else 1.0)
    forma_oaspete_dep.append(sum([m['pts'] for m in hist_a_away])/len(hist_a_away) if hist_a_away else 1.0)
    
    gm_gazda.append(sum([m['gm'] for m in hist_h])/len(hist_h) if hist_h else 1.0)
    gp_gazda.append(sum([m['gp'] for m in hist_h])/len(hist_h) if hist_h else 1.0)
    gm_oaspete.append(sum([m['gm'] for m in hist_a])/len(hist_a) if hist_a else 1.0)
    gp_oaspete.append(sum([m['gp'] for m in hist_a])/len(hist_a) if hist_a else 1.0)
    
    if hist_h2h:
        h2h_g.append(sum([1 for m in hist_h2h if m['w'] == home])/len(hist_h2h))
        h2h_o.append(sum([1 for m in hist_h2h if m['w'] == away])/len(hist_h2h))
    else:
        h2h_g.append(0.33)
        h2h_o.append(0.33)
        
    hg, ag = row['HG'], row['AG']
    p_h, p_a = (3,0) if hg > ag else ((1,1) if hg == ag else (0,3))
    w = home if hg > ag else ('Draw' if hg == ag else away)
    
    if home not in echipa_meciuri: echipa_meciuri[home] = []
    if away not in echipa_meciuri: echipa_meciuri[away] = []
    if pair not in h2h_dict: h2h_dict[pair] = []
    
    echipa_meciuri[home].append({'pts': p_h, 'gm': hg, 'gp': ag, 'locatie': 'H'})
    echipa_meciuri[away].append({'pts': p_a, 'gm': ag, 'gp': hg, 'locatie': 'A'})
    h2h_dict[pair].append({'w': w})

df['forma_gazda_gen'] = forma_gazda_gen
df['forma_oaspete_gen'] = forma_oaspete_gen
df['forma_gazda_acasa'] = forma_gazda_acasa
df['forma_oaspete_dep'] = forma_oaspete_dep
df['gm_gazda'] = gm_gazda
df['gp_gazda'] = gp_gazda
df['gm_oaspete'] = gm_oaspete
df['gp_oaspete'] = gp_oaspete
df['h2h_gazda'] = h2h_g
df['h2h_oaspete'] = h2h_o

features = [
    'forma_gazda_gen', 'forma_oaspete_gen', 'forma_gazda_acasa', 'forma_oaspete_dep',
    'gm_gazda', 'gp_gazda', 'gm_oaspete', 'gp_oaspete', 'h2h_gazda', 'h2h_oaspete',
    'prob_bookie_1', 'prob_bookie_X', 'prob_bookie_2', 'AvgCH', 'AvgCD', 'AvgCA'
]

res_map = {'H': 0, 'D': 1, 'A': 2}
df['target_solist'] = df['Res'].map(res_map)
df['target_peste25'] = (df['HG'] + df['AG'] > 2.5).astype(int)

X = df[features]
y_sol = df['target_solist']
y_p25 = df['target_peste25']

# Antrenare modele optimizate
model_sol = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.03, random_state=42)
model_sol.fit(X, y_sol)

model_p25 = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.03, random_state=42)
model_p25.fit(X, y_p25)

# 2. Funcție pentru generat predicții
def prezice_meci(echipa_h, echipa_a, cota1, cotaX, cota2):
    h_hist = echipa_meciuri.get(echipa_h, [])[-5:]
    a_hist = echipa_meciuri.get(echipa_a, [])[-5:]
    h_home = [m for m in echipa_meciuri.get(echipa_h, []) if m['locatie'] == 'H'][-5:]
    a_away = [m for m in echipa_meciuri.get(echipa_a, []) if m['locatie'] == 'A'][-5:]
    
    f_h_g = sum([m['pts'] for m in h_hist])/len(h_hist) if h_hist else 1.0
    f_a_g = sum([m['pts'] for m in a_hist])/len(a_hist) if a_hist else 1.0
    f_h_a = sum([m['pts'] for m in h_home])/len(h_home) if h_home else 1.0
    f_a_d = sum([m['pts'] for m in a_away])/len(a_away) if a_away else 1.0
    
    gm_h = sum([m['gm'] for m in h_hist])/len(h_hist) if h_hist else 1.0
    gp_h = sum([m['gp'] for m in h_hist])/len(h_hist) if h_hist else 1.0
    gm_a = sum([m['gm'] for m in a_hist])/len(a_hist) if a_hist else 1.0
    gp_a = sum([m['gp'] for m in a_hist])/len(a_hist) if a_hist else 1.0
    
    pair = tuple(sorted([echipa_h, echipa_a]))
    hist_h2h = h2h_dict.get(pair, [])[-5:]
    if hist_h2h:
        h2h_h = sum([1 for m in hist_h2h if m['w'] == echipa_h])/len(hist_h2h)
        h2h_a = sum([1 for m in hist_h2h if m['w'] == echipa_a])/len(hist_h2h)
    else:
        h2h_h, h2h_a = 0.33, 0.33
        
    sum_p = (1/cota1) + (1/cotaX) + (1/cota2)
    p1 = (1/cota1) / sum_p
    pX = (1/cotaX) / sum_p
    p2 = (1/cota2) / sum_p
    
    input_data = pd.DataFrame([{
        'forma_gazda_gen': f_h_g, 'forma_oaspete_gen': f_a_g,
        'forma_gazda_acasa': f_h_a, 'forma_oaspete_dep': f_a_d,
        'gm_gazda': gm_h, 'gp_gazda': gp_h,
        'gm_oaspete': gm_a, 'gp_oaspete': gp_a,
        'h2h_gazda': h2h_h, 'h2h_oaspete': h2h_a,
        'prob_bookie_1': p1, 'prob_bookie_X': pX, 'prob_bookie_2': p2,
        'AvgCH': cota1, 'AvgCD': cotaX, 'AvgCA': cota2
    }])
    
    probs_sol = model_sol.predict_proba(input_data)[0]
    prob_p25 = model_p25.predict_proba(input_data)[0][1]
    
    print("\n" + "="*45)
    print(f" PREDICȚIE AI: {echipa_h} vs {echipa_a}")
    print("="*45)
    print(f" Șansă 1 ({echipa_h}): {probs_sol[0]*100:.1f}%")
    print(f" Șansă X (Egal): {probs_sol[1]*100:.1f}%")
    print(f" Șansă 2 ({echipa_a}): {probs_sol[2]*100:.1f}%")
    print(f" Șansă Peste 2.5 Goluri: {prob_p25*100:.1f}%")
    print("="*45 + "\n")

# === ADAUGĂ MECIURILE TALE AICI ===
prezice_meci("FC Botosani", "FC Rapid Bucuresti", 2.10, 3.20, 3.50)
prezice_meci("Dinamo Bucuresti", "Univ.Craiova", 2.30, 3.10, 3.20)
