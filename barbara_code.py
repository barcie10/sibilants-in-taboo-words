# Background:
'''
What I did: 
Formulated a directional, 
falsifiable hypothesis grounded in prior sound-symbolism research.

Hypothesis: English taboo words contain a significantly higher density of sibilant phonemes 
(/s/, /z/, /ʃ/, /ʒ/, /tʃ/, /dʒ/) than neutral vocabulary matched for word length and frequency.

Theoretical motivation: Aryani et al. (2018) demonstrated that hissing sibilants raise 
a word's spectral center of gravity and are rated as more arousing and negatively valenced. 
They also noted that George Carlin's "seven words you can't say on television" 
all contain voiceless stops or sibilants. My study tests whether this acoustic-perceptual 
finding translates into a lexical-statistical pattern across the broader English taboo vocabulary.

'''


# Sibilant density in English taboo words — full pipeline

import matplotlib.pyplot as plt
from scipy.stats import shapiro, levene, ttest_rel, wilcoxon, mannwhitneyu
import numpy as np
import nltk
import pandas as pd
from nltk.corpus import cmudict

nltk.download('cmudict', quiet=True)
d = cmudict.dict()

# 1. Taboo word list

''' 
I assembled a curated list of 29 English taboo words covering sexual/anatomical terms,
general profanity, and identity-based slurs. 
The list was based on words consistently identified as taboo in major norming studies, 
particularly Sulpizio et al. (2024) and Janschewitz (2008). 
I included only single-word base forms and excluded compounds 
and inflections to avoid double-counting the same root.
I also excluded merely negative words like death or moron that are emotionally 
heavy but not actually censored in public discourse. This produced a small 
but well-justified experimental group of clearly taboo items.

'''

taboo_words = [
    "cunt", "twat", "pussy", "dick", "cock", "prick", "dickhead",
    "tits", "balls", "cum", "shag",
    "shit", "fuck", "fucker", "bastard", "ass", "asshole",
    "crap", "piss", "bollocks", "bugger", "wanker", "douche",
    "whore", "slut",
    "faggot", "dyke", "nigger", "coon"
]


# Manual transcriptions for words not in CMUdict

'''
I converted each word from its spelling into its phonemic representation 
using the CMU Pronouncing Dictionary, accessed through Python's nltk library. 
CMUdict provides ARPAbet transcriptions for around 134,000 English words 
and avoids the unreliability of English spelling — words like sure, rose, 
and measure contain sibilants that simple letter-counting would miss. 
Of my 29 taboo words, 24 (83%) were found directly in CMUdict. 
The remaining five — twat, dickhead, tits, bollocks, wanker — were transcribed 
manually using standard reference pronunciations and converted to ARPAbet.
I documented these manual additions transparently so the methodology remains 
reproducible.

'''

manual_additions = {
    "twat":     ["T", "W", "AA1", "T"],
    "dickhead": ["D", "IH1", "K", "HH", "EH2", "D"],
    "tits":     ["T", "IH1", "T", "S"],
    "bollocks": ["B", "AA1", "L", "AH0", "K", "S"],
    "wanker":   ["W", "AE1", "NG", "K", "ER0"],
}


# 2. Phoneme lookup and sibilant analysis

'''
I defined sibilants as the six-member ARPAbet set: 
S, Z, SH, ZH, CH, JH — corresponding to IPA /s/, /z/, /ʃ/, /ʒ/, /tʃ/, /dʒ/. 
For each word, I counted the number of sibilant phonemes in its transcription 
and divided by total phoneme length to compute sibilant density, 
a normalized score between 0 and 1. This normalization is important
because it controls for word length, so a short word with one sibilant 
and a long word with three sibilants can be compared on equal footing. 
I stripped vowel stress markers (the digits in AH1, EH0) before counting,
since these are positional annotations and not part of the sibilant set. 
The result was one sibilant density score per taboo word, 
ready for comparison against neutral words.

'''

SIBILANTS = {"S", "Z", "SH", "ZH", "CH", "JH"}


def get_phones(word):
    """Return the first pronunciation for a word from CMUdict or manual additions."""
    word = word.lower()
    if word in d:
        return d[word][0]
    elif word in manual_additions:
        return manual_additions[word]
    else:
        return None


def analyze(word):
    """Compute phoneme count, sibilant count, and sibilant density for a word."""
    phones = get_phones(word)
    if phones is None:
        return None
    clean = [p.rstrip("012") for p in phones]
    sib_count = sum(1 for p in clean if p in SIBILANTS)
    return {
        "word": word,
        "phones": " ".join(clean),
        "n_phonemes": len(clean),
        "n_sibilants": sib_count,
        "sibilant_density": sib_count / len(clean),
    }


# 3. Build the taboo group dataframe

'''
I built a pool of neutral candidate words using two peer-reviewed psycholinguistic databases.
 Warriner et al. (2013) provided valence and arousal ratings for ~14,000 English words, 
 which I used to define "neutral" operationally as valence between 4.5 and 5.5 and arousal below 4.5. 
 SUBTLEX-US (Brysbaert & New, 2009) provided log-frequency counts based on film and TV subtitles for ~74,000 words. 
 I then kept only words present in all three databases — Warriner, SUBTLEX, and CMUdict — so that every 
 candidate had affective ratings, frequency, and a known pronunciation. 
 This produced a final pool of 3,375 neutral candidate words drawn from defensible,
 published sources rather than chosen by intuition.

'''

taboo_df = pd.DataFrame([analyze(w) for w in taboo_words])
print("=== Taboo group ===")
print(taboo_df.to_string(index=False))
print(f"\nMean sibilant density: {taboo_df['sibilant_density'].mean():.3f}")
print(f"SD: {taboo_df['sibilant_density'].std():.3f}\n")


# 4. NEUTRAL WORDS - Load Warriner and SUBTLEX

'''
For each taboo word, I selected one neutral word from the pool with the closest match 
on phoneme length and log-frequency, using a greedy nearest-neighbor algorithm based 
on standardized Euclidean distance. Each neutral word was used only once, so no word 
served as control for more than one taboo item. This matching procedure isolates sibilant 
density as the only meaningful difference between groups, eliminating length and frequency as confounds. 
The matches turned out exceptionally clean: mean phoneme length was 3.86 in both groups, 
and mean log-frequency was 2.83 versus 2.82. This level of precision means that any observed 
sibilant difference cannot be attributed to either of the control variables.

'''

warriner = pd.read_csv("Ratings_Warriner_et_al.csv", index_col=0)
warriner["Word"] = warriner["Word"].str.lower()

subtlex = pd.read_excel("SUBTLEXusExcel2007.xlsx")
subtlex["Word"] = subtlex["Word"].str.lower()
subtlex_lookup = dict(zip(subtlex["Word"], subtlex["Lg10WF"]))


# 5. Build the neutral candidate pool

''' 





'''

neutral_warriner = warriner[
    (warriner["V.Mean.Sum"].between(4.5, 5.5)) &
    (warriner["A.Mean.Sum"] < 4.5)
].copy()
print(f"After Warriner filter: {len(neutral_warriner)} words")

candidates = []
for w in neutral_warriner["Word"]:
    if w in subtlex_lookup and w in d:
        features = analyze(w)
        if features is not None:
            features["log_freq"] = subtlex_lookup[w]
            candidates.append(features)

neutral_pool_df = pd.DataFrame(candidates)
neutral_pool_df = neutral_pool_df[~neutral_pool_df["word"].isin(taboo_words)]

print(f"Final neutral candidate pool: {len(neutral_pool_df)} words")
print("\nFirst 15 neutral candidates:")
print(neutral_pool_df.head(15).to_string(index=False))
print("\nSummary statistics for the pool:")
print(neutral_pool_df[["n_phonemes", "log_freq",
      "sibilant_density"]].describe())


# 6. Match each taboo word to a neutral word
'''




'''

# Compute log-frequency for taboo words too
taboo_df["log_freq"] = taboo_df["word"].map(subtlex_lookup)

# Report any taboo words missing from SUBTLEX
missing = taboo_df[taboo_df["log_freq"].isna()]
if len(missing) > 0:
    print(f"\nNote: {len(missing)} taboo word(s) missing from SUBTLEX:")
    print(missing["word"].tolist())

# Greedy nearest-neighbor matching on (n_phonemes, log_freq)
matches = []
used = set()

for _, taboo_row in taboo_df.iterrows():
    if pd.isna(taboo_row["log_freq"]):
        continue  # skip taboo words with no frequency data
    candidates = neutral_pool_df[~neutral_pool_df["word"].isin(used)].copy()
    # Standardized distance: length difference weighted equally with freq difference
    candidates["dist"] = np.sqrt(
        ((candidates["n_phonemes"] - taboo_row["n_phonemes"]) / 2) ** 2 +
        ((candidates["log_freq"] - taboo_row["log_freq"]) / 1) ** 2
    )
    best = candidates.nsmallest(1, "dist").iloc[0]
    matches.append({
        "taboo_word": taboo_row["word"],
        "neutral_word": best["word"],
        "taboo_phones": taboo_row["phones"],
        "neutral_phones": best["phones"],
        "taboo_len": taboo_row["n_phonemes"],
        "neutral_len": best["n_phonemes"],
        "taboo_freq": taboo_row["log_freq"],
        "neutral_freq": best["log_freq"],
        "taboo_sib_density": taboo_row["sibilant_density"],
        "neutral_sib_density": best["sibilant_density"],
    })
    used.add(best["word"])

match_df = pd.DataFrame(matches)
print("\n=== Matched pairs ===")
print(match_df[["taboo_word", "neutral_word", "taboo_len", "neutral_len",
                "taboo_freq", "neutral_freq",
                "taboo_sib_density", "neutral_sib_density"]].to_string(index=False))


# 7. Compare the two groups

'''






'''

print("\n=== Group comparison ===")
print(
    f"Taboo  - mean sibilant density: {match_df['taboo_sib_density'].mean():.3f}")
print(
    f"Neutral - mean sibilant density: {match_df['neutral_sib_density'].mean():.3f}")
print(f"\nTaboo  - mean phoneme length:   {match_df['taboo_len'].mean():.2f}")
print(f"Neutral - mean phoneme length:   {match_df['neutral_len'].mean():.2f}")
print(f"\nTaboo  - mean log-frequency:    {match_df['taboo_freq'].mean():.2f}")
print(
    f"Neutral - mean log-frequency:    {match_df['neutral_freq'].mean():.2f}")


# 8. Statistical testing

'''





'''

taboo_scores = match_df["taboo_sib_density"].values
neutral_scores = match_df["neutral_sib_density"].values

# Check normality assumption
print("=== Assumption checks ===")
sw_taboo = shapiro(taboo_scores)
sw_neutral = shapiro(neutral_scores)
print(
    f"Shapiro-Wilk on taboo:   W = {sw_taboo.statistic:.3f}, p = {sw_taboo.pvalue:.4f}")
print(
    f"Shapiro-Wilk on neutral: W = {sw_neutral.statistic:.3f}, p = {sw_neutral.pvalue:.4f}")
# If p < .05, data deviates from normal — non-parametric test preferred


# Paired t-test (parametric)

'''
Because each taboo word is paired with a specific neutral word,
# we use the *paired* version, which is more powerful than the independent test.

'''
print("\n=== Paired t-test ===")
t_stat, t_p = ttest_rel(taboo_scores, neutral_scores)

# Convert two-tailed p to one-tailed since hypothesis is directional
t_p_one_tailed = t_p / 2 if t_stat > 0 else 1 - t_p / 2
print(f"t = {t_stat:.3f}, two-tailed p = {t_p:.4f}, one-tailed p = {t_p_one_tailed:.4f}")

# Wilcoxon signed-rank test (non-parametric)
print("\n=== Wilcoxon signed-rank test (non-parametric, paired) ===")
w_stat, w_p = wilcoxon(taboo_scores, neutral_scores, alternative="greater")
print(f"W = {w_stat:.3f}, one-tailed p = {w_p:.4f}")

# Effect size (Cohen's d for paired samples)
diff = taboo_scores - neutral_scores
cohen_d_paired = diff.mean() / diff.std(ddof=1)
print(f"\nCohen's d (paired): {cohen_d_paired:.3f}")

# Summary
print("\n=== Summary ===")
print(
    f"Mean taboo sibilant density:   {taboo_scores.mean():.3f} (SD = {taboo_scores.std(ddof=1):.3f})")
print(
    f"Mean neutral sibilant density: {neutral_scores.mean():.3f} (SD = {neutral_scores.std(ddof=1):.3f})")
print(f"Mean difference:               {diff.mean():.3f}")
print(f"N pairs:                       {len(diff)}")


# 9. Visualization


fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# Left panel: boxplot + individual points
data_to_plot = [taboo_scores, neutral_scores]
bp = axes[0].boxplot(data_to_plot, labels=["Taboo", "Neutral"],
                     widths=0.5, patch_artist=True, showmeans=True,
                     boxprops=dict(facecolor="lightgray", alpha=0.6),
                     medianprops=dict(color="black", linewidth=2),
                     meanprops=dict(marker="D", markerfacecolor="red",
                                    markeredgecolor="red", markersize=8))
# Overlay individual points with jitter
for i, scores in enumerate(data_to_plot, start=1):
    x_jitter = np.random.normal(i, 0.05, size=len(scores))
    axes[0].scatter(x_jitter, scores, alpha=0.6, color="steelblue", s=30)

axes[0].set_ylabel("Sibilant density")
axes[0].set_title("Sibilant density by group\n(red diamond = mean)")
axes[0].grid(axis="y", alpha=0.3)

# Right panel: histogram overlay
bins = np.linspace(0, 0.7, 12)
axes[1].hist(taboo_scores, bins=bins, alpha=0.6, label=f"Taboo (n={len(taboo_scores)})",
             color="crimson", edgecolor="black")
axes[1].hist(neutral_scores, bins=bins, alpha=0.6, label=f"Neutral (n={len(neutral_scores)})",
             color="steelblue", edgecolor="black")
axes[1].axvline(taboo_scores.mean(), color="crimson",
                linestyle="--", linewidth=2)
axes[1].axvline(neutral_scores.mean(), color="steelblue",
                linestyle="--", linewidth=2)
axes[1].set_xlabel("Sibilant density")
axes[1].set_ylabel("Number of words")
axes[1].set_title(
    "Distribution of sibilant density\n(dashed lines = group means)")
axes[1].legend()
axes[1].grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("sibilant_density_results.png", dpi=200, bbox_inches="tight")
plt.show()
print("\nFigure saved as sibilant_density_results.png")
