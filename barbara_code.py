
'''
Sibilants in English Taboo Words: A Quantitative Analysis

Compares sibilant phoneme density between a curated list of 29 English
taboo words and a matched set of 29 affectively neutral words drawn from
Warriner et al. (2013) and SUBTLEX-US (Brysbaert & New, 2009).

Pipeline:
  1. Taboo word list compilation
  2. Phoneme lookup and sibilant analysis (via CMUdict)
  3. Build the taboo group dataframe
  4. Load Warriner and SUBTLEX databases
  5. Build the neutral candidate pool
  6. Match each taboo word to a neutral word
  7. Compare the two groups
  8. Statistical testing (Wilcoxon, paired t-test, Cohen's d)
  9. Visualization (boxplots and histograms)

Author: Barbara Cieślak 

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
I built a DataFrame containing every taboo word together with its phonemic
transcription, total phoneme count, sibilant count, and computed sibilant
density. This consolidates the experimental group into a structured, analyzable
format and immediately gives me descriptive statistics: a mean sibilant
density of 0.116 (SD = 0.152) across the 29 taboo words. This is the baseline
number that will be compared against the matched neutral group in subsequent
steps. At this stage, no conclusions can be drawn yet, since the comparison
group has not been built or matched.

'''


taboo_df = pd.DataFrame([analyze(w) for w in taboo_words])
print("=== Taboo group ===")
print(taboo_df.to_string(index=False))
print(f"\nMean sibilant density: {taboo_df['sibilant_density'].mean():.3f}")
print(f"SD: {taboo_df['sibilant_density'].std():.3f}\n")


# 4. NEUTRAL WORDS - Load Warriner and SUBTLEX

'''
I loaded the two external psycholinguistic resources that will be used to
build the neutral baseline. Warriner et al. (2013) provides affective ratings
(valence and arousal) for ~14,000 English lemmas; this lets me define
"neutral" operationally rather than by intuition. SUBTLEX-US (Brysbaert & New,
2009) provides word-frequency counts derived from film and TV subtitles for
~74,000 English words; this lets me match neutral words to taboo words on
frequency. I lowercased all words in both databases for consistent matching,
and built a fast dictionary lookup mapping each SUBTLEX word to its log10
word frequency (Lg10WF) for efficient access during the matching stage.

'''

warriner = pd.read_csv("Ratings_Warriner_et_al.csv", index_col=0)
warriner["Word"] = warriner["Word"].str.lower()

subtlex = pd.read_excel("SUBTLEXusExcel2007.xlsx")
subtlex["Word"] = subtlex["Word"].str.lower()
subtlex_lookup = dict(zip(subtlex["Word"], subtlex["Lg10WF"]))


# 5. Build the neutral candidate pool

'''
I constructed a pool of affectively neutral candidate words by intersecting
three sources. From Warriner I kept only words with valence between 4.5 and
5.5 (the true middle of the 1-9 scale) and arousal below 4.5 — this filter
excludes both positively and negatively valenced words, as well as emotionally
arousing ones, leaving 3,522 affectively flat items. I then kept only words
that are also present in SUBTLEX (so I have frequency data) and in CMUdict
(so I have a phonemic transcription), producing a final pool of 3,375
candidates. I also excluded any of my taboo words that might have slipped
through the affective filter, to prevent contamination of the control group.
This pool is built entirely from peer-reviewed published databases rather
than from intuition, which makes the neutral category defensible.

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
I implemented a greedy nearest-neighbor matching algorithm that pairs each
taboo word with the neutral candidate most similar to it on (phoneme length,
log-frequency). For each taboo word I compute a standardized Euclidean
distance to every available neutral candidate and select the closest one;
the chosen neutral word is then removed from the pool so that no neutral
word is used as a control for more than one taboo item. This matching is
the methodologically critical step of the study: without it, any sibilant
difference between groups could be a confound of length (shorter words tend
to have higher proportional sibilant density) or frequency (rarer words have
different phonological profiles). I also added a check for taboo words
missing from SUBTLEX, which would be skipped — though in practice all 29
taboo words had frequency data available.

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
I computed descriptive statistics for both groups to verify that the matching
procedure worked and to get a first look at the result. The key check is that
the two groups are now equivalent on the control variables: mean phoneme
length is 3.86 in both groups (identical to two decimal places), and mean
log-frequency is 2.83 vs 2.82 (differing by 0.01). This means that any
observed difference in sibilant density cannot be attributed to length or
frequency confounds. The descriptive sibilant-density means are 0.116
(taboo) versus 0.076 (neutral) — a 52% relative increase — though formal
significance testing still needs to be done in the next step before drawing
conclusions.

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
Because each taboo word is paired with a specific matched neutral word, I
used paired statistical tests rather than independent-samples tests — these
are both more appropriate and statistically more powerful given the design.
The Shapiro-Wilk test indicated strong non-normality in both groups
(p < 0.0001), which was expected given that sibilant density is bounded
and many words have zero sibilants. The non-parametric Wilcoxon signed-rank
test was therefore my primary analysis (W = 51.5, one-tailed p = 0.16),
with the paired t-test reported for reference (t(28) = 1.17, one-tailed
p = 0.13). I used a one-tailed test because my hypothesis was directional,
predicting specifically that taboo words would show higher sibilant density.
I also computed Cohen's d for paired samples to report effect size
(d = 0.22, a small effect), since reporting only p-values can be misleading
in small samples. The result: the trend is in the predicted direction but
does not reach statistical significance at n = 29 pairs.

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


# .............................................................................................................................................................................

# 9. Visualization

'''
I produced a two-panel figure summarizing the comparison. The left panel
shows boxplots of sibilant density for each group with individual data
points overlaid via jitter and group means marked as red diamonds; this
combines distributional summary with the raw data so the reader can see
both the central tendency and the spread. The right panel shows overlapping
histograms of sibilant density with dashed vertical lines marking each
group's mean; this lets the reader see directly that the two distributions
overlap heavily, which is the visual explanation for why the statistical
test did not reach significance. The figure is saved to disk for embedding
in the final research paper.

'''

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
