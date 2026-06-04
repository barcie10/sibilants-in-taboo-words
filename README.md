# Sibilants in English Taboo Words

A quantitative study testing whether English taboo words contain a higher 
density of sibilant phonemes than neutral words matched on length and frequency.

**Author:** Barbara Cieślak  
**Course:** Python Final Project, Adam Mickiewicz University   
**Date:** 2026

## Research question

To what extent do sibilant phonemes (/s/, /z/, /ʃ/, /ʒ/, /tʃ/, /dʒ/) occur 
more frequently in English taboo words compared to a frequency- and 
length-matched neutral lexical baseline?

## Hypothesis

English taboo words contain a higher density of sibilant phonemes than 
neutral vocabulary matched for word frequency and length. Motivated by 
Aryani et al. (2018), who showed that hissing sibilants raise a word's 
acoustic intensity and are rated as more arousing and negatively valenced.

## Method

- **Taboo group (n = 29):** curated list of single-word English taboo items
- **Neutral group (n = 29):** matched 1:1 on phoneme length and log-frequency
- **Phonemic transcription:** CMU Pronouncing Dictionary (`nltk.corpus.cmudict`)
- **Affective filtering:** Warriner et al. (2013) — valence 4.5–5.5, arousal < 4.5
- **Frequency matching:** SUBTLEX-US (Brysbaert & New, 2009) on log10 word frequency
- **Statistics:** paired Wilcoxon signed-rank (primary, non-parametric, one-tailed), 
  paired t-test, and Cohen's d for effect size

## Results

| Group   | Mean sibilant density | SD    |
|---------|----------------------:|------:|
| Taboo   | 0.116                 | 0.152 |
| Neutral | 0.076                 | 0.128 |

- Mean difference: 0.040 (taboo words 52% denser in sibilants)
- Wilcoxon signed-rank: W = 51.5, one-tailed p = 0.16
- Cohen's d (paired) = 0.22 (small effect)

The trend is in the predicted direction but does not reach statistical 
significance with n = 29 pairs.

## Repository contents

- `barbara_code.py` — full analysis pipeline
- `barbara_data/` — source datasets and exported analysis results
- `requirements.txt` — Python dependencies

## How to reproduce

1. Install dependencies: `pip install -r requirements.txt`
2. Download the Warriner and SUBTLEX datasets (see `barbara_data/README.md`)
3. Place them next to `barbara_code.py`
4. Run `python barbara_code.py`

## Acknowledgments

This project was developed with assistance from Claude (Anthropic) for 
Python syntax scaffolding, debugging, and methodological discussion. 
All study design decisions, the taboo word list, data interpretation, 
and write-up are my own.

## References

- Aryani, A., Conrad, M., Schmidtke, D., & Jacobs, A. (2018). Why 'piss' is 
  ruder than 'pee'? The role of sound in affective meaning making. 
  *PLOS ONE*, 13(6), e0198430.
- Brysbaert, M., & New, B. (2009). Moving beyond Kučera and Francis: A 
  critical evaluation of current word frequency norms. 
  *Behavior Research Methods*, 41(4), 977–990.
- Reilly, J., et al. (2020). Building the perfect curse word. 
  *Psychonomic Bulletin & Review*, 27(6), 1339–1348.
- Sulpizio, S., et al. (2024). Taboo language across the globe: A multi-lab 
  study. *Behavior Research Methods*, 56(4), 3794–3813.
- Svantesson, J.-O. (2017). Sound symbolism: The role of word sound in meaning. 
  *WIREs Cognitive Science*, 8(5), e1441.
- Warriner, A. B., Kuperman, V., & Brysbaert, M. (2013). Norms of valence, 
  arousal, and dominance for 13,915 English lemmas. 
  *Behavior Research Methods*, 45(4), 1191–1207.
