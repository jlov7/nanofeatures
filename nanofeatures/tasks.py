"""Task suite for the breadth sweep. The single-task M2-M6 result (capitals, layer 7)
is an anecdote until the causal-vs-cheap-baseline finding is shown to be task-general.
Each task is a (pairs, template) spec consumed by circuit.aligned_pairs; pairs are
filtered at runtime to those whose SUBJECT and ANSWER are both single Gemma tokens, so
every prompt has identical length (position-aligned patching, the nanocircuits PAD
lesson). Pairs are generous on purpose -- the single-token filter and the behavior gate
in run_suite decide which survive; we do NOT hand-pick to flatter the method.

Tasks span deliberately different mechanisms:
  capitals          factual recall (geography)
  antonyms          lexical-semantic opposition
  country_language  factual recall (different relation, tests recall-generality)
  past_tense        morphology (irregular verbs)
  comparative       morphology (adjective inflection)
  plural            morphology (irregular nouns)
  successor         sequence / counting
"""

from __future__ import annotations

# --- factual recall: geography (the original M2-M6 task) -----------------------------
CAPITALS = [
    ("France", "Paris"),
    ("Japan", "Tokyo"),
    ("Italy", "Rome"),
    ("Germany", "Berlin"),
    ("Spain", "Madrid"),
    ("Russia", "Moscow"),
    ("China", "Beijing"),
    ("Egypt", "Cairo"),
    ("Greece", "Athens"),
    ("Cuba", "Havana"),
    ("Peru", "Lima"),
    ("Iran", "Tehran"),
    ("Austria", "Vienna"),
    ("Poland", "Warsaw"),
    ("Norway", "Oslo"),
    ("Sweden", "Stockholm"),
    ("Portugal", "Lisbon"),
    ("Ireland", "Dublin"),
    ("Kenya", "Nairobi"),
    ("Chile", "Santiago"),
]
CAPITALS_TEMPLATE = "The capital of {subject} is the city of"

# --- lexical-semantic: antonyms ------------------------------------------------------
ANTONYMS = [
    ("hot", "cold"),
    ("big", "small"),
    ("up", "down"),
    ("fast", "slow"),
    ("light", "dark"),
    ("hard", "soft"),
    ("high", "low"),
    ("rich", "poor"),
    ("old", "new"),
    ("good", "bad"),
    ("true", "false"),
    ("day", "night"),
    ("love", "hate"),
    ("buy", "sell"),
    ("push", "pull"),
    ("left", "right"),
    ("north", "south"),
    ("wet", "dry"),
    ("full", "empty"),
    ("happy", "sad"),
    ("strong", "weak"),
    ("open", "shut"),
    ("win", "lose"),
    ("black", "white"),
]
ANTONYM_TEMPLATE = "The opposite of {subject} is"

# --- factual recall: dominant language (distinct single-token answers) ----------------
COUNTRY_LANGUAGE = [
    ("France", "French"),
    ("Japan", "Japanese"),
    ("Germany", "German"),
    ("Spain", "Spanish"),
    ("Russia", "Russian"),
    ("China", "Chinese"),
    ("Greece", "Greek"),
    ("Poland", "Polish"),
    ("Sweden", "Swedish"),
    ("Italy", "Italian"),
    ("Turkey", "Turkish"),
    ("Finland", "Finnish"),
    ("Portugal", "Portuguese"),
    ("Norway", "Norwegian"),
    ("Vietnam", "Vietnamese"),
    ("Korea", "Korean"),
    ("Thailand", "Thai"),
    ("Denmark", "Danish"),
    ("Hungary", "Hungarian"),
    ("Iran", "Persian"),
]
COUNTRY_LANGUAGE_TEMPLATE = "The people of {subject} mainly speak the language called"

# --- morphology: irregular past tense ------------------------------------------------
PAST_TENSE = [
    ("go", "went"),
    ("eat", "ate"),
    ("see", "saw"),
    ("run", "ran"),
    ("take", "took"),
    ("give", "gave"),
    ("come", "came"),
    ("know", "knew"),
    ("make", "made"),
    ("find", "found"),
    ("think", "thought"),
    ("buy", "bought"),
    ("bring", "brought"),
    ("catch", "caught"),
    ("teach", "taught"),
    ("drink", "drank"),
    ("swim", "swam"),
    ("write", "wrote"),
    ("drive", "drove"),
    ("ride", "rode"),
    ("fall", "fell"),
    ("feel", "felt"),
    ("hold", "held"),
    ("keep", "kept"),
    ("sleep", "slept"),
    ("speak", "spoke"),
    ("stand", "stood"),
    ("win", "won"),
    ("sing", "sang"),
    ("grow", "grew"),
]
PAST_TENSE_TEMPLATE = "The past tense of {subject} is"

# --- morphology: comparative adjectives ----------------------------------------------
COMPARATIVE = [
    ("big", "bigger"),
    ("small", "smaller"),
    ("fast", "faster"),
    ("slow", "slower"),
    ("tall", "taller"),
    ("short", "shorter"),
    ("old", "older"),
    ("young", "younger"),
    ("strong", "stronger"),
    ("weak", "weaker"),
    ("hard", "harder"),
    ("soft", "softer"),
    ("warm", "warmer"),
    ("cold", "colder"),
    ("high", "higher"),
    ("low", "lower"),
    ("deep", "deeper"),
    ("long", "longer"),
    ("rich", "richer"),
    ("poor", "poorer"),
    ("clean", "cleaner"),
    ("dark", "darker"),
    ("light", "lighter"),
    ("smart", "smarter"),
]
COMPARATIVE_TEMPLATE = "The comparative form of {subject} is"

# --- morphology: irregular plurals ---------------------------------------------------
PLURAL = [
    ("man", "men"),
    ("woman", "women"),
    ("child", "children"),
    ("foot", "feet"),
    ("tooth", "teeth"),
    ("mouse", "mice"),
    ("goose", "geese"),
    ("person", "people"),
    ("leaf", "leaves"),
    ("knife", "knives"),
    ("life", "lives"),
    ("wife", "wives"),
    ("loaf", "loaves"),
    ("calf", "calves"),
    ("half", "halves"),
    ("wolf", "wolves"),
]
PLURAL_TEMPLATE = "The plural of {subject} is"

# --- sequence: number successor ------------------------------------------------------
SUCCESSOR = [
    ("one", "two"),
    ("two", "three"),
    ("three", "four"),
    ("four", "five"),
    ("five", "six"),
    ("six", "seven"),
    ("seven", "eight"),
    ("eight", "nine"),
    ("nine", "ten"),
    ("ten", "eleven"),
    ("eleven", "twelve"),
]
SUCCESSOR_TEMPLATE = "The number that comes right after {subject} is"

# --- MULTI-TOKEN-SUBJECT recall: the contrastive signal spans 2 token positions, not 1.
# Same relation/template as `capitals`, but every subject is a 2-token compound, so clean
# vs corrupt differ at TWO positions -> an intermediate point on the distributedness axis
# between single-token tasks (1 position) and IOI (3 positions). Single-token distinct
# capitals keep the metric well-defined and position-aligned.
CAPITALS_2TOK = [
    ("Saudi Arabia", "Riyadh"),
    ("South Korea", "Seoul"),
    ("North Korea", "Pyongyang"),
    ("Sri Lanka", "Colombo"),
    ("New Zealand", "Wellington"),
    ("South Africa", "Pretoria"),
    ("Czech Republic", "Prague"),
    ("United Kingdom", "London"),
    ("United States", "Washington"),
]
CAPITALS_2TOK_TEMPLATE = "The capital of {subject} is the city of"

# more 2-token-subject relations, to populate the mid-distributedness range with several
# points (so the gap-vs-distributedness trend does not rest on one or two leverage tasks).
CITY_COUNTRY_2TOK = [
    ("Hong Kong", "China"),
    ("Tel Aviv", "Israel"),
    ("New Delhi", "India"),
    ("Buenos Aires", "Argentina"),
    ("Abu Dhabi", "Emirates"),
    ("Las Vegas", "America"),
    ("Cape Town", "Africa"),
    ("Kuala Lumpur", "Malaysia"),
]
CITY_COUNTRY_2TOK_TEMPLATE = "The city of {subject} is located in the country of"

PERSON_COUNTRY_2TOK = [
    ("Albert Einstein", "Germany"),
    ("Isaac Newton", "England"),
    ("Marie Curie", "Poland"),
    ("Pablo Picasso", "Spain"),
    ("Napoleon Bonaparte", "France"),
    ("Vincent van", "Netherlands"),
    ("Leonardo da", "Italy"),
]
PERSON_COUNTRY_2TOK_TEMPLATE = "{subject} was born in the country of"

# name -> (pairs, template). run_suite filters + behavior-gates each.
TASK_SUITE = {
    "capitals": (CAPITALS, CAPITALS_TEMPLATE),
    "antonyms": (ANTONYMS, ANTONYM_TEMPLATE),
    "country_language": (COUNTRY_LANGUAGE, COUNTRY_LANGUAGE_TEMPLATE),
    "past_tense": (PAST_TENSE, PAST_TENSE_TEMPLATE),
    "comparative": (COMPARATIVE, COMPARATIVE_TEMPLATE),
    "plural": (PLURAL, PLURAL_TEMPLATE),
    "successor": (SUCCESSOR, SUCCESSOR_TEMPLATE),
}

# Back-compat: existing M5/M6 imports expect these two names.
ANTONYM_TASK = (ANTONYMS, ANTONYM_TEMPLATE)
PAIRS = CAPITALS
TEMPLATE = CAPITALS_TEMPLATE
