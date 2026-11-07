"""Everyday words -> medical terms, used to widen a search query.

People write "pass urine", "loose motions" or "sugar", while MedlinePlus says "urinary",
"diarrhea" or "diabetes". Each entry maps a list of everyday phrases to extra search words.
The original question is never changed: the extra words are only added to the search, and only
if they exist in the knowledge base. A matched everyday phrase counts as covered when a topic
contains any of its medical words. Add entries for the wording your users really use.
"""
import re

# everyday phrases -> medical words to add
LAY_TERMS = [
    (["pass urine", "passing urine", "burning urination", "burning when i pee", "peeing", "pee",
      "urinate", "urinating", "burning feeling when i pass"], "urinary urination bladder"),
    (["pale", "paleness", "looking pale", "look pale"], "anemia"),
    (["tired", "tiredness", "exhausted", "no energy",
  "always sleepy", "weak and tired", "feeling tired",
  "unusually tired", "feeling unusually tired"], "fatigue"),
    (["sugar", "sugar level", "sugar problem", "high sugar", "low sugar"], "diabetes glucose"),
    (["bp", "high bp", "low bp"], "blood pressure hypertension"),
    (["heartburn", "acidity", "acid reflux", "gastric", "gastric problem", "burning in chest after eating"],
     "gerd reflux heartburn"),
    (["loose motion", "loose motions", "loose stools", "watery stool", "watery stools", "runny stomach"], "diarrhea"),
    (["constipated", "hard stools", "hard stool", "cant pass stool", "cannot pass stool"], "constipation"),
    (["tummy", "belly", "stomach ache", "stomach pain", "stomach cramps", "tummy ache"], "stomach abdominal pain"),
    (["puking", "throwing up", "threw up"], "nausea vomiting"),
    (["fits", "convulsions", "convulsion"], "seizures epilepsy"),
    (["piles"], "hemorrhoids"),
    (["wheezing", "wheeze", "whistling sound when breathing"], "asthma wheezing"),
    (["lost my sense of smell", "loss of smell", "cant smell", "cannot smell"], "covid smell"),
    (["itchy skin", "skin rash", "red patches", "dry itchy"], "eczema dermatitis rash"),
    (["cant sleep", "cannot sleep", "trouble sleeping", "not sleeping", "sleepless", "sleeplessness"],
     "insomnia sleep"),
    (["backache", "back ache", "lower back pain", "pain in lower back", "pain in my back"], "back pain"),
    (["joint pain", "swollen joints", "stiff joints", "painful joints", "swollen painful joints"],
     "arthritis joint"),
    # "runny nose" and "sneezing" are too general to mean allergy (colds and flu cause them too),
    # so they are not mapped to allergy; only wording that clearly points to it is.
    (["itchy eyes", "hay fever", "pollen allergy"], "allergies allergy"),
    (["runny nose", "stuffy nose", "blocked nose", "sneezing"], "cold"),
    (["pimples", "pimple", "zits"], "acne"),
    (["hair fall", "losing hair", "hair falling", "hair is falling"], "hair loss"),
    (["dizzy", "dizziness", "giddiness", "giddy", "lightheaded"], "dizziness vertigo"),
    (["earache", "ear pain", "pain in my ear"], "ear infection"),
    (["toothache", "tooth pain"], "tooth dental"),
    (["red eye", "red eyes", "pink eye"], "conjunctivitis eye"),
    (["racing heart", "heart is racing", "palpitations", "heart pounding"], "palpitations arrhythmia heart"),
    (["feeling sad", "feeling low", "sad all the time", "hopeless"], "depression"),
    (["panic attack", "panic attacks", "worried all the time", "constant worry"], "anxiety panic"),
    (["yellow eyes", "yellow skin", "yellowish"], "jaundice liver"),
    (["periods", "irregular periods", "period pain", "menstrual cramps"], "menstruation menstrual"),
    (["pain when i swallow", "painful swallowing", "difficulty swallowing"], "throat swallowing"),
]

_COMPILED = [
    (re.compile(r"\b(?:" + "|".join(re.escape(p) for p in phrases) + r")\b"), words)
    for phrases, words in LAY_TERMS
]


def expansion_groups(query: str) -> list:
    """For each everyday phrase found in the query: (matched phrases, extra medical words)."""
    text = (query or "").lower().replace("\u2019", "'").replace("'", "")
    groups = []
    for pattern, words in _COMPILED:
        found = [m.group(0) for m in pattern.finditer(text)]
        if found:
            groups.append((list(dict.fromkeys(found)), words))
    return groups