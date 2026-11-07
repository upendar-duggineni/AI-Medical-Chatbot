"""Safety rules in plain code. They run BEFORE search and BEFORE the LLM, so safety never depends on the model.

Three outcomes: emergency (fixed message), dose (refuse and redirect), or allow.
Emergency wording that is only meaningful about a real person (seizure, stroke signs, ...) needs personal context,
so general questions like 'What is a seizure?' are not treated as emergencies.
"""
import re

EMERGENCY_NUMBER = "112"            # India
CRISIS_LINE = "Tele-MANAS 14416"

CRISIS = re.compile(
    r"suicid|kill(ing)? (myself|himself|herself|themselves)|end my life|take my own life|want to die"
    r"|self[- ]?harm|(hurt|hurting|cut|cutting) myself|don'?t want to (live|be alive|wake up)"
    r"|better off dead|no reason to live|nothing to live for|end it all|wish i (was|were) dead"
    r"|wish i (would|could) (just )?(die|disappear)")

STRONG = re.compile(
    r"can'?t breathe?|cannot breathe?|struggling to breathe?|not breathing|stopped breathing"
    r"|gasping for air|gasping|unable to breathe"
    r"|unresponsive|not responding|not waking up|collapsed(?! lung)"
    r"|crushing chest pain|vomit(ing|ed)? blood|cough(ing)? (up )?blood"
    r"|took too many (pills|tablets)|swallowed (poison|bleach)")


CONTEXTUAL = re.compile(
    r"seizure|convuls|face (is )?droop|slurred speech|sudden (weakness|numbness)|passed out|fainted"
    r"|severe bleeding|heavy bleeding|bleeding heavily|uncontrolled bleeding"
    r"|bleeding (won'?t|will not|can'?t|cannot) stop"
    r"|throat (is )?(closing|swelling)|swollen (tongue|lips)"
    r"|tongue is swelling|tongue swelling|swelling of (the )?tongue"
    r"|anaphyla|severe allergic reaction|overdose|severe chest pain"
    r"|heart attack|cardiac arrest|chok(ing|ed)|unconscious"
    r"|snake ?bite|bitten by (a )?snake|poison(ed|ing)|drown(ing|ed)"
    r"|turning blue|blue (lips|face|skin)|worst headache"
    r"|stroke|having a stroke|signs of a stroke"
    r"|chest (pain|pressure|tightness|hurts).*(short(ness)? of breath|breathless|sweating|left arm|jaw|numb)"
    r"|chest hurts|chest pain|chest pressure"
)

# "they" is left out on purpose: it appears in many general questions.
PERSONAL = re.compile(
    r"\b(i|i'm|im|i've|ive|my|me|myself|he|she|his|her|him|someone|somebody|mom|mum|dad|mother|father|baby|"
    r"child|kid|son|daughter|wife|husband|friend|patient|brother|sister|grandmother|grandfather)\b")

_DRUGS = (r"(paracetamol|acetaminophen|ibuprofen|aspirin|crocin|dolo|combiflam|antibiotics?|amoxicillin|"
          r"azithromycin|metformin|insulin|painkillers?|medicines?|tablets?|pills?|capsules?)")

DOSE = re.compile(
    r"\b(dose|dosage|dosing|overdosing)\b|how (much|many) [^.?!]{0,40}\b(mg|ml|tablets?|pills?|capsules?|drops?|units?)\b"
    r"|\b\d+(\.\d+)?\s?(mg|ml|mcg|iu)\b|can i (take|give) [^.?!]{0,40}\b(tablets?|pills?|capsules?|mg)\b"
    r"|how (much|many) [^.?!]{0,30}\b(insulin|paracetamol|ibuprofen|crocin|aspirin|antibiotic)"
    rf"|\b(can|could|should|may) (i|we) (take|give|use|have|combine|mix)\b[^.?!]{{0,50}}\b{_DRUGS}\b"
    r"|\bis it (safe|ok|okay|fine) to (take|give|combine|mix)\b"
    rf"|\bis it (safe|ok|okay|fine) to use\b[^.?!]{{0,50}}\b{_DRUGS}\b"
    r"|\bhow (often|long|many times) (can|should|do|could) (i|we|you) (take|give)\b"
    rf"|\bhow (often|long|many times) (can|should|do|could) (i|we|you) use\b[^.?!]{{0,50}}\b{_DRUGS}\b")
    

PERSONAL_DX = re.compile(r"\b(do i have|am i having|could i have|might i have|diagnose me|is it cancer|do you think i have)\b")


def _normalize(text: str) -> str:
    """Lower-case, turn curly quotes (common on phones) into plain ones, collapse spaces."""
    text = text.lower().replace("\u2019", "'").replace("\u2018", "'").replace("`", "'")
    return re.sub(r"\s+", " ", text).strip()


# --- crisis false-positive guards (conservative: only ever REMOVE a phrase, then re-test) ---
_FIRST_PERSON = re.compile(r"\b(i|i'm|im|i've|ive|my|me|myself)\b")
_INFO_QUESTION = re.compile(r"^(what|who|define|explain|tell me about)\b")
_SELF_HARM_TOPIC = re.compile(r"self[- ]?harm\w*")
_CUT_MYSELF = re.compile(r"\b(cut|cutting) myself\b")
_BENIGN_CUT = re.compile(
    r"\b(cook(ing|ed)?|chop(ping|ped)?|slic(ing|ed)|shav(ing|ed)|kitchen|vegetables?|onions?|paper|bread|"
    r"accident(ally)?|by accident)\b")
_INTENT = re.compile(r"\b(on purpose|deliberately|again|keep|keeps|urge|urges|tonight|every (day|night|time)|whenever)\b")


def _is_crisis(q: str) -> bool:
    """CRISIS match, ignoring (a) the bare topic 'self-harm' in an informational question with no
    first-person words and (b) an accidental everyday cut with no sign of intent. Any OTHER crisis
    phrase in the same message still triggers, because the rest of the text is re-tested."""
    rest = q
    if _INFO_QUESTION.search(q) and not _FIRST_PERSON.search(q):
        rest = _SELF_HARM_TOPIC.sub(" ", rest)
    if _BENIGN_CUT.search(q) and not _INTENT.search(q):
        rest = _CUT_MYSELF.sub(" ", rest)
    return bool(CRISIS.search(rest))


# --- red-flag layer -----------------------------------------------------------------------
# Warning-sign patterns that a symptom list does not spell out. They run in code, so the urgent
# text is always shown, in the same words, and never depends on the LLM remembering to say it.
#   escalate=True : with personal context ("I have...", "my son...") -> the emergency message
#   otherwise     : an advisory the API should show at the TOP of the answer
# Only the first matching rule per `group` is used, so one pattern doesn't produce two warnings.
_HEADACHE = r"head ?ache|migraine"
_FEVER = r"fever|high temperature"

RED_FLAG_RULES = [
    {"id": "stiff_neck", "group": "meningitis", "escalate": True,
     "all": [r"stiff neck|neck (is |feels )?stiff|neck stiffness",
             rf"{_FEVER}|{_HEADACHE}|vomit|rash|confus|sensitiv\w* to (bright )?light|light sensitiv"],
     "message": "A stiff neck together with fever or headache can be a sign of meningitis, which needs "
                "urgent care. Call 112 or go to the nearest emergency department."},
    {"id": "headache_fever", "group": "meningitis", "escalate": False,
     "all": [_HEADACHE, _FEVER],
     "message": "Headache with fever is often a viral illness, but it can occasionally signal a serious "
                "infection such as meningitis. Get urgent medical care (call 112) if you also have a stiff "
                "neck, confusion or unusual drowsiness, a new rash, a seizure, or pain from bright light."},
    {"id": "head_injury", "group": "head", "escalate": False,
     "all": [_HEADACHE, r"(after|following) (a |an )?(blow|hit|knock|bump|fall|injury|accident)"
                        r"|hit (my|his|her|the) head|head injury"],
     "message": "A headache after a blow to the head needs prompt medical assessment, especially with "
                "vomiting, drowsiness, confusion or a headache that is getting worse."},
    {"id": "infant_fever", "group": "infant", "escalate": False,
     "all": [_FEVER, r"\b(baby|infant|newborn)\b|\b\d+ ?(day|week)s?[- ]?old|\b[0-3] ?months?[- ]?old"],
     "message": "A fever in a baby younger than 3 months needs same-day medical assessment. Contact a "
                "doctor or go to an emergency department now."},
    {"id": "sudden_vision_loss", "group": "vision", "escalate": True,
     "all": [r"sudden(ly)? (loss of )?(vision|sight)|sudden(ly)? (can'?t|cannot|unable to) see"
             r"|lost (my |his |her )?(vision|sight)"],
     "message": "Sudden loss of vision needs emergency care. Call 112 or go to the nearest emergency "
                "department right away."},
    {"id": "blood_stool_urine", "group": "blood", "escalate": False,
     "all": [r"blood (in|with) (my |his |her |the )?(stool|stools|poop|urine|pee)|black,? (tarry )?stools?"],
     "message": "Blood in stool or urine, or black tarry stool, should be checked by a doctor soon. Get "
                "urgent care if it is heavy or comes with dizziness, weakness or fainting."},
    {"id": "severe_abdominal_pain", "group": "abdomen", "escalate": False,
     "all": [r"(severe|worst|sudden|unbearable|excruciating) (abdominal|stomach|belly|tummy) pain"],
     "message": "Severe or sudden abdominal pain, especially with fever, vomiting or a hard, tender belly, "
                "needs prompt medical assessment."},
    {"id": "pregnancy_bleeding", "group": "pregnancy", "escalate": False,
     "all": [r"pregnan|trimester", r"bleed|spotting"],
     "message": "Bleeding during pregnancy should be checked promptly by a doctor or midwife. Seek urgent "
                "care if it is heavy or comes with pain, dizziness or fainting."},
]


def _red_flags(q: str):
    """Return (advisory messages, rule ids, escalate?) for an already-normalised question."""
    messages, ids, groups, escalate = [], [], set(), False
    for rule in RED_FLAG_RULES:
        if rule["group"] in groups:
            continue
        if all(re.search(p, q) for p in rule["all"]):
            groups.add(rule["group"])
            if rule["escalate"] and PERSONAL.search(q):
                escalate = True
            messages.append(rule["message"])
            ids.append(rule["id"])
    return messages, ids, escalate


def check(question: str) -> dict:
    """Return {"action": "emergency" | "dose" | "allow", ...}.

    "allow" also carries "red_flags": a list of fixed warnings the API should show at the top of the
    answer, before anything the LLM wrote. Run this on the WHOLE conversation text, not just the latest
    message, or "stiff neck" typed in a follow-up reply will be missed.
    """
    q = _normalize(question)
    if _is_crisis(q):
        return {"action": "emergency", "kind": "crisis", "message": (
            "I'm really sorry you're feeling this way, and I'm glad you told me. You deserve support right now. "
            f"Please call {CRISIS_LINE} (free, India), or {EMERGENCY_NUMBER} if you may be in immediate danger, "
            "or reach out to someone you trust and stay with them.")}

    flags, flag_ids, escalate = _red_flags(q)
    if (
        escalate
        or STRONG.search(q)
        or (
            CONTEXTUAL.search(q)
            and (
                PERSONAL.search(q)
                or re.search(r"bleeding (won'?t|will not|can'?t|cannot) stop", q)
            )
        )
    ):
        return {"action": "emergency", "kind": "medical", "message": (
            f"What you describe may be a medical emergency. Please call {EMERGENCY_NUMBER} or go to the nearest "
            "emergency department right now. Don't wait for more information from this assistant, and if possible "
            "don't go alone.")}
    if DOSE.search(q):
        return {"action": "dose", "message": (
            "I can't give medicine doses or say what is safe to take, because the right amount depends on age, "
            "weight, health conditions and the exact product. Please check the label or ask a pharmacist or doctor. "
            "For general information about a specific medicine, see MedlinePlus Drugs & Supplements: "
            "https://medlineplus.gov/druginformation.html")}
    return {"action": "allow", "personal": bool(PERSONAL_DX.search(q)), "red_flags": flags,
            "red_flag_ids": flag_ids}
