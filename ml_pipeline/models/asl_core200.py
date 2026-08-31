"""The 200-word ASL vocabulary the app ships.

Cutting 525 -> 200 is not about model capacity, it is about confusability and
per-class data. ASL Citizen gives a median of 15 training clips per class, so a
525-way decision is being made from very little evidence per word, and the
corpus is full of pairs that differ by details the phone cannot resolve at ~5
frames per sign (COOKIE / CHURN, PECK / BEAK, SIDEBURNS / BRAIDS). Dropping the
long tail of low-value words removes those competitors outright.

The selection is deliberately weighted toward what a deaf user actually needs to
say to a hearing stranger — needs, people, places, time, and the function words
that let signs be chained into a sentence — rather than toward whatever the
corpus happens to contain. Words like ASTRONAUT, CORKSCREW, JIGSAWPUZZLE and
CIRCUSWHEEL are real ASL and well represented, but they buy nothing in a
translation app and every one of them is another chance to be wrong.

CORE is the list Dan selected. Everything else is grouped by the role it plays
in a sentence, so the vocabulary can be defended in the write-up as a design
decision rather than an arbitrary cut.
"""

# Dan's selection.
CORE = """
AND ANYONE BAD BATHROOM CAN COME DRINK EAT FAMILY FEEL FRIEND GO GOOD HAPPY
HELLO HELP HOME HOW KNOW LATER LEARN LIKE MONEY MORE NEED NO NOW PHONE PLEASE
REALLY SCHOOL SORRY STOP THANKYOU TIME TODAY TOMORROW UNDERSTAND WANT WATER WE
WHAT WHEN WHERE WHO WHY WORK YES YESTERDAY YOU YOUR HANDSOME
"""

# Every pronoun and person-reference the corpus has. Requested explicitly, and
# they matter more than their frequency suggests: without them a chained gloss
# cannot say who did what to whom.
PRONOUNS = """
ME MY MINE MYSELF SELF HE HIS THEY OTHER PEOPLE
"""

# Function words. These are what turn a list of signs into a sentence, so they
# carry the gloss-to-English feature rather than any single content word.
GRAMMAR = """
NOT BUT BECAUSE IF WITH FOR TO OR ALL SOME MANY MUST SHOULD WILL NEVER ALWAYS
SOMETIMES THAN EACH MAYBE
"""

FAMILY = """
MOTHER FATHER SISTER BROTHER BABY SON WIFE HUSBAND PARENTS KID BOY GIRL MAN
WOMAN
"""

# Health and urgency: the highest-stakes thing the app can be asked to translate.
HEALTH = """
HOSPITAL DOCTOR SICK PAIN HEADACHE HEALTH HUNGRY PATIENT
"""

FOOD = """
BREAKFAST RESTAURANT KITCHEN JUICE MEAT APPLE COOKIE PIZZA
"""

TIME = """
MORNING AFTERNOON NIGHT NOON DAY WEEK YEAR SOON FINISH AGAIN FUTURE EVERYDAY
"""

PLACES = """
HOTEL FARM FLOOR WORLD
"""

# Deaf-community and communication vocabulary — the subject matter of the app.
COMMUNICATION = """
NAME SIGN DEAF HEARING LISTEN READ WRITE TEACH PRACTICE STORY
"""

FEELINGS = """
SAD LOVE BORED UPSET HOPE WISH NICE FRIENDLY RUDE SMART DREAM
"""

DESCRIBE = """
BIG SMALL HOT COLD HARD EASY DIRTY EMPTY DIFFERENT FAR NEAR DARK FINE
"""

VERBS = """
BUY MOVE RUN WATCH FOLLOW CONTINUE ENTER CLOSE COUNT COPY HIDE HUG KISS LAUGH
SMILE SING DANCE WAVE
"""

OBJECTS = """
CAR BOOK KEY PEN CLOTHES WALLET CAMERA PICTURE MOVIE LETTER TOY
"""

SOCIAL = """
WELCOME GREET INVITE PARTY GIFT PROMISE SOCIALIZE PEACE
"""

# ONE only. The corpus has FOUR and THIRD but not a full 1-10 set, and an
# isolated number is worse than none — it invites the user to try counting and
# then fails on everything except the one word that happens to be in vocabulary.
# Numbers belong in fingerspelling/number mode, not here.
NUMBERS = """
ONE
"""

GROUPS = {
    "core (selected)": CORE,
    "pronouns": PRONOUNS,
    "grammar": GRAMMAR,
    "family": FAMILY,
    "health": HEALTH,
    "food": FOOD,
    "time": TIME,
    "places": PLACES,
    "communication": COMMUNICATION,
    "feelings": FEELINGS,
    "describe": DESCRIBE,
    "verbs": VERBS,
    "objects": OBJECTS,
    "social": SOCIAL,
    "numbers": NUMBERS,
}

# Order is irrelevant to training (labels are re-indexed), but a stable sorted
# list keeps the generated label file diffable between runs.
CORE200 = sorted({w for g in GROUPS.values() for w in g.split()})


if __name__ == "__main__":
    total = 0
    for name, g in GROUPS.items():
        n = len(g.split())
        total += n
        print(f"  {name:<18} {n:>3}")
    print(f"  {'-'*18} {'-'*3}")
    print(f"  {'listed':<18} {total:>3}")
    print(f"  {'unique':<18} {len(CORE200):>3}")
