"""Curated multilingual lexicons (English, Hindi, Hinglish).

They are NOT used for the classification decision (that is the BERT model's job).
They are used for:
  * language identification (Hinglish markers),
  * Hinglish spelling normalisation,
  * information extraction (identity groups, targets),
  * explainability (mapping an influential word to a known abuse category).
"""

# ---------------------------------------------------------------- Hinglish markers
# Frequent romanised-Hindi function/content words. Presence of several => Hinglish.
HINGLISH_MARKERS = {
    "hai", "hain", "ho", "hoga", "tha", "thi", "nahi", "nahin", "nhi", "kya", "kyu", "kyun",
    "kaise", "kaun", "kab", "kahan", "mera", "meri", "mere", "tera", "teri", "tere", "tum",
    "tu", "tumhara", "tumhari", "aap", "apna", "apni", "hum", "hamara", "main", "mai", "mujhe",
    "tujhe", "usko", "isko", "unko", "yeh", "ye", "woh", "wo", "bhai", "yaar", "yar", "accha",
    "acha", "theek", "thik", "bahut", "bohot", "bhot", "kar", "karo", "karna", "kr", "krna",
    "raha", "rahi", "rahe", "gaya", "gayi", "aur", "lekin", "par", "ki", "ka", "ke", "ko",
    "se", "mein", "me", "bhi", "sab", "kuch", "koi", "abhi", "kal", "aaj", "pagal", "chal",
    "chalo", "dekh", "dekho", "bol", "bolo", "samajh", "jaa", "ja", "jao", "nikal", "matlab",
    "haan", "ji", "bata", "batao", "log", "logon", "wala", "wali", "wale", "tumhe", "hoon", "hu",
}

# Spelling normalisation for common Hinglish variants (applied in preprocessing).
HINGLISH_NORMALIZATION = {
    "nhi": "nahi", "nai": "nahi", "nahin": "nahi", "kyu": "kyun", "kyon": "kyun",
    "hy": "hai", "bhot": "bahut", "bohot": "bahut", "bht": "bahut",
    "kr": "kar", "krna": "karna", "kro": "karo", "yr": "yaar", "yar": "yaar", "mai": "main",
    "acha": "accha", "thik": "theek", "pls": "please", "plz": "please", "tm": "tum", "hu": "hoon", "sb": "sab", "kch": "kuch",
}

# ---------------------------------------------------------------- abuse lexicon
# word -> category (one of the Jigsaw labels). Lower-case, romanised or Devanagari.
ABUSE_LEXICON = {
    # English insults
    "idiot": "insult", "stupid": "insult", "moron": "insult", "dumb": "insult", "loser": "insult",
    "fool": "insult", "pathetic": "insult", "ugly": "insult", "worthless": "insult", "trash": "insult",
    "clown": "insult", "imbecile": "insult", "retard": "insult", "retarded": "insult", "freak": "insult",
    "disgusting": "toxic", "shut": "toxic", "hate": "toxic", "garbage": "toxic", "scum": "severe_toxic",
    # English obscenity
    "fuck": "obscene", "fucking": "obscene", "fucker": "obscene", "shit": "obscene", "bullshit": "obscene",
    "bitch": "obscene", "bastard": "obscene", "asshole": "obscene", "ass": "obscene", "dick": "obscene",
    "crap": "obscene", "damn": "obscene", "cunt": "severe_toxic", "motherfucker": "severe_toxic",
    # English threats
    "kill": "threat", "murder": "threat", "die": "threat", "dead": "threat", "shoot": "threat",
    "stab": "threat", "destroy": "threat", "hang": "threat", "beat": "threat", "burn": "threat",
    # Hinglish (romanised Hindi)
    "pagal": "insult", "bewakoof": "insult", "bevkoof": "insult", "gadha": "insult", "ullu": "insult",
    "nalayak": "insult", "kamina": "insult", "kameena": "insult", "kamine": "insult", "chutiya": "obscene",
    "chutiye": "obscene", "harami": "obscene", "haramkhor": "obscene", "kutta": "insult",
    "kutte": "insult", "kuttiya": "obscene", "saala": "toxic", "sala": "toxic", "saali": "toxic",
    "gandu": "obscene", "bhosdike": "severe_toxic", "bsdk": "severe_toxic", "madarchod": "severe_toxic",
    "mc": "severe_toxic", "behenchod": "severe_toxic", "bhenchod": "severe_toxic", "bc": "severe_toxic",
    "randi": "severe_toxic", "jhatu": "obscene", "lodu": "obscene", "tatti": "obscene",
    "maar": "threat", "maarunga": "threat", "marunga": "threat", "khatam": "threat", "kaat": "threat",
    # Hindi (Devanagari)
    "पागल": "insult", "बेवकूफ": "insult", "मूर्ख": "insult", "गधा": "insult", "उल्लू": "insult",
    "नालायक": "insult", "कमीना": "insult", "कुत्ता": "insult", "कुत्ते": "insult", "हरामी": "obscene",
    "साला": "toxic", "चुतिया": "obscene", "गांडू": "obscene", "मादरचोद": "severe_toxic",
    "भेनचोद": "severe_toxic", "रंडी": "severe_toxic", "मारूंगा": "threat",
    "मारो": "threat", "खत्म": "threat", "नफरत": "toxic", "घटिया": "insult",
}

# ---------------------------------------------------------------- identity groups (IE)
IDENTITY_TERMS = {
    "religion": ["muslim", "muslims", "hindu", "hindus", "christian", "christians", "sikh", "sikhs",
                 "jew", "jews", "jewish", "islam", "buddhist", "मुसलमान", "मुस्लिम", "हिंदू", "हिन्दू",
                 "ईसाई", "सिख", "musalman", "mulla", "mulle"],
    "caste": ["dalit", "dalits", "brahmin", "brahmins", "caste", "jaat", "jat", "दलित", "ब्राह्मण", "जाति"],
    "gender": ["women", "woman", "girl", "girls", "female", "females", "men", "man", "aurat", "aurate",
               "aurato", "ladki", "ladkiyan", "महिला", "औरत", "लड़की", "feminist", "feminists"],
    "sexual_orientation": ["gay", "gays", "lesbian", "lgbt", "lgbtq", "trans", "transgender", "queer",
                           "homosexual", "hijra", "hijre", "hijro", "किन्नर"],
    "nationality_ethnicity": ["indian", "indians", "pakistani", "pakistanis", "chinese", "african",
                              "black", "blacks", "white", "whites", "asian", "asians", "bihari",
                              "biharis", "madrasi", "bangladeshi", "immigrant", "immigrants"],
    "disability": ["disabled", "handicapped", "autistic", "blind", "deaf", "viklang"],
}

# Second-person pronouns => the comment is directed at an individual.
SECOND_PERSON = {"you", "your", "yours", "you're", "youre", "u", "ur", "yourself",
                 "tu", "tum", "tumhara", "tumhari", "tera", "teri", "tere", "tujhe", "tumhe",
                 "aap", "aapka", "तू", "तुम", "तेरा", "तेरी", "तुम्हारा", "तुझे", "आप"}
