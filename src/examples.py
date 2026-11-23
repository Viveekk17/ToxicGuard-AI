"""Demo comments and a small labelled multilingual benchmark (used for the live self-test)."""

DEMO_EXAMPLES = {
    "— choose an example —": "",
    "🇬🇧 Clean · constructive feedback": "Thanks for the detailed explanation, but I think the second argument needs a source.",
    "🇬🇧 Insult": "You are a complete idiot and a pathetic loser, nobody cares about your stupid opinion.",
    "🇬🇧 Threat": "If I ever see you again I will kill you and your whole family.",
    "🇬🇧 Obscene": "What the fuck is this shit, are you fucking kidding me?",
    "🇬🇧 Identity hate": "All immigrants are filthy criminals and should be thrown out of this country.",
    "🇮🇳 Hindi · clean": "यह वीडियो बहुत अच्छा है, इसे बनाने के लिए धन्यवाद।",
    "🇮🇳 Hindi · insult": "तू एक नंबर का बेवकूफ और नालायक इंसान है, तेरी कोई इज्जत नहीं है।",
    "🇮🇳 Hindi · threat": "अगर दोबारा यहाँ दिखा तो तुझे जान से मार दूंगा।",
    "🔀 Hinglish · clean": "Bhai ye video bahut accha tha, aise hi content banate raho!",
    "🔀 Hinglish · insult": "Tu pagal hai kya? Dimag kharab hai tera, chup kar bewakoof.",
    "🔀 Hinglish · abusive": "Saale kamine, teri aukaat kya hai? Nikal yahan se harami.",
    "🔀 Hinglish · threat": "Ek baar bahar mil, tujhe jaan se maar dunga.",
}

# (text, is_toxic, language)
BENCHMARK = [
    ("Great article, I learned a lot from this.", 0, "en"),
    ("Could you please add references for the second section?", 0, "en"),
    ("I respectfully disagree with your conclusion.", 0, "en"),
    ("This is the best explanation I have read so far, thank you!", 0, "en"),
    ("You are a worthless idiot and everyone hates you.", 1, "en"),
    ("Shut up you stupid moron.", 1, "en"),
    ("I will find you and beat you to death.", 1, "en"),
    ("Go to hell you pathetic piece of shit.", 1, "en"),
    ("आपका काम बहुत अच्छा है, ऐसे ही आगे बढ़ते रहिए।", 0, "hi"),
    ("मुझे यह जानकारी बहुत उपयोगी लगी, धन्यवाद।", 0, "hi"),
    ("आज मौसम बहुत सुहाना है।", 0, "hi"),
    ("तू पागल है क्या, बेवकूफ कहीं का।", 1, "hi"),
    ("तुझे जान से मार दूंगा कमीने।", 1, "hi"),
    ("तुम जैसे घटिया लोगों को देश से निकाल देना चाहिए।", 1, "hi"),
    ("Bhai kya mast video banaya hai, maza aa gaya.", 0, "hinglish"),
    ("Aapki baat bilkul sahi hai, main agree karta hoon.", 0, "hinglish"),
    ("Kal milte hain college mein, notes le aana.", 0, "hinglish"),
    ("Yaar ye movie bahut acchi thi, sabko dekhni chahiye.", 0, "hinglish"),
    ("Tu ek number ka chutiya hai.", 1, "hinglish"),
    ("Chup kar bewakoof, teri aukaat nahi hai.", 1, "hinglish"),
    ("Saale harami, nikal yahan se.", 1, "hinglish"),
    ("Tujhe ghar aake maarunga, yaad rakhna.", 1, "hinglish"),
]

# Neutral sample pool for the UI: grouped by language only (no category hints), so the
# model - not the user - decides whether a comment is toxic and which categories apply.
SAMPLE_COMMENTS = {
    "English": [v for k, v in DEMO_EXAMPLES.items() if k.startswith("🇬🇧")],
    "Hindi": [v for k, v in DEMO_EXAMPLES.items() if k.startswith("🇮🇳")],
    "Hinglish": [v for k, v in DEMO_EXAMPLES.items() if k.startswith("🔀")],
}
