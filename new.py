import argparse
import re
import os
import random
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import names
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import keyboard
import sys

TYPOS = False

def main():
    """
    Main function to run the text generation script.
    """
    parser = argparse.ArgumentParser(description="Generate text with a given model.")
    parser.add_argument("-m", "--model_path", required=True, help="Path to the model directory.")
    parser.add_argument("-o", "--log_name", required=True, help="Name of the log file.")
    args = parser.parse_args()
    model_dir = args.model_path
    log_file_path = args.log_name + ".txt"
    # Validate model directory
    if not os.path.isdir(model_dir):
        print(f"❌ Directory '{model_dir}' does not exist.")
        sys.exit(1)
    print(f"✅ Model directory '{model_dir}' found.")
    # Handle log file
    if not os.path.isfile(log_file_path):
        with open(log_file_path, 'w'):
            pass
        print(f"✅ Log file '{log_file_path}' created.")
    else:
        print(f"✅ Log file '{log_file_path}' already exists.")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    tokenizer.pad_token = tokenizer.eos_token
    generating = False
    while True:
        user_input = input("Enter a prompt to start generating or 'q' to exit: ")
        if user_input.lower() == 'q':
            print("Exiting...")
            break
        if not generating:
            generating = True
            print("Generation started. Press 'Space' to stop.")
        while generating:
            try:
                if keyboard.is_pressed('space'):
                    generating = False
                    print("Generation stopped.")
                    break
                # Generate text
                generated_text = generate_text(user_input if user_input.strip() else "<>", model, tokenizer)  # Pass model and tokenizer
                if not generated_text:
                    print("Failed to generate text.")
                    break
                # Process the generated text
                modified_text = generated_text
                if is_bad_message(modified_text):
                    print("❌ [FILTERED] ")
                    with open(log_file_path, "a") as f:
                        f.write(f"[FILTERED]: {user_input}\n")
                    continue
                modified_text = replace_pronouns(modified_text)
                modified_text = correct_grammar(modified_text)
                modified_text = add_typos(modified_text)
                modified_text = rephrase_sentence(modified_text)
                # Print and log the modified text
                print(f"🤖: {modified_text}")
                with open(log_file_path, "a") as f:
                    f.write(f"[INPUT]: {user_input}\n[OUTPUT]: {modified_text}\n")
                user_input = modified_text  # set the output to the input for the next iteration
            except Exception as e:
                print(f"An error occurred: {e}")
                generating = False
                break

# Get the lists of male and female names
male_names = names.words('male.txt')
female_names = names.words('female.txt')

# Combine the lists and convert to a set
all_names_list = male_names + female_names
all_names_ntlk = set(all_names_list)

# getting all nouns from wordnet
nouns = set()
for synset in wn.all_synsets(pos='n'):
    for lemma in synset.lemmas():
        nouns.add(lemma.name())

# getting all verbs from wordnet
verbs = set()
for synset in wn.all_synsets(pos='v'):
    for lemma in synset.lemmas():
        verbs.add(lemma.name())

online_names = {
    "Pepe", "Wong", "Stu", "Aya", "Sean", "Jug", "Villa", "Alexander", "Muhammad",
    "Mahomet", "Bva", "Alec", "Angela", "Marisol", "Socrates", "Batman", "Kazi",
    "Egg", "Charlie", "Ana", "Rose", "Kayla", "Kafka", "Elaine", "Marley", "Abbie",
    "Felicity", "Griffith", "Ambi", "Abigail", "Toki", "D3", "Nova", "Moxie", "Pony",
    "Kim", "Marceline", "Pey", "Aizen", "Carlito", "Lou", "Jake", "Ifht", "Von", "Bam",
    "Aryan", "Shanene", "Metro", "Young", "Bambi", "Griffith", "Nicky", "Eden",
        "valen","levi","donpaolo","jedi","hane","darkless","thermf",
    "badomen","shura","xump","altruiste","thala","donoven",
    "freddyyyyyy","scarystories","miranda","eunatoun",
    "sirabernaclecleffermore","invectives","vi","johnnyconnecticut",
    "aaron","dontcallmegaga","shibo","ysxv","jzuz","theirryotu",
    "michaelmichaelmichaelmichael","nylaaaa","purevanillacookie",
    "doppek","oin","natsu","dahilya","allen","clitpeeler",
    "cannabisdoggo","rasputin","hexdsaint","stilltraveller","carlbot",
    "uli","sqamuel","lou","dual_blades","alva","firekingextreme",
    "serinaframe","paintonyourwalls","km","spinz","nswish",
    "thebreadman","jockiyshi","ambrrrrrrrr","lmaomomenat","morri",
    "quiet_star","pawulonml","jellyfish","blondi","lvcifer","azula",
    "victus","thedestroyer","supbitches","marleymagnotta","violet",
    "korephilereadmybio","josselin","serofu_havic","streetytailcoat",
    "kys","midgettoucher","astrid","oba","bartus","fredfinkle",
    "iriarain","suk","gma","jessieswig","metzler","asmodeus",
    "jimhalpert","oz","carli","thesuperdude","elon","randomuser",
    "sofia","hars","biblebot","sirius","constantine","juan","fzv",
    "marbleindexfingerbang","xezrza","penqu","moistscoffs","hedyyx",
    "integralanomalypussyslayer","wedo","vix","fmbot","starlost",
    "jeebzpepekek","elmerbudd","adeyhemi","lewarden","maur","ilhan",
    "entro","candylady","danae","quickvids","chapo","syli",
    "thetribalchief","maxlamnace","zach","its_cxnnxr","hwiskey",
    "thepopeofeuthanasia","op","papejunk","fatcat","ballsdex","itis",
    "dylan","averygoodegg","rolf","onanisticintransigent","kal",
    "crzyrmmr","ghste","dekablubbysubsuke","pingmeforyeastinfection",
    "vrissy","aatos","zion","bunii","jhna","youngbloodsuckr","azzy",
    "iloveniggersandwatermelons","xenogeistfireworkglazer","xd",
    "torigaleapple","queenelsa","garfield","notorioussad","xvxvxvx",
    "reptiliancathinoneaddict","rainmanelee","juzyx","demoncore",
    "pripretepe","littleyeetfetus","jf","truthordare","gome",
    "loucifah","querula","djballaballa","verminlord",
    "redstirrednesday","turboloser","abtheguy","schmuck",
    "pumpkinblairrr","tvui","ridemk","bonxzd","johanmonster",
    "sewersydal","sensitivethug","syla","may","banango","nikxsan",
    "bakedinatlanta","e","bigiron","solis","anna","ponyofsymmetry",
    "jhonhenryeden","laje","dopamine","quarvoxiastrixaxyrzael",
    "cigarettes","kc","melpothalia","sei","vilyth","ananick",
    "ruthlessnomadlimited","heavenlymode","lazydaisy","carlos_trk",
    "moxie","tulpa","yves","aly","misa","yagpdbxyz","mia",
    "ponyofchaos","bellabooba","heyheytroll","angela","shamil",
    "noobie","doktorvalkyria","bolbi","lordviper","rynixous","remoji",
    "pepesilvia","sedket","alex","dickensoncluseau","heinball","pnko",
    "sipan","laika","froosted","loona","somniowl",
    "squilliumthunderthighs","opposableurinator","sun_n_sky","heyrae",
    "pinku","tya","svad","egoastasia","candyman","krypton",
    "rigormortisrictus","ghostgirl","tralalerotralala","marriagebot",
    "everybodywantssome","hlliby","jashy","antighostping","owo",
    "jasonallen","ayaisntherenomore","notabot","spus","kulla",
    "chatgpt","mugle","myom","emilrealnotclickbait","ponyofentropy",
    "cowabunganinjatits","felipe","kuro","chugnterp","wolfgirl",
    "chato","intotdeuna","gustavo","dyno","therealsonic","bafoonery",
    "probot","patenidpateman","haa","steve","liverdeathxx","butterz",
    "cowboi","emmakimharrisimarealegirl","zentetsuken",
    "emiliafairheart","amon","pp","confusedserenity","jake_",
    "ichordakeyboardsmashr","sarah","eddiejackson","disboard",
    "keraunos","nandoo","miloromania","emmie","ctrlalt","daisu",
    "luca","aizen","deluxeedition","lastbreath","molliefun","aqui",
    "godschosenemperor","makeitaquote","yume","ostara",
    "brittanycross","hrvstr","aksel","naotastar","lucien",
    "kingratticus","firequeenextreme","ichordahappykeyboardsmashr",
    "sarahlynn","kyokasuigetsu","alivezan","rs","jae","cupidkixxes",
    "malignanthierophant","redpowned","axe","wildjay","dannyphantom",
    "cokemodel","plumbeaux","homura","bimpatty","chrinun","foidhater",
    "mundesi","reax","grxwler","hayden","confessions","annamae",
    "flawontop","conversionsoftwareversion","unitcorrector",
    "ranchranchranchranch","alanas","ivan","geeked","absentqualia",
    "rieus","boobbot","kk","villavanilla","poompa","casca","yonko",
    "lexxx","b","frin","cint_uh","oswald","boredslut","zoziozathx",
    "notsobot","tehres","yggdrasil","formerlywiola","yamchaglazer",
    "yvl","noro","luci","brookescosmicsystem","glowie",
    "sabbracadabra","malcolm","skunkpumper","hannah","narithedino",
    "evilastrid","faggotgaga","skyes","kalifan","justthebeanz_",
    "foxrilux","rambo","amit","nonsense_","lil_puppy_brat_readbio",
    "sephiroth","serana","jime","marley","knifehits","indyroses",
    "fakegirl","prettymuchhopeless","narcika","katus","statbot","stu",
    "mooncake","caliwiser_","niggertron","rizer","harys","rudy","aya",
    "nonserviam","jraceing","dariussings","fibo","qqq","daniellarson",
    "pimpbutters","scurris","dummie","cunnyrapisthitler","anor",
    "ichorrenewal","nfrn","dsme","cryptwell","slow_death",
    "succsukebehaved","freakitty","yuani","karabinek","pumsters",
    "desupair","loko","reii","nayz","casfdaf","jinaa","poketwo",
    "samy","ab","diogenesbarrel","wrong_house","simi","mrkennedy",
    "messianicmanic","chaim","tinj","churchofai","marleyroslyakov",
    "aliiii","hll","ninixten","meow","jbo","griffith","tobi",
    "sebestrong","evilsaint","william","emi","kafka","gamel",
    "jessicatheblackqueen","legoindianajones","mami","vray",
    "livingdeadgirl","kiko","jessie","tatsu","thefaggot","girlgore",
    "southerngentleman","factcheck","iiza","jockiemusic",
    "reincarnatedgoldfish","hypertransport","naylabambi","blacksheep",
    "kkk","bartek","elli","ina","lyn","joyboy","pissfart",
    "questioningangels","kalashnik","vicvanderdoomlinde",
    "jessicaswig","randyprozac","euthymia","losersympa","vicodin",
    "kar","cher","goblinora","soren","aoi","lynn","alicia","arcanist",
    "jay", "superdude"
}

all_names = all_names_ntlk | online_names

vague_replies = {
    "same", "yes", "no", "true", "correct", "ok", "okay", "sure", "well", "maybe", "lol",
    "yeah", "yep", "nope", "thanks", "cheers", "haha", "indeed", "cool", "nice", "k", "hm",
    "hmm", "huh", "meh", "eh", "yuh", "fr", "smh", "bruh", "exactly", "fair", "real", "idk",
    "oh", "fine", "good", "yikes", "aww", "ouch", "later", "omg", "mhm", "whatever",
    "nah", "sup", "yo", "lit", "fire", "savage", "seriously", "tbh", "imho", "frfr", "wyd",
    "thx", "lmk", "np", "ldk", "gucci", "wbu", "bff", "sis", "bro", "chill", "bet", "woke",
    "nope", "stoked", "skrt", "deadass", "hyped", "facts", "nothing", "maybe", "later", "fine",
    "yep", "yuh", "lol", "sure", "bruh", "yeah", "nah", "k", "ok", "okay", "chill",
    "imho", "idk", "nothing", "lmao", "deadass", "cool", "meh", "fr", "facts", "gucci",
    "smh", "hyped", "wbu", "tbh", "fire", "skrt", "cheers", "lmk", "hmm", "yo", "lit", "stoked",
    "bff", "bro", "fine", "seriously", "maybe", "nope", "yikes", "woke",
    "wild", "insane", "crazy", "unreal", "rough", "valid", "based", "solid",
    "dope", "aww", "ouch", "later", "omg", "welp", "mhm", "i see", "whatever", "good",
    "decent", "defo", "deffo", "frl", "yeah", "yep", "ehh", "yuh", "bruh", "tyy", "hot",
    "sick", "lit", "amazing", "decent", "sus", "af", "btw"
}

question_words = {
    "who", "what", "why", "how", "when", "where",
    "is she", "is he", "are you", "do you", "did you", "have you",
    "left or right", "right or left", "which one", "how much", "how many",
    "what if", "who's", "what's", "where's", "when's", "why's", "how's",
    "aren't", "can't", "won't", "haven't", "didn't", "doesn't", "weren't",
    "is it", "was it", "will it", "can it", "has it",
    "will you", "can you", "would you", "could you", "should you",
    "is there", "are there", "was there", "were there", "has there",
    "do we", "can we", "should we", "are we", "were we", "will we",
    "is this", "does this", "should this", "can this", "was this", "could this",
    "is that", "was that", "could that", "should that", "can that",
    "why do", "why did", "why would", "why should", "why does", "why can't",
    "how do", "how does", "how would", "how should",
    "what are", "what do", "what did", "what would", "what could",
    "who are", "who do", "who did", "who would"
}

vulgar_nouns = {
    "bitch", "hoe", "whore", "slut", "nigga", "nigger", "cunt", "twat", "skank", "fuckface", "shithead",
    "asshole", "dumbass", "jackass", "prick", "cock", "dick", "pussy", "retard", "dipshit", "shitstain",
    "faggot", "fag", "tranny", "kike", "chink", "gook", "spic", "wetback", "cracker", "redneck",
    "cocksucker", "motherfucker", "fucker", "bastard", "slanteye", "mongoloid", "slutbag", "douchebag",
    "douche", "queer", "nancyboy", "pansy", "buttmunch", "cumdumpster", "semenhound", "fucktard",
    "shitlord", "buttlicker", "arsehole", "bollocks", "wanker", "tosser", "slag", "pillock", "git",
    "twit", "dipwad", "shitbrain", "scumbag", "dickhead", "nutsack", "balllicker", "taint", "snatch",
    "cumrag", "pissface", "dingleberry", "fuckwit", "arsewipe", "shitbag", "anus", "analbead",
    "crackwhore", "methhead", "junkie", "stoner", "dopehead", "fuckboy", "fuckgirl", "manwhore",
    "thot", "camwhore", "incel", "neckbeard", "simp", "cringe-lord", "edgelord", "fleshlight",
     "milf", "dilf", "numbnuts", "shitlicker", "ballsniffer", "knobhead", "bellend", "meathead"
}

deictic_words = {
    "here", "there", "now", "then", "that", "this", "those", "these",
    "such", "thus", "hence", "where", "when", "afterward", "beforehand",
    "today", "tomorrow", "yesterday"
}

pronouns = {
    # Male to Female Pronouns
    "he": "she", "him": "her", "his": "her",
    "he's": "she's", "he’d": "she’d", "he’ll": "she’ll",
    "he’ve": "she’ve", "he'd": "she’d", "he’ll": "she’ll",
    "he'd've": "she'd've", "he'll've": "she’ll've",

    # Common misspellings/typos (though handling these robustly might require more advanced techniques)
    "himm": "her", "herr": "her", "hur": "her", "hym": "her", "hurr": "her",
    "hhe": "sshe", "hem": "them", # Assuming this was a typo for 'them' in a male context? If not, clarify.
    "hes": "shes", "hed": "she'd", "hee": "shee",

    # Female Pronouns (already quite comprehensive)
    "she": "she", "her": "her", "hers": "hers", # Added possessive 'hers'
    "she's": "she's", "she’d": "she’d", "she’ll": "she’ll",
    "she’ve": "she’ve", "she'd": "she'd", "she'll": "she'll",
    "shes": "shes", "shee": "shee", "sshe": "sshe",

    # Gender-Neutral Pronouns (including singular 'they')
    "they": "they", "them": "them", "their": "their", "theirs": "theirs", # Added possessives
    "they're": "they're", "they’ve": "they’ve", "they’d": "they’d", "they’ll": "they’ll",
    "they’d’ve": "they’d’ve",
    "themz": "themz",

    # Second Person Pronouns
    "you": "you", "your": "your", "yours": "yours", # Added possessive 'yours'
    "you're": "you're", "you've": "you've", "you’d": "you’d", "you’ll": "you’ll", "you’d’ve": "you’d’ve",
    "youre": "youre", "u": "u", "uu": "u", "uuu": "u", "uuuu": "u", "yu": "u", "youu": "you", "youuu": "you", "yoou": "you",
    "ya": "ya", "ya’ll": "ya’ll", "yall": "yall",
    "ur": "ur", "ure": "ure", "urself": "yourself", "yourself": "yourself",
    "urs": "yours", "u r": "you are", "u're": "you're", "yer": "your", "yous": "yous",

    # Plural Forms
    "we": "we", "us": "us", "our": "our", "ours": "ours", # Added possessives
    "we're": "we're", "we’ve": "we’ve", "we’d": "we’d", "we’ll": "we’ll", "we’d’ve": "we’d’ve",

    # Non-Human/Neutral Pronouns
    "it": "it", "its": "its",
    "it’s": "it’s", "it's": "it's", "it'd": "it'd", "it'll": "it'll", "it'd’ve": "it'd’ve",
    "itz": "itz", "thay": "thay",

    # Miscellaneous
    "everyone": "everyone",
    "ones": "ones", # Possessive form
}


question_starters = {
    "is", "are", "do", "does", "did", "have",
    "has", "can", "could", "would", "should", "will",
    "which", "am", "was", "were", "huh", "eh"
}

banned_words = {
    "your", "you", "u", "yall", "you're", "you've", "you'd", "you'll", "ur", "uu", "uuu", "uuuu", "yu", "youu", "youuu", "yoou", "ya", "ya'll", "yer", "urself", "yourself", "urs", "yours", "u r", "u're"
}
# "him", "himm", "his", # "her", "herr", "he", "hee", "she", "shee", "he'd", "he's", "he'll", "he'd've",
# "he'll've", "she'd", "she's", "she'll", "she'd've", "she'll've", "hed", "hes",
# "shes", "she", "he", "him", "he's", "he’d", "he’ll", "he’ve", "he’d’ve", "hes",
# "he'll", "hed", "hee", "hhe", "hym", "she", "her", "she's", "she’d", "she’ll",
# "she’ve", "she’d’ve", "shes", "she'll", "shee", "sshe", "hurr", "hur"
prepositions = {
    "from", "by", "with", "to", "at", "on", "in", "for", "of", "about", "over", "under", "between", "among"
}
allowed_after_its = { "getting", "raining", "snowing", "cold", "dark", "quiet", "becoming", "starting", "ending", "fading", "shifting", "over", "pointless", "fine", "inevitable", "happening", "late", "nothing" }
nouns.update(vulgar_nouns)
def is_noun(word):
    return bool(wn.synsets(word, pos=wn.NOUN))

def is_verb(word):
    return bool(wn.synsets(word, pos=wn.VERB))

def is_fuzzy_verb(word, verbs):
    return word.lower() in verbs

def is_fuzzy_noun(word, nouns):
    return word.lower() in nouns

def is_name(word):
    return word in all_names

def is_bad_message(msg):
    original_msg = msg  # Store the original message
    lower = msg.lower().strip()
    tokens = re.findall(r"\b\w+'\w+|\w+\b", lower)  # Banned words (exact match only)
    for t in tokens:
        if t in banned_words:
            return True
    # Fragment check
    def is_fragment(tokens, verbs, nouns, vulgar_nouns):
        has_verb = any(is_fuzzy_verb(t, verbs) for t in tokens)
        has_noun = any(is_fuzzy_noun(t, nouns | vulgar_nouns) for t in tokens)
        return has_verb and not has_noun
    if is_fragment(tokens, verbs, nouns, vulgar_nouns):
        return True
    # Check for vulgar nouns
    for t in tokens:
        if t in vulgar_nouns:
            return True
    # Check for deictic words
    for t in tokens:
        if t in deictic_words:
            return True
    # Check for pronoun issues
    for t in tokens:
        if t in pronouns:
            return True
    # Check for direct questions
    first_word_lower = tokens[0].lower() if tokens else ""
    if first_word_lower in question_starters:
        return True
    if "?" in msg:
        return True
    # Check for "it's [adjective]" without allowed adjectives
    if len(tokens) >= 2 and tokens[0].lower() == "it's":
        second_word_lower = tokens[1].lower()
        if not any(allowed_word == second_word_lower for allowed_word in allowed_after_its):
            return True
    # Additional check for "i am", "i was", "i can", "i will", "i have", "i had", "i'm"
    if len(tokens) >= 2 and tokens[0].lower() in ("i", "i'm"):
        second_word_lower = tokens[1].lower()
        if second_word_lower in ("am", "was","can","will","have","had"):
            return True
    # Check for "you are", "you were", "you can", "you will", "you have", "you had", "you're"
    if len(tokens) >= 2 and tokens[0].lower() in ("you", "you're"):
        second_word_lower = tokens[1].lower()
        if second_word_lower in ("are", "were", "can", "will", "have", "had"):
            return True
    return False
def replace_pronouns(text):
    """
    Replaces male pronouns with female pronouns, handling edge cases and variations.
    """
    global pronouns
    new_text = text
    # Create a sorted list of pronoun keys, longest first, to avoid partial replacements
    sorted_pronoun_keys = sorted(pronouns.keys(), key=len, reverse=True)
    for key in sorted_pronoun_keys:
        # Create a regex pattern that matches the pronoun,
        # accounting for word boundaries and case insensitivity.
        pattern = r'\b' + re.escape(key) + r'\b'
        new_text = re.sub(pattern, pronouns[key], new_text, flags=re.IGNORECASE)
    return new_text
def correct_grammar(text):
    """
    Corrects simple grammar errors in the given text.
    """
    # Fix contractions
    text = re.sub(r"(\w+)'s", r"\1's", text)
    text = re.sub(r"(\w+)n't", r"\1n't", text)
    text = re.sub(r"(\w+)'re", r"\1're", text)
    text = re.sub(r"(\w+)'ve", r"\1've", text)
    text = re.sub(r"(\w+)'ll", r"\1'll", text)
    text = re.sub(r"(\w+)'d", r"\1'd", text)
    # Fix "i" capitalization
    text = re.sub(r"\bi\b", "I", text)
    # Fix multiple spaces
    text = re.sub(r" +", " ", text)
    # Remove leading/trailing spaces
    text = text.strip()
    return text
def add_typos(text):
    """
    Randomly introduces typos into the given text.  Includes vowel and consonant swaps,
    letter drops, and repeated letters.
    """
    if not TYPOS:
        return text
    vowels = 'aeiou'
    consonants = 'bcdfghjklmnpqrstvwxyz'
    new_text = ""
    for char in text:
        if random.random() < 0.1:  # 10% chance of a typo
            typo_type = random.randint(1, 4)
            if typo_type == 1 and char.lower() in vowels:
                # Vowel swap
                new_text += random.choice(vowels)
            elif typo_type == 2 and char.lower() in consonants:
                # Consonant swap
                new_text += random.choice(consonants)
            elif typo_type == 3:
                # Letter drop
                continue
            elif typo_type == 4:
                # Repeated letter
                new_text += char * 2
        else:
            new_text += char
    return new_text
def rephrase_sentence(sentence):
    """
    Rephrases a sentence, attempting to replace nouns and verbs with synonyms.
    """
    global nouns, verbs
    words = sentence.split()
    new_words = []
    for word in words:
        if is_fuzzy_noun(word, nouns):
            synsets = wn.synsets(word, pos=wn.NOUN)
            if synsets:
                synonyms = [lemma.name() for lemma in synsets[0].lemmas()]
                if synonyms and synonyms[0].lower() != word.lower():
                    new_words.append(random.choice(synonyms))
                else:
                    new_words.append(word)
            else:
                new_words.append(word)
        elif is_fuzzy_verb(word, verbs):
            synsets = wn.synsets(word, pos=wn.VERB)
            if synsets:
                synonyms = [lemma.name() for lemma in synsets[0].lemmas()]
                if synonyms and synonyms[0].lower() != word.lower():
                    new_words.append(random.choice(synonyms))
                else:
                    new_words.append(word)
            else:
                new_words.append(word)
        else:
            new_words.append(word)
    return " ".join(new_words)
def generate_text(prompt, model, tokenizer):
    """
    Generates text based on the given prompt using the provided model and tokenizer.
    """
    global device
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    # Manually set pad_token_id if it's not set
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id  # Or any other appropriate token ID
    # Ensure attention_mask is passed if your model expects it
    attention_mask = inputs.get("attention_mask", None)  # Get attention mask if it exists
    try:
        outputs = model.generate(
            input_ids,
            max_length=100,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id,  # Use the pad token ID
            attention_mask=attention_mask,  # Pass attention mask
        )
    except Exception as e:
        print(f"Error during generation: {e}")
        return ""
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

if __name__ == "__main__":
    main()