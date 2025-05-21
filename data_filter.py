import json
import re
import difflib
import os
import nltk
from nltk.corpus import wordnet as wn
# from fuzzywuzzy import fuzz  # Removed fuzzywuzzy import
from concurrent.futures import ThreadPoolExecutor
from functools import partial
# import time # Removed the original time import
import sys
from nltk.corpus import names
from tqdm import tqdm # Import tqdm
import argparse

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
    "milf", "dilf", "numbnuts", "shitlicker", "ballsniffer", "knobhead", "bellend", "meathead",
    "ghetto"
}

deictic_words = {
    "here", "there", "now", "then", "that", "this", "those", "these",
    "such", "thus", "hence", "where", "when", "afterward", "beforehand",
    "today", "tomorrow", "yesterday"
}

pronouns = {
    "he": "she", "him": "her", "he's": "she's", "he’d": "she’d", "he’ll": "she’ll",
    "he’ve": "she’ve", "he’d’ve": "she’d’ve",
    "she": "she", "her": "her", "she's": "she's", "she’d": "she’d", "she’ll": "she’ll",
    "she’ve": "she’ve", "she’d’ve": "she’d’ve",
    "they": "they", "them": "them", "they're": "they're", "they’ve": "they’ve",
    "they’d": "they’d", "they’ll": "they’ll", "they’d’ve": "they’d’ve",
    "you": "you", "your": "your", "you're": "you're", "you've": "you've",
    "you’d": "you’d", "you’ll": "you’ll", "you’d’ve": "you’d’ve",
    "u": "u", "ya": "ya", "yall": "yall", "we": "we", "us": "us",
    "we're": "we're", "we’ve": "we’ve", "we’d": "we’d", "we’ll": "we’ll",
    "we’d’ve": "we’d’ve",
    "it": "it", "its": "its", "it’s": "it’s", "it's": "it's",
    "it'd": "it'd", "it'll": "it'll", "it'd’ve": "it'd’ve", "everyone": "everyone",
    "youre": "youre",
    "ur": "ur", "ure": "ure", "ya’ll": "ya’ll", "itz": "itz", "thay": "thay",
    "shee": "shee", "hurr": "hurr", "hym": "hym", "hem": "hem", "themz": "themz",
    "yous": "yous"
}

question_starters = {
    "is", "are", "do", "does", "did", "have",
    "has", "can", "could", "would", "should", "will",
    "which", "am", "was", "were", "huh", "eh"
}

banned_words = {
    "your", "you", "u", "yall", "you're", "you've", "you'd", "you'll", "ur",
    "uu", "uuu", "uuuu", "yu", "youu", "youuu", "yoou", "ya", "ya'll", "yer",
    "urself", "yourself", "urs", "yours", "u r", "u're", "him", "himm", "his",
    "her", "herr", "he", "hee", "she", "shee", "he'd", "he's", "he'll", "he'd've",
    "he'll've", "she'd", "she's", "she'll", "she'd've", "she'll've", "hed", "hes",
    "shes", "she"
}

prepositions = {
    "from", "by", "with", "to", "at", "on", "in", "for", "of", "about", "over", "under",
    "between", "among"
}

allowed_after_its = {
    "getting", "raining", "snowing", "cold", "dark", "quiet",
    "becoming", "starting", "ending", "fading", "shifting", "over",
    "pointless", "fine", "inevitable", "happening", "late", "nothing"
}

nouns.update(vulgar_nouns)

def is_noun(word):
    return bool(wn.synsets(word, pos=wn.NOUN))

def is_verb(word):
    return bool(wn.synsets(word, pos=wn.VERB))

def is_fuzzy_verb(word, verbs):
    if len(word) <= 2:
        return word.lower() in verbs
    for v in verbs:
        if difflib.SequenceMatcher(None, word.lower(), v).ratio() >= 0.83:
            return True
    return False

def is_fuzzy_noun(word, nouns):
    if len(word) <= 2:
        return word.lower() in nouns
    for n in nouns:
        if difflib.SequenceMatcher(None, word.lower(), n).ratio() >= 0.83:
            return True
    return False

def is_name(word):
    return word in all_names
def is_bad_message(msg):
    original_msg = msg  # Store the original message
    lower = msg.lower().strip()
    tokens = re.findall(r"\b\w+'\w+|\w+\b", lower)

    # Banned words (exact match only)
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

    # "I" modal check
    def check_i_modal_verbs(message, verbs):
        toks = message.lower().split()
        if toks and toks[0] == "i":
            modal_verbs = {"can", "could", "might", "may", "will", "would", "should", "did", "do", "was", "am", "have", "had"}
            if len(toks) > 1 and toks[1] in modal_verbs:
                has_verb = any(v in verbs for v in toks[2:])
                return not has_verb
            return True
        return False

    if check_i_modal_verbs(msg, verbs):
        return True

    # Vague reply check
    def is_vague(msg):
        tokens = msg.strip().split()
        if len(tokens) <= 4:
            if any(difflib.SequenceMatcher(None, t.lower(), v).ratio() >= 0.83 or t.lower() == v for v in vague_replies):
                if not any(len(word) >= 3 for word in tokens):
                    return True
        return False

    if is_vague(msg):
        return True

    # Question word check
    if lower in question_words:
        return True

    # Demonstrative + noun pair
    def check_demonstrative_noun_pair(lower):
        demonstratives = {"that", "this", "those", "these"}
        toks = lower.split()
        for i in range(len(toks) - 1):
            if toks[i] in demonstratives and toks[i + 1] in nouns:
                return True
        return False

    # Clarifying noun after pronouns with special rule for "it's"/"its"
    def clarify_noun_check(tokens, pronouns, nouns):
        pronoun_set = set(pronouns.keys())
        clarifying_nouns = nouns - pronoun_set
        for i, token in enumerate(tokens):
            token_lower = token.lower()
            if token_lower in {"it's", "its"}:
                # Check if "its" is followed by an allowed word
                if token_lower == "its" and i + 1 < len(tokens) and tokens[i + 1].lower() in allowed_after_its:
                    continue  # "its" followed by allowed word is okay
                # For "it's", we generally expect aclarifying noun in the context
                if token_lower == "it's":
                    context = [tok.lower() for j, tok in enumerate(tokens) if j != i]  # Changed 'word' to 'tok'
                    if not any(difflib.SequenceMatcher(None, tok, cn).ratio() >= 0.83 or tok == cn for cn in clarifying_nouns):  # Changed 'word' to 'tok'
                        return True
            elif token_lower in pronoun_set:
                context = [tok.lower() for j, tok in enumerate(tokens) if j != i]  # Changed 'word' to 'tok'
                if not any(difflib.SequenceMatcher(None, tok, cn).ratio() >= 0.83 or tok == cn for cn in clarifying_nouns):  # Changed 'word' to 'tok'
                    return True
        return False

    # Deictic word check
    for t in tokens:
        if t in deictic_words:
            return True

    # New regex check for messages starting with specific words
    if re.match(r"^(it|its|it's|this|that|they|these|those)\b", lower):
        return True

    # Additions for phrases: "in the chat", "as we were saying", "earlier", "someone said", "you guys"
    if re.search(r"in the chat", lower):
        return True
    if re.search(r"as we were saying", lower):
        return True
    if re.search(r"\bearlier\b", lower):
        return True
    if re.search(r"someone said", lower):
        return True
    if re.search(r"you guys", lower):
        return True

    return False

def strip_names(text, all_names):
    tokens = re.findall(r"\b\w+'\w+|\w+\b", text)
    cleaned_tokens = [token for token in tokens if token not in all_names]
    return " ".join(cleaned_tokens)

def strip_name_phrases_context(text, prepositions):

    def is_likely_name_phrase(phrase):
        words = phrase.split()
        for word in words:
            if is_name(word.strip(",")):
                return True
        return False

    prep_pattern = "|".join(re.escape(prep) for prep in prepositions)
    pattern = rf"\s+(?:{prep_pattern})\s+((?:\w+\s+and\s+)?\w+)"
    matches = re.finditer(pattern, text, re.IGNORECASE)

    new_text = text
    offset = 0

    for match in matches:
        phrase = match.group(1)
        if is_likely_name_phrase(phrase):
            start = match.start(0) + offset
            end = match.end(0) + offset
            new_text = new_text[:start] + "" + new_text[end:]
            offset -= (end - start)

    return new_text.strip()

import re

def clean_line(line):
    line = line.replace('\\o', '\\\\o')
    line = bytes(line, "utf-8").decode("unicode_escape")
    line = re.sub(r'\[[^\]]+\]https?://\S*', '', line)
    line = re.sub(r'\[\s*\]\([^\)]+\)', '', line)
    line = re.sub(r'\[([^\]]+)\]\([^\)]+\)', '', line)
    line = re.sub(r'\[[^\]]*[\~\^\|\`\*][^\]]*\]\(?', '', line)
    line = re.sub(r'https://\S*', '', line)
    line = re.sub(r'^<@&?\d+>\s*', '', line)
    line = re.sub(r'<@&?\d+>', '', line)
    line = re.sub(r'<a?:(\w+):\d+>', '', line)
    line = re.sub(r'\{[^{}]+\}', '', line)
    line = re.sub(r'<#\d+>', '', line)
    line = re.sub(r'```', '', line)           # <-- stripping triple backticks here
    line = line.replace('"', '').replace('\\', '')
    line = re.sub(r'[^\x00-\x7F]+', '', line)
    line = re.sub(r'^:', '', line)
    line = re.sub(r'^[,.;:!#\$%]', '', line)
    line = re.sub(r'^>.*', '', line)
    line = re.sub(r'^<.*', '', line)
    line = line.replace('\n', '').replace('\r', '')
    line = re.sub(r'\bhttps?://\S+', '', line)
    line = re.sub(r'^-', '', line)
    #line = re.sub(r'\bGO\w*', ',', line)
    line = re.sub(r'\s+,', ',', line)
    line = re.sub(r'\s+\.(?!\S)', '.', line)
    line = re.sub(r'\*\*(.*?)\*\*', r'\1', line)
    line = line.strip()
    line = strip_name_phrases_context(line, prepositions)
    if any(name in line for name in all_names):
        return ''
    if not re.search(r'[a-zA-Z]', line):
        return ''
    line = transform_gender(line)
    return line

import re

def transform_gender(text):
    replacements = {
        r"\bi\s+am\s+a\s+boy\b": "i am a girl",
        r"\bi\'m\s+a\s+boy\b": "i'm a girl",
        r"\bi\s+am\s+a\s+dad\b": "i am a mom",
        r"\bi\'m\s+a\s+dad\b": "i'm a mom",
        r"\bas\s+a\s+boy\b": lambda m: "as a girl" if re.search(r"\b(i|my|mine)\b", text[:m.start()], re.IGNORECASE) else "as a boy",
        r"\bwhen\s+i\s+was\s+a\s+boy\b": "when i was a girl",
        r"\bhe\b": lambda m: "she" if re.search(r"\b(i|my|mine|me)\b", text[:m.start()], re.IGNORECASE) else "he",
        r"\bhim\b": lambda m: "her" if re.search(r"\b(i|my|mine|me)\b", text[:m.start()], re.IGNORECASE) else "him",
        r"\bhis\b": lambda m: "her" if re.search(r"\b(i|my|mine|me)\b", text[:m.start()], re.IGNORECASE) else "his",
        r"\bhe\'s\b": lambda m: "she's" if re.search(r"\b(i|my|mine|me)\b", text[:m.start()], re.IGNORECASE) else "he's",
        r"\bhe’d\b": lambda m: "she’d" if re.search(r"\b(i|my|mine|me)\b", text[:m.start()], re.IGNORECASE) else "he’d",
        r"\bhe’ll\b": lambda m: "she’ll"if re.search(r"\b(i|my|mine|me)\b", text[:m.start()], re.IGNORECASE) else "he’ll",
        r"\bhe’ve\b": lambda m: "she’ve" if re.search(r"\b(i|my|mine|me)\b", text[:m.start()], re.IGNORECASE) else "he’ve",
        r"\bhe’d’ve\b":lambda m:"she’d’ve" if re.search(r"\b(i|my|mine|me)\b", text[:m.start()], re.IGNORECASE) else "he’d’ve",
        r"\bmy\s+own\s+boy\b": "my own girl",
        r"\bthe\s+boy\'s\b": lambda m: "the girl's" if re.search(r"\b(i|my|mine|me)\b", text[:m.start()], re.IGNORECASE) else "the boy's",
        r"\bboy\'s\b": lambda m: "girl's" if re.search(r"\b(i|my|mine|me)\b", text[:m.start()], re.IGNORECASE) else "boy's",
        r"\bboy\b": lambda m: "girl" if re.search(r"\b(i|my|mine|me|myself|my own)\b", text[:m.start()] + text[m.end():], re.IGNORECASE) else "boy",
        r"\bboys\b": lambda m: "girls" if re.search(r"\b(i|my|mine|me|myself|my own)\b", text[:m.start()] + text[m.end():], re.IGNORECASE) else "boys",
        r"\bboyhood\b": "girlhood",
    }
    for pattern, replacement in replacements.items():
        if isinstance(replacement, str):
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        else:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def process_line(line):
    """
    Helper function to process a single line.
    The function tags the line with <|endoftext|> or <|endoftext|> based on its content.
    It doesn't modify the target field, just the input text.
    """
    cleaned_line = clean_line(line)
    if cleaned_line:
        # Only add the tag to the input (not the target)
        label = "<|endoftext|>" if not is_bad_message(cleaned_line) else "<|endoftext|>"  # Changed tags here
        return {"input": cleaned_line, "output": label}
    return None

def process_file(input_file, output_file_tagged, output_file_good_only, max_workers=4):
    """
    Processes the input file, tags messages, and writes to output files.
    """
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    total_lines = len(lines)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        process_line_partial = partial(process_line)
        results = list(tqdm(executor.map(process_line_partial, lines), total=total_lines, desc="Processing lines"))

    records = [record for record in results if record is not None]

    tagged_lines = []
    good_only_lines = []

    for entry in records:
        text = entry["input"].strip()
        label = entry["output"]

        # Add the tag and EOS token at the end of each line
        tagged_lines.append(f"{label} {text}<|endoftext|>")

        if label == "<|endoftext|>":
            good_only_lines.append(f"<|endoftext|> {text}<|endoftext|>\n")

    # Sort the lines so that bad lines come last
    tagged_lines.sort(key=lambda x: x.startswith("<BAD>"))

    with open(output_file_tagged, "w", encoding="utf-8") as f_out:
        f_out.write("\n".join(tagged_lines))

    with open(output_file_good_only, "w", encoding="utf-8") as f_good_only:
        f_good_only.writelines(good_only_lines)

    print(f"Saved {len(tagged_lines)} lines to {output_file_tagged}")
    print(f"Saved {len(good_only_lines)} good lines to {output_file_good_only}")

if __name__ == "__main__":
    if len(sys.argv) == 1:  # No arguments provided
        input_file = input("Please enter file name for input (without .txt): ")
         # Add .txt extension if it's not already there
        if not input_file.endswith(".txt"):
            input_file += ".txt"
        output_file_tagged = "all.txt"
        output_file_good_only = "good.txt"

        # Adjust the number of workers based on your CPU cores
        max_workers = 12
        process_file(input_file, output_file_tagged, output_file_good_only, max_workers)
    else:
        parser = argparse.ArgumentParser(description="Process and filter text data.")
        parser.add_argument("-p", "--input_file", help="Input text file path")
        parser.add_argument("-a", "--output_tagged", default="all.txt", help="Output file for tagged lines")
        parser.add_argument("-g", "--output_good", default="good.txt", help="Output file for 'good' lines only")
        parser.add_argument("-w", "--workers", type=int, default=12, help="Number of worker threads")

        args = parser.parse_args()
        process_file(args.input_file, args.output_tagged, args.output_good, args.workers)
    #os.system("python oversample.py")  # Keep these as separate calls
    #os.system("python split.py")