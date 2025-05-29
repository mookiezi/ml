import re
import difflib
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import random
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import names
from fuzzywuzzy import fuzz
from concurrent.futures import ThreadPoolExecutor
from functools import partial
# import keyboard
from pynput import keyboard
import time
import sys
import argparse  # Import the argparse module

TYPOS = False
FILTER_ENABLED = True  # Set to False to disable the message filter

nltk.download('names')
nltk.download('wordnet')

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
    "shes", "she", "ubuntu",  "unix", "i'm", "im"
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
    return word.lower() in verbs

def is_fuzzy_noun(word, nouns):
    return word.lower() in nouns

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
        toks = message.lower().split() # Lowercase for check
        if toks and toks[0] == "i":
            modal_verbs = {"can", "could", "might", "may", "will", "would", "should", "did", "do", "was", "am", "have", "had"}
            if len(toks) > 1 and toks[1] in modal_verbs:
                has_verb = any(t in verbs for t in toks[2:])
                return not has_verb
            return True
        return False

    if check_i_modal_verbs(msg, verbs):
        return True

    # Vague reply check
    def is_vague(msg):
        tokens = msg.strip().split()
        if len(tokens) <= 4:
            if any(t.lower() in vague_replies for t in tokens): # Lowercase for check
                if not any(len(word) >= 3 for word in tokens):
                    return True
        return False

    if is_vague(msg):
        return True

    # Question word check
    if lower in question_words: # Keep using lower here, question_words are all lowercase
        return True

    # Demonstrative + noun pair
    def check_demonstrative_noun_pair(lower): # Keep lower here
        demonstratives = {"that", "this", "those", "these"}
        toks = lower.split()
        for i in range(len(toks) - 1):
            if toks[i] in demonstratives and toks[i + 1] in nouns:
                return True
        return False

    # Clarifying noun after pronouns with special rule for "it's"/"its"
    def clarify_noun_check(tokens, pronouns, nouns):
        # Convert the pronouns dictionary keys to a set
        pronoun_set = set(pronouns.keys())
        clarifying_nouns = nouns - pronoun_set
        for i, token in enumerate(tokens):
            token_lower = token.lower()

            # Special case: allow "it's" or "its" if followed by "getting"
            if token_lower in {"it's", "its"}:
                if i + 1 < len(tokens) and tokens[i + 1].lower() in allowed_after_its:
                    return False  # Valid usage, no need to flag
            if token_lower in pronoun_set:
                context = [word.lower() for j, word in enumerate(tokens) if j != i]
                if not any(word in clarifying_nouns for word in context):
                    return True  # Invalid: pronoun lacks noun anchor
        return False  # All clear!

    if clarify_noun_check(tokens, pronouns, nouns):
        return True

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
    if re.search(r"\bearlier\b", lower):  # \b for word boundary
        return True
    if re.search(r"someone said", lower):
        return True
    if re.search(r"you guys", lower):
        return True

    return False

def strip_names(text, all_names):
    tokens = re.findall(r"\b\w+'\w+|\w+\b", text)  # Tokenize the text (keep original case)
    cleaned_tokens = [token for token in tokens if token not in all_names]  # Filter out names (case-sensitive)
    return " ".join(cleaned_tokens)  # Rejoin the tokens

def strip_name_phrases_context(text, prepositions):

    def is_likely_name_phrase(phrase):
        words = phrase.split()
        for word in words:
            if is_name(word.strip(",")):  # Check for names, removing commas (case-sensitive)
                return True
        return False

    prep_pattern = "|".join(re.escape(prep) for prep in prepositions)
    pattern = rf"\s+(?:{prep_pattern})\s+((?:\w+\s+and\s+)?\w+)"  # Capture the name phrase
    matches = re.finditer(pattern, text, re.IGNORECASE)

    new_text = text
    offset = 0  # To adjust for changes in string length

    for match in matches:
        phrase = match.group(1)  # The captured name phrase
        if is_likely_name_phrase(phrase):
            start = match.start(0) + offset
            end = match.end(0) + offset
            new_text = new_text[:start] + "" + new_text[end:]
            offset -= (end - start)  # Adjust offset

    return new_text.strip()

import re

def clean_line(line):
    # Replace the specific escaped sequence '\o' with '\\o' to handle potential encoding issues.
    line = line.replace('\\o', '\\\\o')
    # Decode the string, interpreting escaped Unicode characters (e.g., '\u00a0' for non-breaking space).
    line = bytes(line, "utf-8").decode("unicode_escape")
    try:
        line = bytes(line, "utf-8").decode("unicode_escape")
    except UnicodeDecodeError:
        # fallback: just keep the original line or remove trailing backslashes
        line = line.rstrip("\\")
    # Remove markdown-style links where the text is followed immediately by http(s)://...
    line = re.sub(r'\[[^\]]+\]https?://\S*', '', line)
    # Remove markdown-style links with empty link text: [](...)
    line = re.sub(r'\[\s*\]\([^\)]+\)', '', line)
    # Remove standard markdown-style links with link text and a URL: [link text](url)
    line = re.sub(r'\[([^\]]+)\]\([^\)]+\)', '', line)
    # Remove markdown-style links where the link text contains formatting characters (~^|`*): [formatted text](url) or [formatted text](
    line = re.sub(r'\[[^\]]*[\~\^\|\`\*][^\]]*\]\(?', '', line)
    # Remove standalone URLs starting with http:// or https://
    line = re.sub(r'https://\S*', '', line)
    # Replace Discord mention that might have a role ID with "sure" at the beginning of the line.
    line = re.sub(r'^<@&?\d+>\s*', 'sure', line)
    # Remove Discord mentions (user or role IDs).
    line = re.sub(r'<@&?\d+>', '', line)
    # Remove Discord custom emojis with their IDs: <a?:name:id>
    line = re.sub(r'<a?:(\w+):\d+>', '', line)
    # Remove curly brace enclosed expressions: {something}
    line = re.sub(r'\{[^{}]+\}', '', line)
    # Remove Discord channel mentions: <#id>
    line = re.sub(r'<#\d+>', '', line)
    # Remove double quotes and backslashes.
    line = line.replace('"', '').replace('\\', '')
    # Remove any characters that are not within the basic ASCII range (0-127).
    line = re.sub(r'[^\x00-\x7F]+', '', line)
    # Remove leading colons.
    line = re.sub(r'^:', '', line)
    # Remove leading punctuation marks (comma, period, semicolon, colon, exclamation mark, hash, dollar sign, percent).
    line = re.sub(r'^[,.;:!#\$%]', '', line)
    # Remove lines that start with a greater-than sign (often used for quotes).
    line = re.sub(r'^>.*', '', line)
    # Remove lines that start with a less-than sign (might be incomplete HTML or other tags).
    line = re.sub(r'^<.*', '', line)
    # Remove newline and carriage return characters.
    line = line.replace('\n', '').replace('\r', '')

    # Remove any stray URLs that might remain after the other checks.
    line = re.sub(r'\bhttps?://\S+', '', line)

    # Replace words starting with capital "GO" with a comma
    line = re.sub(r'\bGO\w*', ',', line)

    # Remove spaces before commas
    line = re.sub(r'\s+,', ',', line)

    # Strip "ubuntu"
    line = re.sub(r'\bubuntu\b', '', line, flags=re.IGNORECASE)

    # Strip I'm
    # line = re.sub(r"\b(i['’]?m)\b", '', line, flags=re.IGNORECASE)

    # Remove all underscores
    line = re.sub(r"_", "", line)

    # Remove extra spaces from the sentence
    line = re.sub(r'\s+', ' ', line).strip()



    # Remove leading and trailing whitespace from the line.
    line = line.strip()

    # Assuming 'strip_name_phrases_context' is a function defined elsewhere that removes name-related phrases based on a list of prepositions.
    # line = strip_name_phrases_context(line, prepositions)
    # Assuming 'strip_names' is a function defined elsewhere that removes names based on a list of all names.
    # line = strip_names(line, all_names)

    # Check if the line contains at least one alphabetic character. If not, return an empty string.
    if not re.search(r'[a-zA-Z]', line):
        return ''

    # Apply gender transformation here
    line = transform_gender(line) # Added gender transformation

    # Return the cleaned line.
    return line

import re

def transform_gender(text):
    # Define words and their feminine replacements, now with improved context awareness
    replacements = {
        # Self-referential uses of "I am a boy" or "I'm a boy"
        r"\bi\s+am\s+a\s+boy\b": "i am a girl",
        r"\bi\'m\s+a\s+boy\b": "i'm a girl",
        r"\bi\s+am\s+a\s+dad\b": "i am a mom",
        r"\bi\'m\s+a\s+dad\b": "i'm a mom",

        # "as a boy" only when referring to the speaker
        r"\bas\s+a\s+boy\b": lambda m: "as a girl" if re.search(r"\b(i|my|mine)\b", text[:m.start()], re.IGNORECASE) else "as a boy",

        # "when I was a boy"
        r"\bwhen\s+i\s+was\s+a\s+boy\b": "when i was a girl",

        # Pronoun replacements, context-aware
        r"\bhe\b": lambda m: "she" if re.search(r"\b(i|my|mine|me)\b", text[:m.start()], re.IGNORECASE) else "he",
        r"\bhim\b": lambda m: "her" if re.search(r"\b(i|my|mine|me)\b", text[:m.start()], re.IGNORECASE) else "him",
        r"\bhis\b": lambda m: "her" if re.search(r"\b(i|my|mine|me)\b", text[:m.start()], re.IGNORECASE) else "his",
        r"\bhe\'s\b": lambda m: "she's" if re.search(r"\b(i|my|mine|me)\b", text[:m.start()], re.IGNORECASE) else "he's",
        r"\bhe’d\b": lambda m: "she’d" if re.search(r"\b(i|my|mine|me)\b", text[:m.start()], re.IGNORECASE) else "he’d",
        r"\bhe’ll\b": lambda m: "she’ll"if re.search(r"\b(i|my|mine|me)\b", text[:m.start()], re.IGNORECASE) else "he’ll",
        r"\bhe’ve\b": lambda m: "she’ve" if re.search(r"\b(i|my|mine|me)\b", text[:m.start()], re.IGNORECASE) else "he’ve",
        r"\bhe’d’ve\b":lambda m:"she’d’ve" if re.search(r"\b(i|my|mine|me)\b", text[:m.start()], re.IGNORECASE) else "he’d’ve",

        # "my own boy"
        r"\bmy\s+own\s+boy\b": "my own girl",

        # "the boy's" and "boy's" only when possessive of self
        r"\bthe\s+boy\'s\b": lambda m: "the girl's" if re.search(r"\b(i|my|mine|me)\b", text[:m.start()], re.IGNORECASE) else "the boy's",
        r"\bboy\'s\b": lambda m: "girl's" if re.search(r"\b(i|my|mine|me)\b", text[:m.start()], re.IGNORECASE) else "boy's",

       #  "boy" and "boys" only when used as self-reference or direct possessive
        r"\bboy\b": lambda m: "girl" if re.search(r"\b(i|my|mine|me|myself|my own)\b", text[:m.start()] + text[m.end():], re.IGNORECASE) else "boy",
        r"\bboys\b": lambda m: "girls" if re.search(r"\b(i|my|mine|me|myself|my own)\b", text[:m.start()] + text[m.end():], re.IGNORECASE) else "boys",

        r"\bboyhood\b": "girlhood",
    }

    # Apply the replacements
    for pattern, replacement in replacements.items():
        if isinstance(replacement, str):
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        else:  # It's a function
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text



previous_outputs = set()  # Initialize previous_outputs

# Typo introducing function
def introduce_typo(text, fuzz_cutoff=80):
    """
    Introduces a typo into a single randomly selected word within the given text.

    Args:
        text (str): The input text string.
        fuzz_cutoff (int, optional): The fuzziness threshold for typo creation.
            Defaults to 80.  This parameter is not currently used.

    Returns:
        str: The text with a typo in one word, or the original text if it
             contains less than 3 words.
    """
    words = text.split()
    if len(words) < 3:
        return text  # Don't introduce typos in short texts

    word_index_to_modify = random.randint(0, len(words) - 1)
    word_to_modify = words[word_index_to_modify]

    # List of possible typo patterns
    possible_typos = [
        lambda x: x[:-1] if len(x) > 1 else x,  # Drop last character
        lambda x: x + random.choice("abcdefghijklmnopqrstuvwxyz"),  # Add random character at the end
        lambda x: x[:-2] + random.choice("aeiou") + x[-1] if len(x) > 2 else x,  # Replace last vowel
        lambda x: x[:random.randint(0, len(x)-1)] + random.choice("abcdefghijklmnopqrstuvwxyz") + x[random.randint(0, len(x)-1):] if len(x) > 1 else x,  # Replace random letter
    ]

    typo_func = random.choice(possible_typos)
    modified_word = typo_func(word_to_modify)
    words[word_index_to_modify] = modified_word
    return " ".join(words)



while True:
    user_model_path = input("Enter path to trained model directory (press Enter to use 'model'): ").strip()
    model_dir = user_model_path if user_model_path else "model"

    if os.path.isdir(model_dir):
        print(f"Model directory '{model_dir}' found.")
        break
    print(f"Directory '{model_dir}' does not exist. Try again.")

model_dir = os.path.abspath(model_dir)
print(f"Final model directory: {model_dir}")
if not os.path.isdir(model_dir):
    print("Not a valid folder! Exiting.")
    exit(1)

while True:
    log_name = input("Enter the name of the log file (without '.txt'): ").strip()
    log_file_path = log_name + ".txt"

    if not os.path.isfile(log_file_path):
        with open(log_file_path, 'w'):
            pass
        print(f"Log file '{log_file_path}' created.")
        break
    else:
        
        print(f"Log file '{log_file_path}' already exists.")

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
tokenizer.pad_token = tokenizer.eos_token

previous_outputs = set()

def log_output(text):
    with open(log_file_path, "a", encoding="utf-8") as f:
        f.write(text.strip() + "\n")

def generate_text(msg, max_attempts=20):
    if not msg.strip():
        msg = " "

    attempt = 0
    while attempt < max_attempts:
        inputs = tokenizer(
            msg,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=16
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        min_length = random.randint(5, 30)
        max_length = random.randint(50, 150)

        '''
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=True,
            min_length=min_length,
            max_length=max_length,
            top_k=40,
            top_p=0.85,
            temperature=0.7,
            no_repeat_ngram_size=2,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        '''

        '''  
        output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        do_sample=True,
        min_length=5,  # Adjust based on your desired output length
        max_length=20,  # Adjust based on your desired output length
        top_k=10,  # Reduce randomness
        top_p=0.8,  # Adjust to focus on more probable tokens
        temperature=0.5,  # Make output more deterministic
        no_repeat_ngram_size=2,  # Prevent repetitive phrases
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
        )
        '''
        

        '''
        output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        do_sample=True,
        min_length=30,  # longer minimum to ensure substance
        max_length=150,  # allow a full thought
        top_k=50,        # increase possible next tokens
        top_p=0.95,      # allow more diversity
        temperature=0.9, # add creative variation
        no_repeat_ngram_size=3,  # increase repetition blocking
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
        )
        '''

        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=True,
            min_length=20,
            max_length=80,
            top_k=30,
            top_p=0.9,
            temperature=0.75,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        cleaned = output_text.replace("\n", " ")

        if TYPOS:
            cleaned = introduce_typo(cleaned)

        cleaned = re.sub(r"<.*?>", "", cleaned).strip()
        cleaned = re.sub(r"#\w+-\w+-\w+ \d{4}-\d{2}-\d{2}", "", cleaned).strip()

        cleaned = clean_line(cleaned)
        cleaned = strip_name_phrases_context(cleaned, prepositions)
        cleaned = strip_names(cleaned, all_names)

        if cleaned and cleaned not in previous_outputs and (not FILTER_ENABLED or not is_bad_message(cleaned)):
            previous_outputs.add(cleaned)
            log_output(cleaned)
            print(f"{cleaned}")
            return
        else:
            attempt += 1

    print(f"No new unique output after {max_attempts} attempts.")

generating = True  # global flag

def on_press(key):
    global generating
    try:
        if key.char == ' ':
            generating = False
            print("Generation stopped.")
            return False
    except AttributeError:
        pass

if __name__ == "__main__":
    while True:
        user_input = input("Enter a prompt to start generating or 'q' to exit: ")
        if user_input.lower() == 'q':
            print("Exiting...")
            break

        generating = True
        print("Generation started. Press 'Space' to stop.")

        with keyboard.Listener(on_press=on_press) as listener:
            while generating:
                generate_text(user_input if user_input.strip() else "<GOOD>")
                time.sleep(0.01)

            listener.join()

#if __name__ == "__main__":
#    generating = False
#    while True:
#        user_input = input("Enter a prompt to start generating or 'q' to exit: ")
#        if user_input.lower() == 'q':
#            print("Exiting...")
#            break
#
#        if not generating:
#            generating = True
#            print("Generation started. Press 'Space' to stop.")
#
#        while generating:
#            generate_text(user_input if user_input.strip() else "<GOOD>")
#            if keyboard.is_pressed('space'):
#                generating = False
#                print("Generation stopped.")
#                break
#
#            time.sleep(0.01)
