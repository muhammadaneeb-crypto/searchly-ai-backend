"""Lightweight language detection, including Roman Urdu (Urdu typed in Latin letters)."""
import re
from dataclasses import dataclass

from langdetect import DetectorFactory, detect_langs
from langdetect.lang_detect_exception import LangDetectException

DetectorFactory.seed = 0

LANG_NAMES = {
    "en": "English", "ur": "Urdu", "ur-Latn": "Roman Urdu (Urdu written in English letters)", "ar": "Arabic",
    "fa": "Persian", "zh": "Chinese", "zh-cn": "Chinese (Simplified)", "zh-tw": "Chinese (Traditional)",
    "es": "Spanish", "fr": "French", "de": "German", "hi": "Hindi", "pt": "Portuguese", "ru": "Russian",
    "tr": "Turkish", "it": "Italian", "ja": "Japanese", "ko": "Korean", "id": "Indonesian", "bn": "Bengali",
}
RTL_LANGS = {"ar", "ur", "fa", "he", "ps", "sd"}
URDU_ONLY_CHARS = set("ٹڈڑںےگچپھۓ")
ROMAN_URDU_WORDS = {
    "kya", "hai", "hain", "mujhe", "mujhay", "kaise", "kese", "kaisay", "kaun", "kon", "nahi", "nahin", "aap", "ap",
    "mein", "main", "ka", "ki", "ko", "ye", "yeh", "woh", "kar", "karo", "karna", "karen", "batao", "bataen", "batain",
    "kahan", "kab", "kyun", "kyu", "acha", "accha", "theek", "hota", "hoti", "hoga", "liye", "wala", "wali", "samjhao",
    "samjha", "chahiye", "chahie", "tum", "hum", "mera", "meri", "tera", "uska", "iska", "kitna", "kitni", "abhi",
    "bohat", "bahut", "zaroor", "shukriya", "salam", "assalam", "paisa", "paise", "kuch", "sab", "aur", "lekin",
}


@dataclass
class LanguageInfo:
    code: str
    name: str
    rtl: bool


def _info(code: str) -> LanguageInfo:
    base = code.split("-")[0]
    return LanguageInfo(code, LANG_NAMES.get(code, LANG_NAMES.get(base, code)), base in RTL_LANGS)


def language_from_code(code: str | None) -> LanguageInfo | None:
    """Turn a user-requested language ('Urdu', 'ur', 'French') into LanguageInfo."""
    if not code or not code.strip():
        return None
    c = code.strip()
    for k, v in LANG_NAMES.items():
        if c.lower() in (k.lower(), v.lower(), v.split(" (")[0].lower()):
            return _info(k)
    return LanguageInfo(c, c, False)  # free text such as "Punjabi" is passed to the model as-is


def detect_language(text: str) -> LanguageInfo:
    text = (text or "").strip()
    if not text:
        return _info("en")
    if re.search(r"[\u4e00-\u9fff]", text):
        if re.search(r"[\u3040-\u30ff]", text):
            return _info("ja")
        return _info("zh")
    if re.search(r"[\uac00-\ud7af]", text):
        return _info("ko")
    if re.search(r"[\u0600-\u06ff]", text):
        if any(ch in URDU_ONLY_CHARS for ch in text):
            return _info("ur")
        try:
            top = detect_langs(text)[0]
            if top.lang in ("ar", "fa", "ur"):
                return _info(top.lang)
        except LangDetectException:
            pass
        return _info("ar")
    tokens = re.findall(r"[a-zA-Z']+", text.lower())
    hits = sum(1 for t in tokens if t in ROMAN_URDU_WORDS)
    if tokens and (hits >= 2 or (hits >= 1 and len(tokens) <= 3)):
        return _info("ur-Latn")
    try:
        top = detect_langs(text)[0]
        if top.prob >= 0.6 and len(tokens) >= 3:
            return _info(top.lang)
    except LangDetectException:
        pass
    return _info("en")
