import pandas as pd
import numpy as np
import streamlit as st
import json
import re
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer, util

st.title("Transcript Scoring System")


user_input = st.text_area("Paste your transcript below:")

duration_in_sec = 52



salutation_patterns = [
    {"phrases": ["Hii", "Hello"], "score": 2},
    {"phrases": ["Good Morning", "Good Afternoon", "Good Evening", "Good Day", "Hello everyone"], "score": 4},
    {"phrases": ["I am excited to introduce", "I am feeling great"], "score": 5}
]

def get_salutation_score(text):
    text = text.lower()
    matches = []
    best_score = 0

    for pattern in salutation_patterns:
        for phrase in pattern["phrases"]:
            p = phrase.lower()
            if p in text:
                best_score = max(best_score, pattern["score"])
                matches.append(p)

    return best_score, matches


def key_word_presence(text):
    text = text.lower()
    must_have_keywords = {
        "name": ["my name is", "myself", "i am", "this is"],
        "age": ["years old"],
        "school": ["school", "class", "studying in"],
        "family": ["family", "mother", "father", "parents", "brother", "sister"],
        "hobby": ["i like", "i enjoy", "my hobby", "playing", "my interest"],
    }

    found = []
    score = 0

    for key, phrases in must_have_keywords.items():
        for p in phrases:
            if p in text:
                found.append(p)
                score += 4
                break

    return score, found


def good_to_have(text):
    text = text.lower()
    gth_keywords = {
        "origin_location": ["i am from", "i'm from", "my parents are from"],
        "goal": ["goal", "ambition", "dream", "aim", "aspire"],
        "family": ["family", "mother", "father", "parents", "brother", "sister"],
        "achievement": ["achievement", "achievements", "strength", "strengths", "award", "accomplishment"],
        "fun_fact": ["fun fact", "interesting thing", "something unique", "special"]
    }

    found = []
    score = 0

    for key, phrases in gth_keywords.items():
        for p in phrases:
            if p in text:
                found.append(p)
                score += 2
                break

    return score, found


def check_order(txt):

    txt = txt.lower()

    salutation_words = ["hi", "hello", "good morning", "good afternoon", "good evening", "good day", "hello everyone", "i am excited to introduce", "i am feeling great"]
    basic_words = ["my name is", "myself", "i am", "this is", "years old", "class", "school", "studying in", "i am from", "my parents are from"]
    additional_words = ["hobby", "hobbies", "i enjoy", "fun fact", "unique", "strength", "achievement", "special"]
    closing_words = ["thank you", "thanks"]

    def first_index(words):
        positions = [txt.find(w) for w in words if txt.find(w) != -1]
        return min(positions) if positions else None

    sal_idx = first_index(salutation_words)
    basic_idx = first_index(basic_words)
    closing_idx = first_index(closing_words)

    if sal_idx is None or basic_idx is None or closing_idx is None:
        return "Order Not followed"

    return "Order followed" if sal_idx < basic_idx < closing_idx else "Order Not followed"


def speech_rate(text):
    if duration_in_sec == 0:
        return 0

    words = len(text.split(" "))
    wpm = words / (duration_in_sec / 60)

    if wpm > 161:
        return 2
    elif 141 <= wpm <= 160:
        return 6
    elif 111 <= wpm <= 140:
        return 10
    elif 81 <= wpm <= 110:
        return 6
    else:
        return 2


def grammar_score(text):
    words = text.split()
    if len(words) == 0:
        return 0

    corrected = TextBlob(text).correct()
    corrected_words = corrected.split()

    errors = sum(1 for w1, w2 in zip(words, corrected_words) if w1.lower() != w2.lower())
    errors_per_100 = (errors / len(words)) * 100
    score_ratio = 1 - min(errors_per_100 / 10, 1)

    if score_ratio > 0.9:
        return 10
    elif score_ratio > 0.7:
        return 8
    elif score_ratio > 0.5:
        return 6
    elif score_ratio > 0.3:
        return 4
    else:
        return 2


def calculate_ttr(text):
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", "", text.lower())
    words = cleaned.split()

    if len(words) == 0:
        return 0

    ttr = len(set(words)) / len(words)

    if ttr >= 0.9:
        return 10
    elif ttr >= 0.7:
        return 8
    elif ttr >= 0.5:
        return 6
    elif ttr >= 0.3:
        return 4
    return 2


def clarity(txt):
    filler_words = ["um", "uh", "like", "you know", "so", "actually", "basically",
                    "right", "i mean", "well", "kinda", "kind of", "sort of",
                    "okay", "hmm", "ah"]

    total_words = len(txt.split(" "))
    count = sum(txt.count(w) for w in filler_words)

    if total_words == 0:
        return 0

    rate = (count / total_words) * 100

    if 0 < rate <= 3:
        return 15
    elif rate <= 6:
        return 12
    elif rate <= 9:
        return 9
    elif rate <= 12:
        return 6
    else:
        return 3


def sentiment_score(text):
    scores = SentimentIntensityAnalyzer().polarity_scores(text)["compound"]

    if scores >= 0.9:
        return 15
    elif scores >= 0.7:
        return 12
    elif scores >= 0.5:
        return 9
    elif scores >= 0.3:
        return 6
    else:
        return 3



if st.button("Score"):

    if user_input.strip() == "":
        st.warning("Please enter a transcript.")
        st.stop()

    # RULE BASED
    sal_score, sal_kw = get_salutation_score(user_input)
    must_score, must_kw = key_word_presence(user_input)
    gth_score, gth_kw = good_to_have(user_input)
    grammar = grammar_score(user_input)
    ttr = calculate_ttr(user_input)
    speech = speech_rate(user_input)
    clarity_score_val = clarity(user_input)
    sentiment_val = sentiment_score(user_input)

    rule_based_total = sal_score + must_score + gth_score + grammar + ttr + speech + clarity_score_val + sentiment_val

    # NLP SIMILARITY
    model = SentenceTransformer("all-MiniLM-L6-v2")

    emb_script = model.encode(user_input, convert_to_tensor=True)
    emb_sal = model.encode(" ".join(sal_kw), convert_to_tensor=True)
    emb_must = model.encode(" ".join(must_kw), convert_to_tensor=True)
    emb_gth = model.encode(" ".join(gth_kw), convert_to_tensor=True)

    sim_sal = util.cos_sim(emb_sal, emb_script).item() if len(sal_kw) else 0
    sim_must = util.cos_sim(emb_must, emb_script).item() if len(must_kw) else 0
    sim_gth = util.cos_sim(emb_gth, emb_script).item() if len(gth_kw) else 0

    def sim_to_score(sim, max_s):
        return max(sim, 0) * max_s

    sal_nlp = sim_to_score(sim_sal, 5)
    must_nlp = sim_to_score(sim_must, 20)
    gth_nlp = sim_to_score(sim_gth, 10)

    sal_combined = sal_score + sal_nlp
    must_combined = must_score + must_nlp
    gth_combined = gth_score + gth_nlp

    total_max = 5 + 30 + 15
    overall_score = ((sal_combined + must_combined + gth_combined) / total_max) * 100


    detailed_output = {
        "overall_score": round(overall_score, 2),
        "criteria_scores": {
            "salutation": {"rule_score": sal_score, "nlp_similarity": round(sim_sal, 2),
                           "combined_score": round(sal_combined, 2), "keywords_found": sal_kw},
            "must_have": {"rule_score": must_score, "nlp_similarity": round(sim_must, 2),
                          "combined_score": round(must_combined, 2), "keywords_found": must_kw},
            "good_to_have": {"rule_score": gth_score, "nlp_similarity": round(sim_gth, 2),
                             "combined_score": round(gth_combined, 2), "keywords_found": gth_kw},
            "grammar": grammar,
            "ttr": ttr,
            "speech_rate": speech
        }
    }

    st.subheader("Detailed Output")
    st.json(detailed_output)

    json_data = json.dumps(detailed_output, indent=4)

    st.download_button(
        label="📥 Download Detailed Score (JSON)",
        data=json_data,
        file_name="transcript_score.json",
        mime="application/json"
    )


    st.subheader("Score Summary")

    nlp_total = sim_sal + sim_must + sim_gth

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div style="background:pink;padding:18px;border-radius:15px;text-align:center;
        box-shadow:0 0 8px rgba(0,0,0,0.15);">
            <h3>Overall Score</h3>
            <h2 style='color:#0d47a1;'>{round(overall_score,2)}%</h2>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style="background:white;padding:18px;border-radius:15px;text-align:center;
        box-shadow:0 0 8px rgba(0,0,0,0.15);">
            <h3 style="color:red">Rule-Based Score</h3>
            <h2 style='color:#1b5e20;'>{rule_based_total}</h2>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div style="background:hotpink;padding:18px;border-radius:15px;text-align:center;
        box-shadow:0 0 8px rgba(0,0,0,0.15);">
            <h3>NLP Similarity Boost</h3>
            <h2 style='color:#e65100;'>{round(nlp_total,2)}</h2>
        </div>
        """, unsafe_allow_html=True)
