"""API FastAPI d'EduGuardIA Lang (flow cours puis QCM sur validation)."""

import json
import os
import random
import re
import uuid
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from backend.prompts import get_chat_system_prompt, get_quiz_generation_prompt, get_test_generation_prompt

load_dotenv()

HF_API_URL = "https://router.huggingface.co/v1/chat/completions"
HF_MODEL = "Qwen/Qwen3.5-9B"
HF_TIMEOUT_SECONDS = 45.0
SUPER_HINT_COST = 30
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUPPORTED_LANGUAGES = ["anglais", "espagnol", "allemand", "italien"]
BANNED_WORDS = ["pute", "encule", "enculé", "fdp", "nazi", "raciste", "suicide", "kill yourself"]

# Commentaire: stockage memoire simple des QCM generes.
GENERATED_QUIZZES: dict[str, dict[str, Any]] = {}
TEST_SESSIONS: dict[str, dict[str, Any]] = {}


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2600)
    language: str = Field(default="anglais")
    history: list[Any] = Field(default_factory=list)
    current_points: int = Field(default=0, ge=0)
    request_super_hint: bool = Field(default=False)


class ChatResponse(BaseModel):
    response: str
    points_delta: int
    super_hint_cost: int
    can_generate_quiz: bool


class QuizGenerateRequest(BaseModel):
    language: str = Field(default="anglais")
    lesson_text: str = Field(..., min_length=20, max_length=10000)
    user_question: str = Field(..., min_length=3, max_length=1200)


class QuizPayload(BaseModel):
    id: str
    question: str
    choices: list[str]


class QuizAnswerRequest(BaseModel):
    quiz_id: str
    selected: str = Field(..., min_length=1)


class QuizAnswerResponse(BaseModel):
    feedback: str
    points_gained: int
    understanding_detected: bool


class TestGenerateRequest(BaseModel):
    language: str = Field(default="anglais")
    topic: str = Field(..., min_length=3, max_length=200)
    count: int = Field(default=20, ge=20, le=20)


class TestQuestionPayload(BaseModel):
    question: str
    choices: list[str]


class TestGenerateResponse(BaseModel):
    session_id: str
    topic: str
    questions: list[TestQuestionPayload]
    generation_source: str


class TestAnswerRequest(BaseModel):
    session_id: str
    question_index: int = Field(..., ge=0, le=19)
    selected: str = Field(..., min_length=1)


class TestAnswerResponse(BaseModel):
    correct: bool
    explanation: str
    points_gained: int
    score: int
    answered: int
    total: int
    finished: bool


app = FastAPI(title="EduGuardIA Lang API", version="4.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "null",
        "http://localhost",
        "http://localhost:8000",
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1",
        "http://127.0.0.1:8000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _normalize_language(value: str) -> str:
    cleaned = value.strip().lower()
    return cleaned if cleaned in SUPPORTED_LANGUAGES else "anglais"


def _extract_hf_error(raw: str) -> str:
    if not raw:
        return "Aucun detail."
    try:
        parsed = json.loads(raw)
    except ValueError:
        return raw.strip()
    if isinstance(parsed, dict) and "error" in parsed:
        return str(parsed["error"])
    return str(parsed)


def _format_history(history: list[Any]) -> str:
    if not history:
        return "Aucun historique."
    lines: list[str] = []
    for item in history[-10:]:
        if isinstance(item, dict):
            role = str(item.get("role", "user")).strip()
            content = str(item.get("content", "")).strip()
            lower = content.lower()
            if any(word in lower for word in BANNED_WORDS):
                # Commentaire: evite de reinjecter des messages toxiques qui polluent la session.
                continue
            lines.append(f"{role}: {content}")
        else:
            lines.append(str(item))
    return "\n".join(lines)


def _is_noisy(text: str) -> bool:
    lower = text.lower()
    noise = ["thinking process", "analyze the request", "constraints", "final review"]
    return not text.strip() or sum(1 for marker in noise if marker in lower) >= 2


async def _call_llm(system_prompt: str, user_prompt: str, max_tokens: int, temperature: float) -> str:
    token = os.getenv("HF_TOKEN", "").strip()
    if not token or token == "your_huggingface_token_here":
        raise RuntimeError("MISSING_TOKEN")

    payload = {
        "model": HF_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "chat_template_kwargs": {"enable_thinking": False},
        "reasoning": {"enabled": False},
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": 0.9,
        "stream": False,
    }
    headers = {"Authorization": f"Bearer {token}"}

    data: dict[str, Any] = {}
    last_timeout: httpx.TimeoutException | None = None
    for _ in range(2):
        try:
            async with httpx.AsyncClient(timeout=HF_TIMEOUT_SECONDS) as client:
                response = await client.post(HF_API_URL, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
                break
        except httpx.TimeoutException as exc:
            last_timeout = exc
            continue
    else:
        if last_timeout is not None:
            raise last_timeout
        raise httpx.TimeoutException("LLM request timed out")

    choices = data.get("choices", [])
    if choices and isinstance(choices[0], dict):
        message = choices[0].get("message", {})
        if isinstance(message, dict):
            return str(message.get("content", "")).strip()
    return ""


def _detect_learning_focus(text: str) -> str:
    lower = text.lower()
    if any(word in lower for word in ["vocab", "lexique", "mot", "champ lexical"]):
        return "vocabulaire"
    if any(word in lower for word in ["grammaire", "syntaxe", "structure"]):
        return "grammaire"
    if any(word in lower for word in ["orthographe", "hortographe", "spelling", "ecrire", "écrire"]):
        return "orthographe"
    if any(word in lower for word in ["conjugaison", "temps", "verbe", "present", "passé", "passé", "futur"]):
        return "conjugaison"
    return "general"


def _focus_user_instruction(focus: str) -> str:
    if focus == "vocabulaire":
        return (
            "Pedagogie attendue (vocabulaire):\n"
            "- Donne une liste de 10 mots/expressions lies au theme.\n"
            "- Pour chaque entree, indique: terme en langue cible -> traduction francaise -> mini exemple.\n"
            "- Classe la liste du plus frequent au plus specifique.\n"
        )
    if focus == "grammaire":
        return (
            "Pedagogie attendue (grammaire):\n"
            "- Explique la regle principale et 2 erreurs frequentes.\n"
            "- Donne au moins 4 exemples progressifs (simple -> plus complexe).\n"
            "- Ajoute une mini methode de verification en 3 etapes.\n"
        )
    if focus == "orthographe":
        return (
            "Pedagogie attendue (orthographe):\n"
            "- Donne les regles orthographiques cle du cas demande.\n"
            "- Propose 6 mots d'entrainement avec forme correcte + piege courant.\n"
            "- Ajoute 3 micro-exercices (correction incluse).\n"
        )
    if focus == "conjugaison":
        return (
            "Pedagogie attendue (conjugaison):\n"
            "- Explique le temps verbal cible et ses usages.\n"
            "- Donne un mini tableau de 6 formes (je/tu/il-nous-vous-ils ou equivalent cible).\n"
            "- Ajoute 4 phrases exemples avec ce temps.\n"
        )
    return (
        "Pedagogie attendue: explique clairement la logique, puis donne des exemples progressifs "
        "et une mini verification actionnable."
    )


def _fallback_course(language: str, super_mode: bool, focus: str = "") -> str:
    focus_lower = focus.lower()
    learning_focus = _detect_learning_focus(focus)
    if "medieval" in focus_lower or "moyen age" in focus_lower or "médiéval" in focus_lower:
        return (
            f"Titre: Vocabulaire medieval guide ({language})\n"
            "Logique grammaticale: regroupe le lexique par categories (personnages, lieux, equipements, actions) "
            "pour memoriser plus vite et reutiliser les mots dans des phrases naturelles.\n"
            "Structures utiles: [sujet + verbe + complement] pour decrire des scenes historiques, "
            "et [there is/there are + groupe nominal] pour decrire lieux et elements de decor.\n"
            "Vocabulaire cle: knight, castle, sword, shield, king, queen, peasant, monk, village, battle, armor, crown.\n"
            "Exemples progressifs: commence par des groupes nominaux simples, puis formule des phrases courtes, "
            "ensuite ajoute un contexte temporel ou spatial.\n"
            "Mini verification: choisis 5 mots de la liste et cree mentalement 2 phrases coherentes avant le QCM.\n\n"
            "Si tu penses avoir compris, clique sur 'As-tu compris ? Oui' pour generer un QCM."
        )
    if learning_focus == "vocabulaire":
        return (
            f"Titre: Fiche vocabulaire guidee ({language})\n"
            "Logique grammaticale: memorise le lexique par familles (objets, actions, qualites) pour accelerer la reutilisation.\n"
            "Structures utiles: introduis les mots dans des phrases courtes sujet + verbe + complement pour consolider le sens.\n"
            "Vocabulaire cle: 1) castle -> chateau -> The castle is old. 2) knight -> chevalier -> A knight rides fast. "
            "3) shield -> bouclier -> He carries a shield. 4) sword -> epee -> The sword is sharp. "
            "5) crown -> couronne -> The crown is gold. 6) king -> roi -> The king speaks. "
            "7) queen -> reine -> The queen arrives. 8) peasant -> paysan -> The peasant works. "
            "9) monk -> moine -> The monk reads. 10) village -> village -> The village is quiet.\n"
            "Exemples progressifs: commence par 2 mots isoles, puis 2 groupes nominaux, puis 2 phrases completes.\n"
            "Mini verification: recite 5 termes avec leur traduction et produis 2 phrases personnelles.\n\n"
            "Si tu penses avoir compris, clique sur 'As-tu compris ? Oui' pour generer un QCM."
        )
    if learning_focus == "grammaire":
        return (
            f"Titre: Cours de grammaire appliquee ({language})\n"
            "Logique grammaticale: identifie d'abord la fonction des mots (sujet, verbe, complement), puis applique la regle cible.\n"
            "Structures utiles: schema de base affirmative/negative/interrogative + point de controle sur l'ordre des mots.\n"
            "Vocabulaire cle: sujet, verbe, auxiliaire, complement, accord, inversion, marqueur temporel.\n"
            "Exemples progressifs: exemple 1 tres simple, exemple 2 avec negation, exemple 3 interrogatif, exemple 4 phrase enrichie.\n"
            "Mini verification: verifie en 3 etapes (ordre, forme verbale, coherence de sens) avant de valider.\n\n"
            "Si tu penses avoir compris, clique sur 'As-tu compris ? Oui' pour generer un QCM."
        )
    if learning_focus == "orthographe":
        return (
            f"Titre: Atelier orthographe ({language})\n"
            "Logique grammaticale: la bonne orthographe depend de la forme canonique du mot et du contexte de la phrase.\n"
            "Structures utiles: repere radical + terminaison, puis controle les lettres pieges (doubles consonnes, voyelles muettes).\n"
            "Vocabulaire cle: forme correcte, erreur frequente, variante proche, homophone, terminaison, accentuation.\n"
            "Exemples progressifs: mot isole, mot en groupe nominal, mot dans phrase courte, mot dans phrase longue.\n"
            "Mini verification: corrige 3 mots difficiles et explique la regle appliquee pour chacun.\n\n"
            "Si tu penses avoir compris, clique sur 'As-tu compris ? Oui' pour generer un QCM."
        )
    if learning_focus == "conjugaison":
        return (
            f"Titre: Cours de conjugaison pratique ({language})\n"
            "Logique grammaticale: choisis le temps selon l'intention (habitude, action en cours, passe, projet) puis accorde au sujet.\n"
            "Structures utiles: sujet + forme verbale + complement, avec controle des marqueurs temporels.\n"
            "Vocabulaire cle: infinitif, base verbale, terminaison, auxiliaire, marqueur de temps, irregularite.\n"
            "Exemples progressifs: phrase simple au temps cible, puis 3 variantes avec sujets differents.\n"
            "Mini verification: recite 6 formes cles du verbe cible et valide-les dans 2 phrases.\n\n"
            "Si tu penses avoir compris, clique sur 'As-tu compris ? Oui' pour generer un QCM."
        )
    if super_mode:
        return (
            f"Titre: Super indice ({language})\n"
            "Logique grammaticale: pars de la structure de base, puis ajuste le verbe selon le sujet.\n"
            "Structures utiles: sujet + verbe, puis verification de l'accord.\n"
            "Vocabulaire cle: forme de base, accord, terminaison.\n"
            "Exemples progressifs: commence simple, puis ajoute le detail qui manque.\n"
            "Mini verification: quel est le dernier element que tu dois ajuster pour finaliser ta phrase ?"
        )
    return (
        f"Titre: Cours guide ({language})\n"
        "Logique grammaticale: identifie d'abord le sujet, puis le type de phrase attendu.\n"
        "Structures utiles: affirmative, negative et interrogative avec ordre correct des mots.\n"
        "Vocabulaire cle: mots de frequence, connecteurs utiles et verbes pivots.\n"
        "Exemples progressifs: construis d'abord une phrase simple, puis enrichis-la avec precision.\n"
        "Mini verification: reformule la regle avec tes mots avant de lancer le QCM.\n\n"
        "Si tu penses avoir compris, clique sur 'As-tu compris ? Oui' pour generer un QCM."
    )


def _contains_forbidden(text: str) -> bool:
    lower = text.lower()
    return any(word in lower for word in BANNED_WORDS)


def _extract_quoted_phrase(text: str) -> str:
    match = re.search(r'"([^"]+)"', text)
    if match:
        return match.group(1).strip()
    lower = text.lower()
    marker = "comment on dit"
    if marker in lower:
        idx = lower.find(marker) + len(marker)
        return text[idx:].strip(" :?!.")
    return ""


def _is_translation_request(text: str) -> bool:
    lower = text.lower()
    return "comment on dit" in lower or "how do you say" in lower or "traduis" in lower


def _targeted_translation_fallback(language: str, message: str) -> str | None:
    phrase = _extract_quoted_phrase(message).lower()
    if "chaussettes de l'archiduchesse" in phrase and language == "anglais":
        return (
            "Titre: Construction de la phrase de l'archiduchesse\n"
            "Logique grammaticale: on construit d'abord le groupe possessif, puis la question, puis l'intensification.\n"
            "Structures utiles: [possesseur + 's + nom] + [be + sujet + adjectif] + [intensifieur].\n"
            "Vocabulaire cle: archduchess, socks, dry, very.\n"
            "Exemples progressifs: commence par le noyau '... socks', puis ajoute 'are they dry?', puis renforce avec 'very'.\n"
            "Mini verification: peux-tu reconstituer la phrase finale complete sans que je te la donne ?\n"
            "Si tu penses avoir compris, clique sur 'As-tu compris ? Oui' pour generer un QCM."
        )
    return None


def _guided_translation_course(language: str, message: str, super_mode: bool) -> str:
    phrase = _extract_quoted_phrase(message) or "ta phrase"
    if language == "anglais":
        lower_phrase = phrase.lower()
        looks_like_nominal_group = (
            len(lower_phrase.split()) <= 4
            and not any(token in lower_phrase for token in [" est ", " sont ", " ai ", " as ", " a ", " was ", " were "])
        )
        if super_mode:
            return (
                "Titre: Super indice de construction\n"
                f"Logique grammaticale: pour \"{phrase}\", garde l'ordre anglais strict et verifie chaque bloc.\n"
                "Structures utiles: [Sujet] + [verbe adapte au sujet] + [article] + [adjectif] + [nom].\n"
                "Vocabulaire cle: repere d'abord les mots pivots (sujet, verbe, nom), puis ajoute les details.\n"
                "Exemples progressifs: assemble 2 blocs, puis 3, puis la phrase complete sans l'ecrire ici.\n"
                "Mini verification: quelle est ta proposition finale ? Compare-la avec les options du QCM."
            )
        if looks_like_nominal_group:
            return (
                "Titre: Traduction guidee d'un groupe nominal\n"
                f"Logique grammaticale: la demande \"{phrase}\" correspond a un groupe nominal, c'est-a-dire un bloc sans verbe conjugue. "
                "La priorite est donc de reconstruire l'ordre interne correctement. En anglais, l'adjectif qualificatif se place avant le nom, "
                "contrairement au schema francais le plus frequent. On doit aussi verifier la nature de l'article (defini ou indefini) avant d'assembler le bloc.\n"
                "Structures utiles: la structure de reference est [article + adjectif + nom]. "
                "Si ce groupe nominal doit ensuite etre place dans une phrase complete, on l'insere comme sujet ou complement sans changer son ordre interne. "
                "Exemple de mecanisme: d'abord le noyau nominal, puis l'element descriptif positionne avant ce noyau.\n"
                "Vocabulaire cle: identifie le nom principal en premier, car c'est lui qui fixe la categorie semantique de la phrase. "
                "Ensuite, choisis l'adjectif le plus naturel dans le contexte, puis l'article correspondant. "
                "Cette sequence evite les traductions mot-a-mot rigides et permet d'obtenir une expression idiomatique.\n"
                "Exemples progressifs: etape 1, isole le nom cible; etape 2, ajoute l'adjectif au bon endroit; etape 3, ajoute l'article; "
                "etape 4, teste mentalement une phrase plus longue qui reutilise ce groupe nominal.\n"
                "Mini verification: sans ecrire la formulation finale ici, verifie que ton ordre respecte bien [article + adjectif + nom], "
                "puis compare ta proposition aux options du QCM.\n\n"
                "Si tu penses avoir compris, clique sur 'As-tu compris ? Oui' pour generer un QCM."
            )
        return (
            "Titre: Traduction guidee mot par mot\n"
            f"Logique grammaticale: pour \"{phrase}\", commence par identifier la nature de chaque element: sujet, verbe, article, adjectif, nom et eventuels complements. "
            "Ensuite, reorganise ces blocs selon la syntaxe anglaise, qui est plus stable que la syntaxe francaise. "
            "Le point critique est de choisir la bonne forme verbale selon le sujet, puis de positionner l'adjectif avant le nom.\n"
            "Structures utiles: utilise une trame de construction progressive: [Sujet] + [verbe adapte] + [article] + [adjectif] + [nom] + [complement optionnel]. "
            "Cette trame sert de repere, puis tu ajustes selon le type de phrase (affirmative, interrogative ou negative). "
            "L'objectif est d'eviter la traduction litterale et de privilegier une formulation naturelle.\n"
            "Vocabulaire cle: repere les mots pivots qui portent le sens principal (verbe central, nom principal, adjectif descriptif). "
            "Traite d'abord ces pivots, puis ajoute les mots de liaison. "
            "Cette methode permet de rester precis sans perdre la coherence globale de la phrase.\n"
            "Exemples progressifs: etape 1, construis un squelette minimal sujet + verbe; etape 2, ajoute le groupe nominal complet; "
            "etape 3, verifie l'ordre des mots; etape 4, controle la fluidite finale en lecture continue.\n"
            "Mini verification: garde ta proposition finale en tete sans l'ecrire ici; "
            "ensuite valide-la avec le QCM pour confirmer que ta logique de construction est correcte.\n\n"
            "Si tu penses avoir compris, clique sur 'As-tu compris ? Oui' pour generer un QCM."
        )
    if super_mode:
        return (
            "Titre: Super indice de construction\n"
            f"Logique grammaticale: pour \"{phrase}\", decoupe la phrase en blocs et assemble-les dans l'ordre cible.\n"
            "Structures utiles: [Sujet] + [verbe] + [complements], selon la syntaxe de la langue choisie.\n"
            "Vocabulaire cle: valide chaque mot principal avant de finaliser.\n"
            "Exemples progressifs: construis une version courte, puis enrichis-la.\n"
            "Mini verification: trouve la meilleure formulation parmi les choix du QCM."
        )
    return (
        "Titre: Traduction guidee mot par mot\n"
        f"Logique grammaticale: pour \"{phrase}\", identifie les blocs de sens avant d'ecrire la phrase cible.\n"
        "Structures utiles: suis un ordre stable (sujet, verbe, complements) propre a la langue choisie.\n"
        "Vocabulaire cle: traduis d'abord les mots pivots, puis les mots de precision.\n"
        "Exemples progressifs: avance par petits blocs plutot qu'en phrase finale directe.\n"
        "Mini verification: garde ta proposition finale pour le QCM.\n\n"
        "Si tu penses avoir compris, clique sur 'As-tu compris ? Oui' pour generer un QCM."
    )


def _fallback_quiz(user_question: str) -> dict[str, Any]:
    lower = user_question.lower()
    if "archiduchesse" in lower or "comment on dit" in lower:
        return {
            "question": "Quelle phrase finale respecte la construction mot par mot expliquee par le coach ?",
            "choices": [
                "The socks of archduchess are dry very dry?",
                "The Archduchess's socks, are they dry, very dry?",
                "Archduchess socks are she dry?",
                "The archduchess are socks dry.",
            ],
            "correct": "The Archduchess's socks, are they dry, very dry?",
        }
    if "present" in lower or "simple" in lower:
        return {
            "question": "Quelle phrase est correcte au present simple ?",
            "choices": ["She go to school.", "She goes to school.", "She going to school.", "She gone to school."],
            "correct": "She goes to school.",
        }
    return {
        "question": "Quel choix applique la logique expliquee par le coach ?",
        "choices": [
            "Choisir le verbe sans verifier le sujet.",
            "Identifier sujet + structure avant de choisir la forme finale.",
            "Traduire mot a mot sans adaptation.",
            "Ignorer les indices de temps.",
        ],
        "correct": "Identifier sujet + structure avant de choisir la forme finale.",
    }


def _parse_quiz_json(raw: str) -> dict[str, Any] | None:
    text = raw.strip()
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except ValueError:
        return None
    if not isinstance(parsed, dict):
        return None
    question = str(parsed.get("question", "")).strip()
    choices = parsed.get("choices", [])
    correct = str(parsed.get("correct", "")).strip()
    if not question or not isinstance(choices, list) or len(choices) != 4 or not correct:
        return None
    clean_choices = [str(choice).strip() for choice in choices]
    if correct not in clean_choices:
        return None
    return {"question": question, "choices": clean_choices, "correct": correct}


def _parse_test_json(raw: str, expected_count: int) -> list[dict[str, Any]] | None:
    text = (raw or "").strip()
    if not text:
        return None
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    json_candidate = match.group(0)
    try:
        parsed = json.loads(json_candidate)
    except ValueError:
        return None
    if not isinstance(parsed, dict):
        return None
    questions = parsed.get("questions", [])
    if not isinstance(questions, list) or len(questions) < expected_count:
        return None
    cleaned: list[dict[str, Any]] = []
    signatures: set[str] = set()
    for item in questions:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question", "")).strip()
        choices = item.get("choices", [])
        correct = str(item.get("correct", "")).strip()
        explanation = str(item.get("explanation", "")).strip()
        if not question or not isinstance(choices, list) or len(choices) != 4 or not correct:
            continue
        clean_choices = [str(choice).strip() for choice in choices]
        if correct not in clean_choices:
            continue
        signature = re.sub(r"\s+", " ", question.lower()).strip()
        if signature in signatures:
            continue
        signatures.add(signature)
        cleaned.append(
            {
                "question": question,
                "choices": clean_choices,
                "correct": correct,
                "explanation": explanation or "Applique la logique vue dans le test.",
            }
        )
        if len(cleaned) == expected_count:
            break
    return cleaned if len(cleaned) == expected_count else None


def _clean_test_topic(raw_topic: str) -> str:
    topic = (raw_topic or "").strip()
    lower = topic.lower()
    patterns = [
        r"^(fais|fait|genere|génère|cree|crée)\s+(moi\s+)?(un\s+)?qcm\s+(sur|de)\s+",
        r"^(create|generate|make)\s+(me\s+)?(a\s+)?quiz\s+(on|about)\s+",
        r"^(fais|fait|genere|génère|cree|crée)\s+(un\s+)?test\s+(sur|de)\s+",
    ]
    for pat in patterns:
        lower = re.sub(pat, "", lower, flags=re.IGNORECASE).strip()
    cleaned = lower.strip(" \"'`.:;!?")
    return cleaned if len(cleaned) >= 2 else topic


def _fallback_they_test_questions(count: int) -> list[dict[str, Any]]:
    contexts = [
        "___ going to the castle tonight.",
        "Look over ___ near the old bridge.",
        "I think ___ late for class.",
        "___ books are on the table.",
        "___ planning a medieval roleplay event.",
        "We can meet ___ after lunch.",
        "___ house is next to mine.",
        "___ not ready yet.",
        "Put the shield over ___.",
        "___ friends are waiting outside.",
        "___ building a new project.",
        "The keys are right ___.",
        "___ teacher gave homework.",
        "___ excited about the quiz.",
        "Go ___ and check the room.",
        "___ team won the match.",
        "___ coming in five minutes.",
        "I saw ___ car near the market.",
        "___ always very polite.",
        "Sit over ___ by the window.",
    ]
    explanations = {
        "they": "'they' est un pronom sujet (ils/elles).",
        "they're": "'they're' est la contraction de 'they are'.",
        "there": "'there' indique un lieu (la/bas).",
        "their": "'their' est un adjectif possessif (leur).",
    }
    answers = [
        "they're",
        "there",
        "they're",
        "their",
        "they're",
        "there",
        "their",
        "they're",
        "there",
        "their",
        "they're",
        "there",
        "their",
        "they're",
        "there",
        "their",
        "they're",
        "their",
        "they",
        "there",
    ]
    base: list[dict[str, Any]] = []
    for i in range(count):
        idx = i % len(contexts)
        correct = answers[idx]
        base.append(
            {
                "question": f"Choisis le bon mot: {contexts[idx]} (Q{i + 1}/{count})",
                "choices": ["they", "they're", "there", "their"],
                "correct": correct,
                "explanation": explanations[correct],
            }
        )
    return base


def _fallback_test_questions(topic: str, count: int) -> list[dict[str, Any]]:
    topic_lower = topic.lower()
    if all(token in topic_lower for token in ["they", "there"]) and ("theyre" in topic_lower or "they're" in topic_lower):
        return _fallback_they_test_questions(count)

    templates: list[dict[str, Any]] = [
        {
            "question": f"[{topic}] Choisis la phrase correcte au present simple avec 'I'.",
            "choices": [
                "I have a solid understanding of this topic.",
                "I has a solid understanding of this topic.",
                "I having a solid understanding of this topic.",
                "I have a solidly understanding of this topic.",
            ],
            "correct": "I have a solid understanding of this topic.",
            "explanation": "Avec le sujet I, on utilise have (pas has).",
        },
        {
            "question": f"[{topic}] Repere le bon ordre adjectif + nom.",
            "choices": ["the castle old", "old the castle", "the old castle", "castle the old"],
            "correct": "the old castle",
            "explanation": "En anglais, l'adjectif se place avant le nom.",
        },
        {
            "question": f"[{topic}] Complete: 'They ____ every day.'",
            "choices": ["practice", "practices", "practicing", "practisedly"],
            "correct": "practice",
            "explanation": "Avec they au present simple, le verbe reste a la base.",
        },
        {
            "question": f"[{topic}] Choisis la meilleure traduction de groupe nominal.",
            "choices": ["a vocabulary rich", "rich a vocabulary", "a rich vocabulary", "vocabulary a rich"],
            "correct": "a rich vocabulary",
            "explanation": "Article + adjectif + nom est la structure attendue.",
        },
        {
            "question": f"[{topic}] Quelle phrase est correctement negative ?",
            "choices": ["He don't study.", "He doesn't study.", "He not studies.", "He doesn't studies."],
            "correct": "He doesn't study.",
            "explanation": "Avec does not, le verbe revient a la forme de base.",
        },
    ]
    base: list[dict[str, Any]] = []
    for i in range(count):
        tpl = templates[i % len(templates)]
        q = dict(tpl)
        q["question"] = f"{tpl['question']} (Q{i + 1}/{count})"
        base.append(q)
    return base


@app.get("/")
async def serve_index() -> FileResponse:
    return FileResponse(PROJECT_ROOT / "index.html")


@app.get("/style.css")
async def serve_css() -> FileResponse:
    return FileResponse(PROJECT_ROOT / "style.css")


@app.get("/main.js")
async def serve_js() -> FileResponse:
    return FileResponse(PROJECT_ROOT / "main.js")


@app.get("/meta")
async def meta_endpoint() -> dict[str, Any]:
    return {"languages": SUPPORTED_LANGUAGES, "super_hint_cost": SUPER_HINT_COST}


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest) -> ChatResponse:
    language = _normalize_language(payload.language)
    super_mode = payload.request_super_hint and payload.current_points >= SUPER_HINT_COST
    points_delta = -SUPER_HINT_COST if super_mode else 0
    translation_mode = _is_translation_request(payload.message)

    if _contains_forbidden(payload.message):
        return ChatResponse(
            response=(
                "Ce message contient un mot que je ne peux pas traiter tel quel. "
                "Reformule ta demande en restant sur l'apprentissage de langue, puis on repart proprement."
            ),
            points_delta=0,
            super_hint_cost=SUPER_HINT_COST,
            can_generate_quiz=False,
        )

    targeted = _targeted_translation_fallback(language, payload.message)
    if targeted is not None:
        return ChatResponse(
            response=targeted,
            points_delta=points_delta,
            super_hint_cost=SUPER_HINT_COST,
            can_generate_quiz=True,
        )

    if translation_mode:
        return ChatResponse(
            response=_guided_translation_course(language, payload.message, super_mode),
            points_delta=points_delta,
            super_hint_cost=SUPER_HINT_COST,
            can_generate_quiz=True,
        )

    system_prompt = get_chat_system_prompt(language=language, hint_mode="super" if super_mode else "normal")
    learning_focus = _detect_learning_focus(payload.message)
    user_prompt = (
        f"Historique:\n{_format_history(payload.history)}\n\n"
        f"Question eleve: {payload.message}\n\n"
        "Fais un mini-cours adapte a ce cas precis, pas un template repetitif.\n"
        "Longueur cible: 140 a 220 mots pour eviter les reponses tronquees.\n"
        "Donne une logique utile, un exemple concret, puis une mini verification.\n"
        f"{_focus_user_instruction(learning_focus)}\n"
        "Format: texte simple, sans markdown.\n"
        "Termine par: 'Si tu penses avoir compris, clique sur As-tu compris ? Oui pour generer un QCM.'"
    )

    try:
        response_text = await _call_llm(system_prompt, user_prompt, max_tokens=900, temperature=0.75)
        if _is_noisy(response_text):
            response_text = _fallback_course(language, super_mode, payload.message)
    except RuntimeError:
        response_text = "HF_TOKEN manquant dans .env. Le coach detaille est indisponible."
    except httpx.TimeoutException:
        response_text = _fallback_course(language, super_mode, payload.message)
    except httpx.HTTPStatusError as exc:
        response_text = f"Erreur Hugging Face ({exc.response.status_code}): {_extract_hf_error(exc.response.text)}"
    except (httpx.HTTPError, ValueError):
        response_text = _fallback_course(language, super_mode, payload.message)

    return ChatResponse(
        response=response_text,
        points_delta=points_delta,
        super_hint_cost=SUPER_HINT_COST,
        can_generate_quiz=not _contains_forbidden(response_text),
    )


@app.post("/quiz/generate", response_model=QuizPayload)
async def generate_quiz_endpoint(payload: QuizGenerateRequest) -> QuizPayload:
    language = _normalize_language(payload.language)

    system_prompt = get_quiz_generation_prompt(language)
    user_prompt = (
        f"Question initiale de l'eleve:\n{payload.user_question}\n\n"
        f"Cours produit:\n{payload.lesson_text}\n\n"
        "Genere un seul QCM qui verifie la comprehension de ce cours."
    )

    quiz_data: dict[str, Any] | None = None
    try:
        raw = await _call_llm(system_prompt, user_prompt, max_tokens=300, temperature=0.4)
        quiz_data = _parse_quiz_json(raw)
    except (RuntimeError, httpx.HTTPError, ValueError):
        quiz_data = None

    if quiz_data is None:
        quiz_data = _fallback_quiz(payload.user_question)

    quiz_id = str(uuid.uuid4())
    GENERATED_QUIZZES[quiz_id] = quiz_data
    return QuizPayload(id=quiz_id, question=quiz_data["question"], choices=quiz_data["choices"])


@app.post("/test/generate", response_model=TestGenerateResponse)
async def generate_test_endpoint(payload: TestGenerateRequest) -> TestGenerateResponse:
    language = _normalize_language(payload.language)
    topic = _clean_test_topic(payload.topic.strip())
    count = payload.count
    system_prompt = get_test_generation_prompt(language=language, count=count)
    user_prompt = (
        f"Sujet choisi par l'eleve: {topic}\n"
        f"Genere un test progressif de {count} questions sur ce sujet."
    )

    questions: list[dict[str, Any]] | None = None
    generation_source = "llm"
    try:
        raw = await _call_llm(system_prompt, user_prompt, max_tokens=3500, temperature=0.45)
        questions = _parse_test_json(raw, expected_count=count)
        if questions is None:
            repair_prompt = (
                "Reformate strictement en JSON valide et rien d'autre.\n"
                f"Sujet: {topic}\n"
                f"Nombre requis: {count}\n"
                "Format attendu: {'questions':[{'question':'...','choices':['...','...','...','...'],"
                "'correct':'...','explanation':'...'}]}\n"
                "Rappel: aucune question repetitive."
            )
            raw_repair = await _call_llm(system_prompt, repair_prompt, max_tokens=3500, temperature=0.2)
            questions = _parse_test_json(raw_repair, expected_count=count)
    except (RuntimeError, httpx.HTTPError, ValueError):
        questions = None

    if questions is None:
        questions = _fallback_test_questions(topic=topic, count=count)
        generation_source = "fallback"

    session_id = str(uuid.uuid4())
    TEST_SESSIONS[session_id] = {
        "topic": topic,
        "questions": questions,
        "answers": [None] * count,
        "score": 0,
    }
    public_questions = [TestQuestionPayload(question=q["question"], choices=q["choices"]) for q in questions]
    return TestGenerateResponse(
        session_id=session_id,
        topic=topic,
        questions=public_questions,
        generation_source=generation_source,
    )


@app.post("/test/answer", response_model=TestAnswerResponse)
async def answer_test_endpoint(payload: TestAnswerRequest) -> TestAnswerResponse:
    session = TEST_SESSIONS.get(payload.session_id)
    if not session:
        return TestAnswerResponse(
            correct=False,
            explanation="Session de test introuvable. Regenere un test.",
            points_gained=0,
            score=0,
            answered=0,
            total=20,
            finished=False,
        )

    questions = session["questions"]
    idx = payload.question_index
    if idx < 0 or idx >= len(questions):
        return TestAnswerResponse(
            correct=False,
            explanation="Index de question invalide.",
            points_gained=0,
            score=int(session["score"]),
            answered=sum(1 for ans in session["answers"] if ans is not None),
            total=len(questions),
            finished=False,
        )

    q = questions[idx]
    selected = payload.selected.strip()
    is_correct = selected == q["correct"]
    already_answered = session["answers"][idx] is not None
    points_gained = 0
    if not already_answered:
        session["answers"][idx] = selected
        if is_correct:
            session["score"] += 1
            points_gained = 5

    answered = sum(1 for ans in session["answers"] if ans is not None)
    return TestAnswerResponse(
        correct=is_correct,
        explanation=str(q.get("explanation", "Applique les regles vues pendant le cours.")),
        points_gained=points_gained,
        score=int(session["score"]),
        answered=answered,
        total=len(questions),
        finished=answered >= len(questions),
    )


@app.post("/quiz/answer", response_model=QuizAnswerResponse)
async def answer_quiz_endpoint(payload: QuizAnswerRequest) -> QuizAnswerResponse:
    quiz = GENERATED_QUIZZES.get(payload.quiz_id)
    if not quiz:
        return QuizAnswerResponse(
            feedback="QCM introuvable. Redemande un nouveau QCM.",
            points_gained=0,
            understanding_detected=False,
        )

    if payload.selected.strip() == quiz["correct"]:
        return QuizAnswerResponse(
            feedback=(
                "Bonne reponse ! Tu as bien compris le cours. "
                "Tu gagnes des points utilisables pour des super indices."
            ),
            points_gained=15,
            understanding_detected=True,
        )

    return QuizAnswerResponse(
        feedback=(
            "Pas encore. Reprends la logique du cours et reessaie. "
            "Si besoin, demande un super indice."
        ),
        points_gained=0,
        understanding_detected=False,
    )
