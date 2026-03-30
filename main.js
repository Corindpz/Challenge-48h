// Commentaire: frontend principal (coach + onglet se tester 20 QCM).
const API_BASE = "http://localhost:8000";

const state = {
  points: 0,
  language: "anglais",
  superHintCost: 30,
  history: [],
  lastUserQuestion: "",
  lastLessonText: "",
  pendingQuizIds: new Set(),
  activeTab: "coach",
  testSession: null,
};

const languageSelectEl = document.getElementById("languageSelect");
const pointsEl = document.getElementById("points");
const superHintBtnEl = document.getElementById("superHintBtn");
const resetBtnEl = document.getElementById("resetBtn");
const tabCoachBtnEl = document.getElementById("tabCoachBtn");
const tabTestBtnEl = document.getElementById("tabTestBtn");
const coachViewEl = document.getElementById("coachView");
const testViewEl = document.getElementById("testView");
const chatMessagesEl = document.getElementById("chatMessages");
const chatInputEl = document.getElementById("chatInput");
const sendBtnEl = document.getElementById("sendBtn");
const understandingPanelEl = document.getElementById("understandingPanel");
const understoodYesBtnEl = document.getElementById("understoodYesBtn");
const understoodNoBtnEl = document.getElementById("understoodNoBtn");
const testMessagesEl = document.getElementById("testMessages");
const testTopicInputEl = document.getElementById("testTopicInput");
const generateTestBtnEl = document.getElementById("generateTestBtn");

function updateHud() {
  pointsEl.textContent = String(state.points);
  superHintBtnEl.textContent = `Super indice (-${state.superHintCost})`;
}

function setBusy(isBusy) {
  sendBtnEl.disabled = isBusy;
  superHintBtnEl.disabled = isBusy;
  understoodYesBtnEl.disabled = isBusy;
  understoodNoBtnEl.disabled = isBusy;
  generateTestBtnEl.disabled = isBusy;
}

function addPoints(delta) {
  state.points = Math.max(0, state.points + delta);
  updateHud();
}

function setTab(tabName) {
  state.activeTab = tabName;
  const isCoach = tabName === "coach";
  coachViewEl.classList.toggle("hidden", !isCoach);
  testViewEl.classList.toggle("hidden", isCoach);
  tabCoachBtnEl.classList.toggle("tab-btn-active", isCoach);
  tabTestBtnEl.classList.toggle("tab-btn-active", !isCoach);
}

function appendBubble(role, text) {
  const el = document.createElement("div");
  el.className = `bubble ${role}`;
  if (role === "ai") {
    el.innerHTML = formatAssistantText(text);
  } else {
    el.textContent = text;
  }
  chatMessagesEl.appendChild(el);
  chatMessagesEl.scrollTop = chatMessagesEl.scrollHeight;
}

function appendTestBubble(role, text) {
  const el = document.createElement("div");
  el.className = `bubble ${role}`;
  if (role === "ai") {
    el.innerHTML = formatAssistantText(text);
  } else {
    el.textContent = text;
  }
  testMessagesEl.appendChild(el);
  testMessagesEl.scrollTop = testMessagesEl.scrollHeight;
}

function escapeHtml(value) {
  return (value || "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function formatAssistantText(raw) {
  // Commentaire: nettoie l'output et garde un rendu lisible meme si la structure varie.
  let text = escapeHtml(raw || "");
  text = text.replace(/^#{1,6}\s*/gm, "");
  text = text.replace(/^>\s?/gm, "");
  text = text.replace(/\*\*(.*?)\*\*/g, "$1");
  text = text.replace(/\*(.*?)\*/g, "$1");
  text = text.replace(/^\s*[-*]\s+/gm, "• ");
  text = text.replace(/`([^`]+)`/g, "$1");
  text = text.replace(/\n{3,}/g, "\n\n");

  const sectionRegex =
    /^(Titre|Traduction proposee|Traduction proposée|Logique grammaticale|Structures utiles|Vocabulaire cle|Vocabulaire clé|Exemples progressifs|Mini verification|Mini vérification|Conclusion|Plan de construction|Point de depart|Point de départ|Regle cle|Règle clé|Auto-check)\s*:\s*(.*)$/i;
  const lines = text.split("\n");
  const htmlLines = lines.map((line) => {
    const trimmed = line.trim();
    if (!trimmed) return "<br>";
    const sectionMatch = trimmed.match(sectionRegex);
    if (sectionMatch) {
      const label = sectionMatch[1];
      const content = sectionMatch[2] || "";
      return `<span class="ai-section">${label}:</span>${content ? ` ${content}` : ""}`;
    }
    if (/^•\s*/.test(trimmed)) return `<span class="ai-bullet">${trimmed}</span>`;
    return trimmed;
  });
  return htmlLines.join("<br>");
}

function resetConversation() {
  state.history = [];
  state.lastUserQuestion = "";
  state.lastLessonText = "";
  state.pendingQuizIds = new Set();
  chatMessagesEl.innerHTML = "";
  understandingPanelEl.classList.add("hidden");
  appendBubble("ai", "Conversation reinitialisee. Pose une nouvelle question et on repart proprement.");
}

function fillLanguageSelect(languages) {
  languageSelectEl.innerHTML = "";
  languages.forEach((lang) => {
    const option = document.createElement("option");
    option.value = lang;
    option.textContent = lang.charAt(0).toUpperCase() + lang.slice(1);
    if (lang === state.language) option.selected = true;
    languageSelectEl.appendChild(option);
  });
}

function buildQuizCard(quiz) {
  const wrapper = document.createElement("div");
  wrapper.className = "quiz-inline";
  wrapper.dataset.quizId = quiz.id;

  const title = document.createElement("h3");
  title.textContent = "QCM de verification";
  wrapper.appendChild(title);

  const q = document.createElement("p");
  q.className = "quiz-question";
  q.textContent = quiz.question;
  wrapper.appendChild(q);

  const options = document.createElement("div");
  options.className = "quiz-options";
  quiz.choices.forEach((choice, idx) => {
    const label = document.createElement("label");
    label.className = "quiz-option";
    label.innerHTML = `<input type="radio" name="quiz-${quiz.id}" value="${idx}" />${escapeHtml(choice)}`;
    options.appendChild(label);
  });
  wrapper.appendChild(options);

  const button = document.createElement("button");
  button.className = "btn btn-primary";
  button.textContent = "Valider QCM";
  wrapper.appendChild(button);

  const feedback = document.createElement("div");
  feedback.className = "quiz-feedback";
  wrapper.appendChild(feedback);

  button.addEventListener("click", async () => {
    if (state.pendingQuizIds.has(quiz.id)) return;
    const selected = wrapper.querySelector(`input[name="quiz-${quiz.id}"]:checked`);
    if (!selected) {
      feedback.textContent = "Choisis une réponse avant de valider.";
      feedback.className = "quiz-feedback bad";
      return;
    }
    const selectedText = quiz.choices[Number(selected.value)];
    state.pendingQuizIds.add(quiz.id);
    button.disabled = true;

    try {
      const res = await fetch(`${API_BASE}/quiz/answer`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ quiz_id: quiz.id, selected: selectedText }),
      });
      if (!res.ok) throw new Error("quiz answer failed");
      const data = await res.json();
      addPoints(data.points_gained);
      feedback.textContent = data.feedback + (data.points_gained > 0 ? ` (+${data.points_gained} points)` : "");
      feedback.className = data.understanding_detected ? "quiz-feedback good" : "quiz-feedback bad";
      state.history.push({ role: "assistant", content: data.feedback });
      understandingPanelEl.classList.add("hidden");
    } catch (error) {
      feedback.textContent = "Erreur de validation du QCM.";
      feedback.className = "quiz-feedback bad";
    } finally {
      state.pendingQuizIds.delete(quiz.id);
      button.disabled = false;
    }
  });

  chatMessagesEl.appendChild(wrapper);
  chatMessagesEl.scrollTop = chatMessagesEl.scrollHeight;
}

function buildTestQuestionCard(questionObj, index, total, sessionId) {
  const wrapper = document.createElement("div");
  wrapper.className = "quiz-inline";
  const title = document.createElement("h3");
  title.textContent = `QCM ${index + 1}/${total}`;
  wrapper.appendChild(title);

  const q = document.createElement("p");
  q.className = "quiz-question";
  q.textContent = questionObj.question;
  wrapper.appendChild(q);

  const options = document.createElement("div");
  options.className = "quiz-options";
  questionObj.choices.forEach((choice, idx) => {
    const label = document.createElement("label");
    label.className = "quiz-option";
    label.innerHTML = `<input type="radio" name="test-${index}" value="${idx}" />${escapeHtml(choice)}`;
    options.appendChild(label);
  });
  wrapper.appendChild(options);

  const button = document.createElement("button");
  button.className = "btn btn-primary";
  button.textContent = index === total - 1 ? "Valider le test" : "Valider et continuer";
  wrapper.appendChild(button);

  const feedback = document.createElement("div");
  feedback.className = "quiz-feedback";
  wrapper.appendChild(feedback);

  button.addEventListener("click", async () => {
    const selected = wrapper.querySelector(`input[name="test-${index}"]:checked`);
    if (!selected) {
      feedback.textContent = "Choisis une réponse avant de valider.";
      feedback.className = "quiz-feedback bad";
      return;
    }
    const selectedText = questionObj.choices[Number(selected.value)];
    button.disabled = true;
    try {
      const res = await fetch(`${API_BASE}/test/answer`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionId,
          question_index: index,
          selected: selectedText,
        }),
      });
      if (!res.ok) throw new Error("test answer failed");
      const data = await res.json();
      addPoints(data.points_gained || 0);
      feedback.textContent = `${data.correct ? "Bonne reponse" : "Mauvaise reponse"} - ${data.explanation}`;
      feedback.className = data.correct ? "quiz-feedback good" : "quiz-feedback bad";
      state.testSession.currentIndex += 1;
      setTimeout(() => {
        if (data.finished) {
          appendTestBubble(
            "ai",
            `Test termine. Score final: ${data.score}/${data.total}. Tu as valide ${data.answered} questions.`
          );
          return;
        }
        const nextIdx = state.testSession.currentIndex;
        const nextQuestion = state.testSession.questions[nextIdx];
        buildTestQuestionCard(nextQuestion, nextIdx, state.testSession.questions.length, sessionId);
      }, 300);
    } catch (error) {
      feedback.textContent = "Erreur de validation du test.";
      feedback.className = "quiz-feedback bad";
      button.disabled = false;
    }
  });

  testMessagesEl.appendChild(wrapper);
  testMessagesEl.scrollTop = testMessagesEl.scrollHeight;
}

async function generateTest() {
  const topic = testTopicInputEl.value.trim();
  if (!topic) {
    appendTestBubble("ai", "Entre un sujet pour generer ton test de 20 questions.");
    return;
  }
  appendTestBubble("user", topic);
  testTopicInputEl.value = "";
  setBusy(true);
  try {
    const res = await fetch(`${API_BASE}/test/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ language: state.language, topic, count: 20 }),
    });
    if (!res.ok) throw new Error("test generate failed");
    const data = await res.json();
    state.testSession = {
      sessionId: data.session_id,
      questions: data.questions,
      currentIndex: 0,
    };
    const sourceLabel = data.generation_source === "llm" ? "Modele IA" : "Fallback de secours";
    appendTestBubble(
      "ai",
      `Test cree sur le theme "${data.topic}". Source: ${sourceLabel}. On commence maintenant: 20 questions progressives.`
    );
    testMessagesEl.innerHTML = "";
    appendTestBubble(
      "ai",
      `Test cree sur le theme "${data.topic}". Source: ${sourceLabel}. On commence maintenant: 20 questions progressives.`
    );
    buildTestQuestionCard(data.questions[0], 0, data.questions.length, data.session_id);
  } catch (error) {
    appendTestBubble("ai", "Impossible de generer le test pour le moment.");
  } finally {
    setBusy(false);
  }
}

async function sendChat(message, requestSuperHint = false) {
  const content = message.trim();
  if (!content) return;

  state.lastUserQuestion = content;
  appendBubble("user", content);
  state.history.push({ role: "user", content });
  chatInputEl.value = "";
  setBusy(true);
  understandingPanelEl.classList.add("hidden");

  try {
    const res = await fetch(`${API_BASE}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: content,
        language: state.language,
        history: state.history,
        current_points: state.points,
        request_super_hint: requestSuperHint,
      }),
    });
    if (!res.ok) throw new Error("chat failed");
    const data = await res.json();
    addPoints(data.points_delta || 0);
    state.superHintCost = data.super_hint_cost ?? state.superHintCost;
    state.lastLessonText = data.response;
    appendBubble("ai", data.response);
    state.history.push({ role: "assistant", content: data.response });
    understandingPanelEl.classList.toggle("hidden", !data.can_generate_quiz);
  } catch (error) {
    appendBubble("ai", "Erreur de connexion au serveur. Verifie localhost:8000.");
  } finally {
    setBusy(false);
    updateHud();
  }
}

async function generateQuizFromLesson() {
  if (!state.lastLessonText || !state.lastUserQuestion) {
    appendBubble("ai", "Je n'ai pas assez de contexte pour generer un QCM. Pose d'abord une question.");
    return;
  }

  setBusy(true);
  try {
    const res = await fetch(`${API_BASE}/quiz/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        language: state.language,
        lesson_text: state.lastLessonText,
        user_question: state.lastUserQuestion,
      }),
    });
    if (!res.ok) throw new Error("quiz generate failed");
    const quiz = await res.json();
    appendBubble("ai", "Super, voici le QCM base sur le cours qu'on vient de faire :");
    buildQuizCard(quiz);
  } catch (error) {
    appendBubble("ai", "Impossible de generer le QCM pour le moment.");
  } finally {
    setBusy(false);
  }
}

function askSuperHint() {
  if (state.points < state.superHintCost) {
    appendBubble(
      "ai",
      `Tu n'as pas assez de points pour un super indice. Il faut ${state.superHintCost} points.`
    );
    return;
  }
  sendChat("Je veux un super indice sur ce sujet.", true);
}

async function init() {
  try {
    const res = await fetch(`${API_BASE}/meta`);
    if (!res.ok) throw new Error("meta failed");
    const data = await res.json();
    fillLanguageSelect(data.languages ?? ["anglais"]);
    state.superHintCost = data.super_hint_cost ?? 30;
  } catch (error) {
    fillLanguageSelect(["anglais", "espagnol", "allemand", "italien"]);
  }

  languageSelectEl.addEventListener("change", () => {
    state.language = languageSelectEl.value;
    appendBubble("ai", `Langue active: ${state.language}. Pose une question sur cette langue.`);
    appendTestBubble("ai", `Langue active pour le test: ${state.language}.`);
  });

  tabCoachBtnEl.addEventListener("click", () => setTab("coach"));
  tabTestBtnEl.addEventListener("click", () => setTab("test"));
  sendBtnEl.addEventListener("click", () => sendChat(chatInputEl.value, false));
  superHintBtnEl.addEventListener("click", askSuperHint);
  resetBtnEl.addEventListener("click", resetConversation);
  generateTestBtnEl.addEventListener("click", generateTest);
  understoodYesBtnEl.addEventListener("click", generateQuizFromLesson);
  understoodNoBtnEl.addEventListener("click", () =>
    sendChat("Je n'ai pas compris, peux-tu expliquer autrement avec plus de structure ?", false)
  );

  chatInputEl.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      sendChat(chatInputEl.value, false);
    }
  });
  testTopicInputEl.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      generateTest();
    }
  });

  appendBubble(
    "ai",
    "Bienvenue sur EduGuardIA Lang. Pose une question, je te fais un cours detaille. Ensuite clique sur 'As-tu compris ? Oui' pour un QCM cible."
  );
  appendTestBubble(
    "ai",
    "Onglet Se tester: entre un sujet, puis je te genere un test complet de 20 QCM progressifs."
  );
  updateHud();
}

init();
