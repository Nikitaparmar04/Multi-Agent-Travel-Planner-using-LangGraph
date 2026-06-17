/* =========================================
   TripMind — app.js
   ========================================= */

const API_BASE = "http://localhost:8000";

/* ── State ─────────────────────────────── */
let sessions = [];          // [{id, title, messages:[]}]
let activeSessionId = null;
let isThinking = false;

/* ── DOM Refs ──────────────────────────── */
const messagesEl      = document.getElementById("messages");
const welcomeScreen   = document.getElementById("welcomeScreen");
const queryInput      = document.getElementById("queryInput");
const sendBtn         = document.getElementById("sendBtn");
const chatContainer   = document.getElementById("chatContainer");
const agentStatus     = document.getElementById("agentStatus");
const statusLabel     = agentStatus.querySelector(".status-label");
const chatHistoryEl   = document.getElementById("chatHistory");
const thinkingToast   = document.getElementById("thinkingToast");
const sidebar         = document.getElementById("sidebar");
const menuBtn         = document.getElementById("menuBtn");
const sidebarClose    = document.getElementById("sidebarClose");
const newChatBtn      = document.getElementById("newChatBtn");

/* ── Stars ─────────────────────────────── */
function generateStars() {
  const container = document.getElementById("stars");
  for (let i = 0; i < 120; i++) {
    const s = document.createElement("div");
    s.className = "star";
    s.style.cssText = `
      left:${Math.random()*100}%;
      top:${Math.random()*100}%;
      --t:${2 + Math.random()*4}s;
      --o:${0.15 + Math.random()*0.5};
      animation-delay:${Math.random()*4}s;
    `;
    container.appendChild(s);
  }
}
generateStars();

/* ── Sidebar toggle (mobile) ────────────── */
let overlay = document.createElement("div");
overlay.className = "sidebar-overlay";
document.body.appendChild(overlay);

function openSidebar()  { sidebar.classList.add("open");  overlay.classList.add("visible"); }
function closeSidebar() { sidebar.classList.remove("open"); overlay.classList.remove("visible"); }

menuBtn.addEventListener("click", openSidebar);
sidebarClose.addEventListener("click", closeSidebar);
overlay.addEventListener("click", closeSidebar);

/* ── Session management ─────────────────── */
function createSession(firstMessage = null) {
  const id = Date.now().toString();
  const title = firstMessage
    ? firstMessage.slice(0, 38) + (firstMessage.length > 38 ? "…" : "")
    : "New Trip Plan";
  const session = { id, title, messages: [] };
  sessions.unshift(session);
  activeSessionId = id;
  renderSidebarHistory();
  return session;
}

function getActiveSession() {
  return sessions.find(s => s.id === activeSessionId);
}

function renderSidebarHistory() {
  chatHistoryEl.innerHTML = "";
  sessions.forEach(session => {
    const li = document.createElement("li");
    li.className = "history-item" + (session.id === activeSessionId ? " active" : "");
    li.innerHTML = `<span class="history-icon">✈️</span><span>${session.title}</span>`;
    li.addEventListener("click", () => {
      activeSessionId = session.id;
      renderSidebarHistory();
      renderMessages();
      closeSidebar();
    });
    chatHistoryEl.appendChild(li);
  });
}

/* ── Render messages ───────────────────── */
function renderMessages() {
  const session = getActiveSession();
  messagesEl.innerHTML = "";

  if (!session || session.messages.length === 0) {
    welcomeScreen.style.display = "flex";
    messagesEl.style.display = "none";
    return;
  }

  welcomeScreen.style.display = "none";
  messagesEl.style.display = "flex";

  session.messages.forEach(msg => appendMessageToDOM(msg, false));
  scrollToBottom();
}

/* ── Append a single message ────────────── */
function appendMessageToDOM(msg, animate = true) {
  welcomeScreen.style.display = "none";
  messagesEl.style.display = "flex";

  const row = document.createElement("div");
  row.className = `message-row ${msg.role}`;
  if (!animate) row.style.animation = "none";

  const avatar = document.createElement("div");
  avatar.className = `avatar ${msg.role === "ai" ? "ai-avatar" : "user-avatar"}`;
  avatar.textContent = msg.role === "ai" ? "🤖" : "👤";

  const content = document.createElement("div");
  content.className = "message-content";

  // Agent chips (for AI messages that have them)
  if (msg.role === "ai" && msg.agents && msg.agents.length > 0) {
    const chipsRow = document.createElement("div");
    chipsRow.className = "agent-chips";
    msg.agents.forEach(a => {
      const chip = document.createElement("div");
      chip.className = "agent-chip";
      chip.innerHTML = `<span>${a.icon}</span><span>${a.name}</span>`;
      chipsRow.appendChild(chip);
    });
    content.appendChild(chipsRow);
  }

  const bubble = document.createElement("div");
  bubble.className = "message-bubble";
  bubble.innerHTML = formatMessage(msg.text);

  const time = document.createElement("div");
  time.className = "message-time";
  time.textContent = formatTime(msg.timestamp || Date.now());

  content.appendChild(bubble);
  content.appendChild(time);

  if (msg.role === "ai") {
    row.appendChild(avatar);
    row.appendChild(content);
  } else {
    row.appendChild(content);
    row.appendChild(avatar);
  }

  messagesEl.appendChild(row);
  if (animate) scrollToBottom();
  return row;
}

/* ── Typing indicator ───────────────────── */
function showTyping(agents = []) {
  const row = document.createElement("div");
  row.className = "message-row ai";
  row.id = "typingRow";

  const avatar = document.createElement("div");
  avatar.className = "avatar ai-avatar";
  avatar.textContent = "🤖";

  const content = document.createElement("div");
  content.className = "message-content";

  if (agents.length > 0) {
    const chipsRow = document.createElement("div");
    chipsRow.className = "agent-chips";
    agents.forEach(a => {
      const chip = document.createElement("div");
      chip.className = "agent-chip";
      chip.innerHTML = `<div class="chip-dot"></div><span>${a.icon}</span><span>${a.name}</span>`;
      chipsRow.appendChild(chip);
    });
    content.appendChild(chipsRow);
  }

  const bubble = document.createElement("div");
  bubble.className = "message-bubble typing-bubble";
  bubble.innerHTML = `
    <div class="typing-dot"></div>
    <div class="typing-dot"></div>
    <div class="typing-dot"></div>`;

  content.appendChild(bubble);
  row.appendChild(avatar);
  row.appendChild(content);
  messagesEl.appendChild(row);

  welcomeScreen.style.display = "none";
  messagesEl.style.display = "flex";
  scrollToBottom();
}

function removeTyping() {
  const t = document.getElementById("typingRow");
  if (t) t.remove();
}

/* ── Format message text ────────────────── */
function formatMessage(text) {
  if (!text) return "";
  return text
    // Headers
    .replace(/^### (.+)$/gm, "<h3>$1</h3>")
    .replace(/^## (.+)$/gm, "<h3>$1</h3>")
    // Bold
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    // Italic
    .replace(/\*(.+?)\*/g, "<em>$1</em>")
    // Inline code
    .replace(/`(.+?)`/g, "<code>$1</code>")
    // Unordered lists
    .replace(/^\s*[-•] (.+)$/gm, "<li>$1</li>")
    .replace(/(<li>.*<\/li>)/s, "<ul>$1</ul>")
    // Numbered lists
    .replace(/^\s*\d+\. (.+)$/gm, "<li>$1</li>")
    // Line breaks
    .replace(/\n\n/g, "</p><p>")
    .replace(/\n/g, "<br>");
}

/* ── Format time ────────────────────────── */
function formatTime(ts) {
  return new Date(ts).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

/* ── Scroll to bottom ───────────────────── */
function scrollToBottom() {
  setTimeout(() => {
    chatContainer.scrollTop = chatContainer.scrollHeight;
  }, 50);
}

/* ── Set thinking state ─────────────────── */
function setThinking(val) {
  isThinking = val;
  sendBtn.disabled = val;
  queryInput.disabled = val;

  if (val) {
    agentStatus.classList.add("thinking");
    statusLabel.textContent = "Thinking…";
    thinkingToast.classList.add("visible");
  } else {
    agentStatus.classList.remove("thinking");
    statusLabel.textContent = "Ready";
    thinkingToast.classList.remove("visible");
  }
}

/* ── Send message ───────────────────────── */
async function sendMessage(text) {
  text = text.trim();
  if (!text || isThinking) return;

  // Create session if needed
  if (!activeSessionId) createSession(text);

  const session = getActiveSession();

  // Push user message
  const userMsg = { role: "user", text, timestamp: Date.now() };
  session.messages.push(userMsg);
  appendMessageToDOM(userMsg);

  // Update sidebar title on first message
  if (session.messages.length === 1) {
    session.title = text.slice(0, 38) + (text.length > 38 ? "…" : "");
    renderSidebarHistory();
  }

  queryInput.value = "";
  autoResize();

  // Show thinking
  setThinking(true);
  const agents = detectAgents(text);
  showTyping(agents);

  try {
    const response = await fetch(`${API_BASE}/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: text }),
    });

    removeTyping();

    if (!response.ok) {
      throw new Error(`Server error: ${response.status}`);
    }

    const data = await response.json();
    const answerText = data.answer || data.response || JSON.stringify(data);

    const aiMsg = {
      role: "ai",
      text: answerText,
      timestamp: Date.now(),
      agents,
    };
    session.messages.push(aiMsg);
    appendMessageToDOM(aiMsg);

  } catch (err) {
    removeTyping();
    const errorMsg = {
      role: "ai",
      text: `⚠️ **Connection error:** ${err.message}\n\nMake sure your FastAPI backend is running on \`localhost:8000\`.`,
      timestamp: Date.now(),
      agents: [],
    };
    session.messages.push(errorMsg);
    appendMessageToDOM(errorMsg);
  } finally {
    setThinking(false);
    queryInput.focus();
  }
}

/* ── Detect which agents are likely used ── */
function detectAgents(text) {
  const lower = text.toLowerCase();
  const agents = [];

  if (/weather|forecast|temperature|rain|climate/i.test(lower))
    agents.push({ name: "Weather Agent", icon: "🌤️" });
  if (/currency|convert|usd|eur|inr|exchange/i.test(lower))
    agents.push({ name: "Currency Agent", icon: "💱" });
  if (/place|attraction|restaurant|hotel|landmark|visit/i.test(lower))
    agents.push({ name: "Places Agent", icon: "📍" });
  if (/budget|cost|price|expense|calcul/i.test(lower))
    agents.push({ name: "Budget Agent", icon: "🧮" });
  if (agents.length === 0)
    agents.push({ name: "Travel Agent", icon: "✈️" });

  return agents;
}

/* ── Auto-resize textarea ───────────────── */
function autoResize() {
  queryInput.style.height = "auto";
  queryInput.style.height = Math.min(queryInput.scrollHeight, 140) + "px";
}

queryInput.addEventListener("input", autoResize);

/* ── Enter to send (Shift+Enter = newline) ─ */
queryInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage(queryInput.value);
  }
});

/* ── Send button ────────────────────────── */
sendBtn.addEventListener("click", () => sendMessage(queryInput.value));

/* ── Suggestion cards ───────────────────── */
document.querySelectorAll(".suggestion-card").forEach(card => {
  card.addEventListener("click", () => {
    const prompt = card.dataset.prompt;
    queryInput.value = prompt;
    autoResize();
    sendMessage(prompt);
  });
});

/* ── New Chat button ────────────────────── */
newChatBtn.addEventListener("click", () => {
  createSession();
  renderMessages();
  closeSidebar();
  queryInput.focus();
});

/* ── Init ───────────────────────────────── */
(function init() {
  // Load any persisted sessions from localStorage
  try {
    const saved = localStorage.getItem("tripmind_sessions");
    if (saved) {
      sessions = JSON.parse(saved);
      if (sessions.length > 0) {
        activeSessionId = sessions[0].id;
        renderSidebarHistory();
        renderMessages();
        return;
      }
    }
  } catch (_) {}

  // Show welcome screen
  welcomeScreen.style.display = "flex";
  messagesEl.style.display = "none";
})();

/* ── Persist sessions ───────────────────── */
setInterval(() => {
  try {
    localStorage.setItem("tripmind_sessions", JSON.stringify(sessions.slice(0, 20)));
  } catch (_) {}
}, 3000);

/* ── Connection check badge ─────────────── */
async function checkBackend() {
  try {
    const r = await fetch(`${API_BASE}/`, { signal: AbortSignal.timeout(2000) });
    if (r.ok || r.status < 500) return;
  } catch (_) {}
  // Show subtle offline badge in status
  statusLabel.textContent = "Backend offline";
  agentStatus.style.borderColor = "rgba(239,68,68,0.4)";
  agentStatus.querySelector(".status-dot").style.background = "#ef4444";
  agentStatus.querySelector(".status-dot").style.boxShadow = "0 0 8px #ef4444";
}
checkBackend();
