const state = {
  batch: 0,
  history: [],
  quickQuestions: [],
  loading: false,
};

const landingView = document.getElementById("landing-view");
const chatView = document.getElementById("chat-view");
const startChatBtn = document.getElementById("start-chat-btn");
const backHomeBtn = document.getElementById("back-home-btn");
const refreshBtn = document.getElementById("refresh-recommendations-btn");
const quickQuestionsWrap = document.getElementById("quick-questions");
const recommendationList = document.getElementById("recommendation-list");
const chatForm = document.getElementById("chat-form");
const chatTextarea = document.getElementById("chat-textarea");
const sendBtn = document.getElementById("send-btn");
const messageList = document.getElementById("message-list");
const routePill = document.getElementById("route-pill");

function getInlinePlaceholderSvg(name) {
  const safeName = String(name || "暂无图片")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
  return `data:image/svg+xml;utf8,${encodeURIComponent(
    `<svg xmlns="http://www.w3.org/2000/svg" width="640" height="360" viewBox="0 0 640 360">
      <defs>
        <linearGradient id="g" x1="0" x2="1" y1="0" y2="1">
          <stop offset="0%" stop-color="#eaf5ff"/>
          <stop offset="100%" stop-color="#f7fbff"/>
        </linearGradient>
      </defs>
      <rect width="640" height="360" fill="url(#g)"/>
      <text x="50%" y="48%" text-anchor="middle" fill="#315879"
            font-family="KaiTi, STKaiti, Kaiti SC, serif" font-size="28">图片加载失败</text>
      <text x="50%" y="60%" text-anchor="middle" fill="#4b6b8b"
            font-family="KaiTi, STKaiti, Kaiti SC, serif" font-size="22">${safeName}</text>
    </svg>`
  )}`;
}

function switchToChatView() {
  landingView.classList.add("hidden");
  chatView.classList.remove("hidden");
  chatTextarea.focus();
}

function switchToHomeView() {
  chatView.classList.add("hidden");
  landingView.classList.remove("hidden");
}

function setLoading(loading) {
  state.loading = loading;
  sendBtn.disabled = loading;
  chatTextarea.disabled = loading;
  sendBtn.textContent = loading ? "思考中..." : "发送";
}

function normalizeText(text) {
  return String(text || "").trim();
}

function formatErrorDetail(detail, status) {
  if (typeof detail === "string" && detail.trim()) {
    return detail.trim();
  }

  if (Array.isArray(detail)) {
    const messages = detail
      .map((item) => {
        if (!item || typeof item !== "object") {
          return normalizeText(item);
        }
        const location = Array.isArray(item.loc) ? item.loc.join(".") : "";
        const message = normalizeText(item.msg || item.message);
        if (location && message) {
          return `${location}：${message}`;
        }
        return message || location;
      })
      .filter(Boolean);
    if (messages.length > 0) {
      return messages.join("；");
    }
  }

  if (detail && typeof detail === "object") {
    for (const key of ["message", "error", "detail"]) {
      if (detail[key] !== undefined && detail[key] !== detail) {
        return formatErrorDetail(detail[key], status);
      }
    }
    try {
      return JSON.stringify(detail);
    } catch (_error) {
      // Continue to the HTTP status fallback below.
    }
  }

  return status ? `请求失败（HTTP ${status}）` : "请求失败";
}

function appendMessage(role, content, sources = []) {
  const item = document.createElement("article");
  item.className = `message message-${role === "user" ? "user" : "assistant"}`;
  item.textContent = content;

  if (role === "assistant" && Array.isArray(sources) && sources.length > 0) {
    const sourceStrip = document.createElement("div");
    sourceStrip.className = "source-strip";

    sources.forEach((source) => {
      const chip = document.createElement("span");
      chip.className = "source-chip";
      const recipeName = source.recipe_name || "未知菜品";
      const category = source.category || "未知分类";
      chip.textContent = `${recipeName} · ${category}`;
      sourceStrip.appendChild(chip);
    });

    item.appendChild(sourceStrip);
  }

  messageList.appendChild(item);
  messageList.scrollTop = messageList.scrollHeight;
}

function updateRoute(route) {
  if (!route) {
    routePill.textContent = "策略未知";
    return;
  }
  const strategy = route.strategy || "unknown";
  const confidence = typeof route.confidence === "number" ? route.confidence : 0;
  routePill.textContent = `${strategy} · 置信度 ${confidence.toFixed(2)}`;
}

function renderQuickQuestions(items) {
  quickQuestionsWrap.innerHTML = "";
  items.forEach((q) => {
    const btn = document.createElement("button");
    btn.className = "quick-item";
    btn.type = "button";
    btn.textContent = q;
    btn.addEventListener("click", () => {
      switchToChatView();
      sendQuestion(q);
    });
    quickQuestionsWrap.appendChild(btn);
  });
}

function renderRecommendations(items) {
  recommendationList.innerHTML = "";
  const imageVersion = Date.now();

  async function debugImageFailure(recipe) {
    try {
      if (!recipe || !recipe.key) {
        return;
      }
      const resp = await fetch(`/api/debug/image/${recipe.key}?probe=true&timeout=8`);
      const data = await resp.json();
      console.warn("[image-debug]", {
        name: recipe.name,
        key: recipe.key,
        resolved_type: data.resolved_type,
        candidate_image_exists: data.candidate_image_exists,
        raw_image_url: data.raw_image_url,
        resolved_url: data.resolved_url,
        remote_probe: data.remote_probe || null,
      });
    } catch (error) {
      console.warn("[image-debug] failed to fetch debug info", recipe && recipe.name, error);
    }
  }

  items.forEach((recipe, idx) => {
    const card = document.createElement("article");
    card.className = "recipe-card";
    card.style.animationDelay = `${idx * 60}ms`;
    card.addEventListener("click", () => {
      switchToChatView();
      sendQuestion(`推荐一道${recipe.name}，并告诉我关键步骤。`);
    });

    const cover = document.createElement("div");
    cover.className = "recipe-cover";

    const img = document.createElement("img");
    img.alt = recipe.name;
    img.loading = "lazy";
    const rawImageUrl =
      (typeof recipe.image_url === "string" && recipe.image_url.trim()) ||
      (typeof recipe.imageUrl === "string" && recipe.imageUrl.trim()) ||
      (recipe.key ? `/api/recipe-image/${recipe.key}` : "");
    const imageSrc = rawImageUrl
      ? `${rawImageUrl}${rawImageUrl.includes("?") ? "&" : "?"}v=${imageVersion}`
      : getInlinePlaceholderSvg(recipe.name);
    img.src = imageSrc;
    img.addEventListener("error", () => {
      debugImageFailure(recipe);
      if (!img.dataset.fallbackApplied) {
        img.dataset.fallbackApplied = "1";
        img.src = getInlinePlaceholderSvg(recipe.name);
      }
    });
    cover.appendChild(img);

    const tag = document.createElement("span");
    tag.className = "tag-chip";
    const firstTag = (recipe.tags && recipe.tags[0]) || recipe.category || "推荐";
    tag.textContent = firstTag;
    cover.appendChild(tag);

    const heart = document.createElement("button");
    heart.type = "button";
    heart.className = "heart-btn";
    heart.textContent = "♡";
    heart.addEventListener("click", (event) => {
      event.stopPropagation();
      heart.classList.toggle("active");
      heart.textContent = heart.classList.contains("active") ? "♥" : "♡";
    });
    cover.appendChild(heart);

    const body = document.createElement("div");
    body.className = "recipe-body";

    const title = document.createElement("h3");
    title.className = "recipe-title";
    title.textContent = recipe.name;
    body.appendChild(title);

    const meta = document.createElement("p");
    meta.className = "recipe-meta";
    meta.textContent = recipe.category;
    body.appendChild(meta);

    card.appendChild(cover);
    card.appendChild(body);
    recommendationList.appendChild(card);
  });
}

async function fetchHome() {
  const resp = await fetch("/api/home");
  if (!resp.ok) {
    throw new Error(`加载首页失败: ${resp.status}`);
  }
  const payload = await resp.json();
  state.quickQuestions = payload.quick_questions || [];
  renderQuickQuestions(state.quickQuestions);
  renderRecommendations(payload.recommendations || []);
}

async function fetchRecommendations(batch) {
  const url = `/api/recommendations?batch=${batch}&size=6`;
  const resp = await fetch(url);
  if (!resp.ok) {
    throw new Error(`加载推荐失败: ${resp.status}`);
  }
  const payload = await resp.json();
  renderRecommendations(payload.recommendations || []);
}

function buildHistoryPayload() {
  return state.history
    .filter((item) => normalizeText(item.content))
    .slice(-12)
    .map((item) => ({
      role: item.role,
      content: normalizeText(item.content),
    }));
}

async function sendQuestion(rawMessage) {
  const message = normalizeText(rawMessage);
  if (!message || state.loading) {
    return;
  }

  const historyPayload = buildHistoryPayload();
  appendMessage("user", message);
  state.history.push({ role: "user", content: message });
  setLoading(true);

  try {
    const resp = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message,
        history: historyPayload,
      }),
    });

    const payload = await resp.json().catch(() => ({}));
    if (!resp.ok) {
      throw new Error(formatErrorDetail(payload.detail, resp.status));
    }

    const answer = normalizeText(payload.answer);
    if (!answer) {
      throw new Error("服务未返回有效回答，请重试");
    }
    updateRoute(payload.route);
    appendMessage("assistant", answer, payload.sources || []);
    state.history.push({ role: "assistant", content: answer });
  } catch (error) {
    const text = error instanceof Error ? error.message : "未知错误";
    appendMessage("assistant", `请求失败：${text}`);
  } finally {
    setLoading(false);
    chatTextarea.focus();
  }
}

startChatBtn.addEventListener("click", switchToChatView);
backHomeBtn.addEventListener("click", switchToHomeView);

refreshBtn.addEventListener("click", async () => {
  state.batch += 1;
  try {
    await fetchRecommendations(state.batch);
  } catch (error) {
    const text = error instanceof Error ? error.message : "刷新失败";
    alert(text);
  }
});

chatForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const text = chatTextarea.value;
  chatTextarea.value = "";
  sendQuestion(text);
});

chatTextarea.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    chatForm.requestSubmit();
  }
});

async function bootstrap() {
  try {
    await fetchHome();
    appendMessage("assistant", "你好，我是你的菜谱助手。你可以从快捷问题开始，或直接提问。");
  } catch (error) {
    const text = error instanceof Error ? error.message : "初始化失败";
    appendMessage("assistant", `页面初始化失败：${text}`);
  }
}

bootstrap();
