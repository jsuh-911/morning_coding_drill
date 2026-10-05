/* Morning Coding Drill -- browser side.
 *
 * Holds no judgement of its own: it renders what the server decided and sends
 * back code. The two buttons are the whole interaction model -- Run is free,
 * Submit costs an attempt.
 */

const CDN = "https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/0.52.2/min";

const el = (id) => document.getElementById(id);

const state = {
  challengeId: null,
  answerName: "answer",
  passed: false,
  datasetView: "rows",
  dataset: null,
  busy: false,
};

/* ------------------------------------------------------------------ api */

function token() {
  const fromUrl = new URLSearchParams(location.search).get("token");

  if (fromUrl) {
    try { localStorage.setItem("drill_token", fromUrl); } catch {}
    return fromUrl;
  }

  try { return localStorage.getItem("drill_token") || ""; } catch { return ""; }
}

async function api(path, body) {
  const headers = { "Content-Type": "application/json" };
  const tok = token();

  if (tok) headers["X-Drill-Token"] = tok;

  const res = await fetch(`/api${path}`, {
    method: body === undefined ? "GET" : "POST",
    headers,
    body: body === undefined ? undefined : JSON.stringify(body),
  });

  const text = await res.text();
  let data = {};

  try { data = text ? JSON.parse(text) : {}; } catch { data = { detail: text }; }

  if (!res.ok) throw new Error(data.detail || `${res.status} ${res.statusText}`);

  return data;
}

/* -------------------------------------------------------------- editor */

let editor = null;
let fallback = null;

const STARTER = (name) =>
  `# df is the challenge's DataFrame. pd and np are imported.\n` +
  `# Run is free; only Submit counts an attempt.\n` +
  `# Put your result in \`${name}\`.\n\n` +
  `df.head()\n`;

function code() {
  return editor ? editor.getValue() : fallback.value;
}

function setCode(text) {
  if (editor) editor.setValue(text);
  else fallback.value = text;
}

function draftKey(id) {
  return `drill_draft_${id}`;
}

function saveDraft() {
  if (!state.challengeId) return;
  try { localStorage.setItem(draftKey(state.challengeId), code()); } catch {}
}

function loadDraft(id) {
  try { return localStorage.getItem(draftKey(id)); } catch { return null; }
}

function bootEditor(onReady) {
  const dark = matchMedia("(prefers-color-scheme: dark)").matches;

  const ready = (ok) => {
    if (!ok) {
      // Monaco unreachable -- offline, or the CDN is blocked. A textarea is
      // a worse editor but it is never the reason you cannot practise.
      fallback = el("fallback");
      fallback.hidden = false;
      el("editor").hidden = true;
      bindKeys(fallback);
    }
    onReady();
  };

  if (typeof require === "undefined") return ready(false);

  const failed = setTimeout(() => ready(false), 6000);

  window.MonacoEnvironment = {
    getWorkerUrl: () =>
      "data:text/javascript;charset=utf-8," +
      encodeURIComponent(
        `self.MonacoEnvironment={baseUrl:'${CDN}/'};` +
        `importScripts('${CDN}/vs/base/worker/workerMain.js');`
      ),
  };

  require.config({ paths: { vs: `${CDN}/vs` } });

  require(["vs/editor/editor.main"], () => {
    clearTimeout(failed);

    editor = monaco.editor.create(el("editor"), {
      value: "",
      language: "python",
      theme: dark ? "vs-dark" : "vs",
      fontSize: 13.5,
      lineHeight: 22,
      fontFamily: 'ui-monospace, "SF Mono", Menlo, monospace',
      minimap: { enabled: false },
      scrollBeyondLastLine: false,
      automaticLayout: true,
      renderLineHighlight: "none",
      padding: { top: 14, bottom: 14 },
      tabSize: 4,
      rulers: [88],
      overviewRulerLanes: 0,
      scrollbar: { verticalScrollbarSize: 9, horizontalScrollbarSize: 9 },
    });

    editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.Enter, doRun);
    editor.addCommand(
      monaco.KeyMod.CtrlCmd | monaco.KeyMod.Shift | monaco.KeyCode.Enter,
      doSubmit
    );
    editor.onDidChangeModelContent(saveDraft);

    ready(true);
  }, () => { clearTimeout(failed); ready(false); });
}

function bindKeys(node) {
  node.addEventListener("input", saveDraft);
  node.addEventListener("keydown", (e) => {
    if (e.key !== "Enter" || !(e.metaKey || e.ctrlKey)) return;
    e.preventDefault();
    e.shiftKey ? doSubmit() : doRun();
  });
}

/* ------------------------------------------------------------- tables */

function tableHtml(t) {
  if (!t) return "";

  const head = t.columns.map((c) => `<th>${esc(c)}</th>`).join("");

  const rows = t.rows
    .map((r) => {
      const cells = r
        .map((v, i) => {
          const cls = [i === 0 && t.columns[0] === "index" ? "idx" : "", v === null ? "nil" : ""]
            .filter(Boolean)
            .join(" ");
          return `<td${cls ? ` class="${cls}"` : ""}>${v === null ? "NaN" : esc(v)}</td>`;
        })
        .join("");
      return `<tr>${cells}</tr>`;
    })
    .join("");

  return `<table><thead><tr>${head}</tr></thead><tbody>${rows}</tbody></table>`;
}

function esc(v) {
  return String(v).replace(/[&<>"]/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])
  );
}

function renderDataset() {
  const d = state.dataset;
  if (!d) return;

  const showing = state.datasetView === "describe" && d.describe ? d.describe : d;

  el("dataset-shape").textContent =
    `${d.shape[0].toLocaleString()} rows x ${d.shape[1]} columns` +
    (showing.truncated ? ` - first ${showing.rows.length} shown` : "");

  el("dataset-table").innerHTML = tableHtml(showing);

  document.querySelectorAll(".tab").forEach((b) =>
    b.classList.toggle("on", b.dataset.view === state.datasetView)
  );
}

/* ---------------------------------------------------------- challenge */

function renderChallenge(c) {
  el("empty").hidden = true;
  el("question").hidden = false;

  const fresh = c.key !== state.challengeId;
  state.challengeId = c.key;
  state.answerName = "answer";

  el("crumb").textContent = `${c.domain} / ${c.topic} / level ${c.level}`;
  el("title").textContent = c.title || "";
  el("mission").textContent = c.mission || "";

  show("progress-note", c.progress_note, (n, v) => (n.textContent = v));
  show("return-wrap", c.return_kind, () => (el("return-kind").textContent = c.return_kind));

  show("toolbox-wrap", c.toolbox?.length, () => {
    el("toolbox").innerHTML = c.toolbox.map((t) => `<li><code>${esc(t)}</code></li>`).join("");
  });

  show("constraints-wrap", c.constraints?.length, () => {
    el("constraints").innerHTML = c.constraints.map((t) => `<li>${esc(t)}</li>`).join("");
  });

  state.dataset = c.dataset;
  show("dataset-wrap", c.dataset, renderDataset);

  el("attempt").textContent = c.attempt ? `attempt ${c.attempt} spent` : "";

  if (fresh) {
    // A miss keeps your code -- that is the whole point of the hint ladder.
    // A new challenge starts clean, from the draft if the page was reloaded.
    setCode(loadDraft(c.key) ?? STARTER(state.answerName));
    el("pane-question").scrollTop = 0;
    editor?.focus();
  }
}

function show(id, cond, fill) {
  const node = el(id);
  node.hidden = !cond;
  if (cond && fill) fill(node, cond);
}

/* ------------------------------------------------------------- output */

function renderRun(r) {
  const parts = [];

  if (r.traceback) {
    parts.push(block("Error", `<pre class="bad">${esc(r.traceback)}</pre>`));
  }

  if (r.stdout) {
    parts.push(block("Printed", `<pre>${esc(r.stdout)}</pre>`));
  }

  if (r.value) {
    const body = r.value.table
      ? `<div class="scroll">${tableHtml(r.value.table)}</div>`
      : `<pre>${esc(r.value.repr)}</pre>`;

    parts.push(block(`Last expression &middot; ${esc(r.value.kind)}`, body));
  }

  if (r.answer_bound && r.answer) {
    const body = r.answer.table
      ? `<div class="scroll">${tableHtml(r.answer.table)}</div>`
      : `<pre>${esc(r.answer.repr)}</pre>`;

    parts.push(block(`${state.answerName} &middot; ${esc(r.answer.kind)}`, body));
  } else if (r.ok && !r.value) {
    parts.push(
      block(
        "No answer",
        `<p>The code ran, but nothing called <code>${state.answerName}</code> was bound.
         Submit will be turned away until it is.</p>`
      )
    );
  }

  el("output").innerHTML =
    `<div class="verdict ${r.ok ? "warn" : "fail"}"><b>${r.ok ? "RAN" : "RAISED"}</b>
     <span>nothing counted</span></div>` + parts.join("");
}

function renderReview(v, extra = {}) {
  if (!v) {
    el("output").innerHTML = "";
    return;
  }

  const parts = [];
  const cls = v.passed ? "pass" : "fail";
  const word = v.passed ? "PASS" : "TRY AGAIN";
  const aside = v.passed
    ? `on attempt ${v.attempt}`
    : `attempt ${v.attempt}`;

  parts.push(`<div class="verdict ${cls}"><b>${word}</b><span>${aside}</span></div>`);

  if (v.constraints_only) {
    parts.push(
      `<p class="violation">The answer is right, but the mission's constraints were not met.</p>`
    );
  }

  v.violations.forEach((text) =>
    parts.push(`<p class="violation">constraint &mdash; ${esc(text)}</p>`)
  );

  if (!v.passed) {
    if (v.root_cause) {
      parts.push(
        block(`Diagnosis &middot; ${esc(v.error_class || "?")}`, `<p>${esc(v.root_cause)}</p>`)
      );
    }
    if (v.hint && v.hint !== v.root_cause) {
      parts.push(block(`Hint &middot; attempt ${v.attempt}`, `<p>${esc(v.hint)}</p>`));
    }
  } else {
    if (v.strengths) parts.push(block("Strengths", `<p>${esc(v.strengths)}</p>`));
    if (v.idiomatic_note) parts.push(block("Note", `<p>${esc(v.idiomatic_note)}</p>`));
    if (v.reference) parts.push(block("Reference", `<pre>${esc(v.reference)}</pre>`));

    parts.push(
      `<div class="next"><button class="primary" id="btn-next">Next challenge</button></div>`
    );
  }

  if (extra.stdout) parts.push(block("Printed", `<pre>${esc(extra.stdout)}</pre>`));
  // coach_status() reports what the environment is configured for; this
  // session may still have been built with agents off, so say that instead.
  if (!v.agents_enabled) {
    parts.push(
      `<p class="said">Rules only &mdash; no model this session, so the verdict,
       the constraints and the structural report stand, and the prose is what
       is missing.</p>`
    );
  }

  el("output").innerHTML = parts.join("");
  el("btn-next")?.addEventListener("click", start);

  state.passed = v.passed;
}

function block(title, inner) {
  return `<div class="block"><h3>${title}</h3>${inner}</div>`;
}

function renderRejection(d) {
  el("output").innerHTML =
    `<div class="verdict warn"><b>NOT SUBMITTED</b><span>no attempt counted</span></div>` +
    `<div class="block"><p>${esc(d.reason)}</p></div>` +
    (d.exec?.traceback ? block("Error", `<pre class="bad">${esc(d.exec.traceback)}</pre>`) : "") +
    (d.exec?.stdout ? block("Printed", `<pre>${esc(d.exec.stdout)}</pre>`) : "");
}

/* ------------------------------------------------------------ actions */

function busy(on, label) {
  state.busy = on;
  ["btn-run", "btn-submit", "btn-start", "btn-ask"].forEach((id) => {
    const b = el(id);
    if (b) b.disabled = on;
  });

  if (on && label) {
    el("output").innerHTML = `<p class="hint-line">${esc(label)}</p>`;
  }
}

function fail(err) {
  el("output").innerHTML =
    `<div class="verdict fail"><b>ERROR</b><span>the app, not your code</span></div>` +
    `<pre class="bad">${esc(err.message || err)}</pre>`;
}

async function start() {
  busy(true, "The progress agent is reading your record...");

  try {
    const d = await api("/start", {});
    apply(d);
    el("output").innerHTML =
      `<p class="hint-line">Bind your result to <code>${state.answerName}</code>. ` +
      `<b>Run</b> is free. <b>Submit</b> costs an attempt.</p>`;
  } catch (e) {
    fail(e);
  } finally {
    busy(false);
  }
}

async function doRun() {
  if (state.busy) return;
  busy(true, "Running...");

  try {
    renderRun(await api("/run", { code: code() }));
  } catch (e) {
    fail(e);
  } finally {
    busy(false);
  }
}

async function doSubmit() {
  if (state.busy) return;
  busy(true, "The grader agent is reading your code...");

  try {
    const d = await api("/submit", { code: code() });

    if (d.rejected) {
      renderRejection(d);
    } else {
      apply(d, { skipChallenge: !d.review?.passed });
      renderReview(d.review, { stdout: d.exec?.stdout });
    }
  } catch (e) {
    fail(e);
  } finally {
    busy(false);
  }
}

function apply(d, opts = {}) {
  state.answerName = d.answer_name || "answer";
  el("answer-name") && (el("answer-name").textContent = state.answerName);

  const dot = el("mode-dot");
  dot.className = `dot ${d.agents_enabled ? "live" : "rules"}`;
  dot.title = d.agents_enabled
    ? `agents on - exec ${d.exec_mode}`
    : `rules only, no model - exec ${d.exec_mode}`;

  if (d.challenge && !opts.skipChallenge) renderChallenge(d.challenge);
  else if (d.challenge) el("attempt").textContent = `attempt ${d.challenge.attempt} spent`;
}

/* ------------------------------------------------------------- sheets */

function sheet(html) {
  el("sheet-body").innerHTML = html;
  el("sheet").hidden = false;
}

function closeSheet() {
  el("sheet").hidden = true;
}

async function showStatus() {
  sheet(`<h2>Progress</h2><p class="hint-line">Loading...</p>`);

  try {
    const s = await api("/status");

    const domains = s.domains
      .map(
        (d) =>
          `<h3>${esc(d.domain)}</h3><table class="rows">` +
          d.topics
            .map(
              (t) =>
                `<tr><td class="t">${esc(t.topic)}</td>` +
                `<td class="s ${esc(t.status)}">${esc(t.status)}</td>` +
                `<td class="n">L${t.level ?? "-"}</td>` +
                `<td class="n">${t.correct}/${t.attempts}</td></tr>`
            )
            .join("") +
          `</table>`
      )
      .join("");

    const score = s.today.score;

    const today = s.today.challenges.length
      ? `<h3>Today</h3><table class="rows">` +
        s.today.challenges
          .map(
            (c) =>
              `<tr><td class="t">${esc(c.topic)}</td>` +
              `<td class="s ${c.passed ? "mastered" : "needs_rework"}">` +
              `${c.passed ? "pass" : "unfinished"}</td>` +
              `<td class="n">L${c.level}</td>` +
              `<td class="n">${c.attempts} att</td>` +
              `<td class="n">${c.score}</td></tr>`
          )
          .join("") +
        `</table>` +
        (score
          ? `<p class="said">${score.passed}/${score.challenges} passed,
             ${score.first_try} first try, mean ${score.mean_attempts} attempts,
             points ${score.points}</p>`
          : "")
      : `<h3>Today</h3><p class="said">Nothing answered yet.</p>`;

    sheet(
      `<h2>Progress</h2>${domains}${today}` +
      `<p class="said">${s.agents_enabled ? esc(s.coach_status) : "agents off - rules only"} ` +
      `&middot; exec ${esc(s.exec_mode)} &middot; state in ${esc(s.data_dir)}</p>`
    );
  } catch (e) {
    sheet(`<h2>Progress</h2><pre class="bad">${esc(e.message)}</pre>`);
  }
}

function showSave() {
  sheet(
    `<h2>Close out the session</h2>
     <p class="hint-line">Every submission is already written. This adds the one
     thing the score cannot see &mdash; and tomorrow's progress agent weighs it
     against the pass rate.</p>
     <textarea class="notes" id="notes" placeholder="merges still feel unfamiliar"></textarea>
     <div class="sheet-actions">
       <button class="primary" id="btn-save-go">Write to drill_records.json</button>
       <button class="ghost" id="btn-save-cancel">Cancel</button>
     </div>`
  );

  el("notes").focus();
  el("btn-save-cancel").addEventListener("click", closeSheet);

  el("btn-save-go").addEventListener("click", async () => {
    const notes = el("notes").value;
    el("btn-save-go").disabled = true;

    try {
      const d = await api("/save", { notes });
      const s = d.score;
      sheet(
        `<h2>Written</h2>
         <p>${s.passed}/${s.challenges} passed, ${s.first_try} first try,
         mean ${s.mean_attempts} attempts, points ${s.points}.</p>
         <p class="said">${esc(d.records_path)}</p>`
      );
    } catch (e) {
      sheet(`<h2>Could not save</h2><pre class="bad">${esc(e.message)}</pre>`);
    }
  });
}

function showAsk() {
  sheet(
    `<h2>Ask</h2>
     <p class="hint-line">A question about the mission, without spending an attempt.</p>
     <textarea class="notes" id="q" placeholder="why idxmax rather than max?"></textarea>
     <div class="sheet-actions">
       <button class="primary" id="btn-ask-go">Ask</button>
       <button class="ghost" id="btn-ask-cancel">Cancel</button>
     </div>
     <div id="answer"></div>`
  );

  el("q").focus();
  el("btn-ask-cancel").addEventListener("click", closeSheet);

  const go = async () => {
    const question = el("q").value.trim();
    if (!question) return;

    el("btn-ask-go").disabled = true;
    el("answer").innerHTML = `<p class="said">Thinking...</p>`;

    try {
      const d = await api("/ask", { question });
      el("answer").innerHTML = d.text
        ? block("Answer", `<p>${esc(d.text)}</p>`)
        : `<p class="said">No model configured, so there is no prose coach.</p>`;
    } catch (e) {
      el("answer").innerHTML = `<pre class="bad">${esc(e.message)}</pre>`;
    } finally {
      el("btn-ask-go").disabled = false;
    }
  };

  el("btn-ask-go").addEventListener("click", go);
  el("q").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) go();
  });
}

/* --------------------------------------------------------------- boot */

el("btn-start").addEventListener("click", start);
el("btn-run").addEventListener("click", doRun);
el("btn-submit").addEventListener("click", doSubmit);
el("btn-ask").addEventListener("click", showAsk);
el("btn-status").addEventListener("click", showStatus);
el("btn-save").addEventListener("click", showSave);
el("sheet-close").addEventListener("click", closeSheet);

el("sheet").addEventListener("click", (e) => {
  if (e.target === el("sheet")) closeSheet();
});

document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") closeSheet();
});

document.querySelectorAll(".tab").forEach((b) =>
  b.addEventListener("click", () => {
    state.datasetView = b.dataset.view;
    renderDataset();
  })
);

/* Resume whatever the server already has, so a refresh is never a reset. */
(async function boot() {
  let existing = null;

  try {
    existing = await api("/state");
  } catch (e) {
    fail(e);
  }

  bootEditor(() => {
    if (!existing?.challenge) {
      // Nothing in flight: wait for the button rather than spending a model
      // call because a browser tab happened to open.
      if (existing) apply(existing);
      return;
    }

    apply(existing);
    if (existing.review) renderReview(existing.review);
    else el("output").innerHTML =
      `<p class="hint-line">Bind your result to <code>${state.answerName}</code>. ` +
      `<b>Run</b> is free. <b>Submit</b> costs an attempt.</p>`;
  });
})();
