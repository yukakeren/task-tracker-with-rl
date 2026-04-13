import { useState, useEffect, useMemo, useCallback } from "react";

const API_URL = process.env.REACT_APP_API_URL || "http://localhost:8000";

const CHARACTER_TYPES = [
  { key: "early_grinder", name: "Early Grinder", emoji: "🌅", peak: "6am–10am" },
  { key: "slow_starter", name: "Slow Starter", emoji: "🌙", peak: "2pm–6pm" },
  { key: "sprinter", name: "Sprinter", emoji: "⚡", peak: "Burst 90min" },
  { key: "steady_pacer", name: "Steady Pacer", emoji: "🔄", peak: "9am–3pm" },
  { key: "night_owl", name: "Night Owl", emoji: "🦉", peak: "8pm–1am" },
];

const cogDemandStrToInt = (s) => s === "low" ? 2 : s === "medium" ? 3 : 5;

function deadlineToHour(str) {
  if (!str) return null;
  const [h, m] = str.split(":").map(Number);
  return h + (m || 0) / 60;
}

const STORAGE_KEY = "apuahrls-data-v2";
const saveData = (d) => { try { localStorage.setItem(STORAGE_KEY, JSON.stringify(d)); } catch {} };
const loadData = () => { try { return JSON.parse(localStorage.getItem(STORAGE_KEY)); } catch { return null; } };

const fmt = (s) => {
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
  return `${String(h).padStart(2,"0")}:${String(m).padStart(2,"0")}:${String(sec).padStart(2,"0")}`;
};
const slotToTime = (slot) => {
  const h = Math.floor(slot / 2);
  const m = slot % 2 === 0 ? "00" : "30";
  return `${String(h).padStart(2,"0")}:${m}`;
};
const TOTAL_SLOTS = 48; // Full 24h day, 30-min slots starting from midnight (00:00)
const ENERGY_LABELS = ["", "Exhausted", "Low", "Moderate", "Good", "Peak"];
const DEMAND_COLORS = {
  high: { bg: "rgba(220,38,38,0.15)", border: "#dc2626", text: "#fca5a5", solid: "#dc2626" },
  medium: { bg: "rgba(245,158,11,0.15)", border: "#f59e0b", text: "#fcd34d", solid: "#f59e0b" },
  low: { bg: "rgba(34,197,94,0.15)", border: "#22c55e", text: "#86efac", solid: "#22c55e" },
};
const BLOCK_COLORS = {
  "Deep work": { bg: "rgba(168,85,247,0.08)", border: "#a855f7", text: "#c084fc" },
  "Light work": { bg: "rgba(34,197,94,0.06)", border: "#22c55e", text: "#86efac" },
  "Admin": { bg: "rgba(245,158,11,0.06)", border: "#f59e0b", text: "#fcd34d" },
  "Recovery": { bg: "rgba(99,102,241,0.06)", border: "#6366f1", text: "#a5b4fc" },
};

// Character-specific energy curves based on CatBoost ground truth profiles
const ENERGY_CURVES = {
  early_grinder: (h) => h < 4 ? 0.5 : h < 6 ? 0.7 : h < 8 ? 0.85 : h < 10 ? 0.95 : h < 12 ? 0.75 : h < 13 ? 0.3 : h < 15 ? 0.4 : h < 17 ? 0.35 : h < 19 ? 0.25 : 0.15,
  slow_starter:  (h) => h < 4 ? 0.1 : h < 9 ? 0.15 : h < 12 ? 0.25 : h < 14 ? 0.45 : h < 16 ? 0.8 : h < 18 ? 0.9 : h < 20 ? 0.65 : h < 22 ? 0.4 : 0.2,
  sprinter:      (h) => { const cycle = h % 3; return cycle < 1.5 ? 0.85 : 0.25; },
  steady_pacer:  (h) => h < 4 ? 0.3 : h < 9 ? 0.5 : h < 15 ? 0.68 : h < 17 ? 0.55 : h < 19 ? 0.4 : 0.2,
  night_owl:     (h) => h < 4 ? 0.7 : h < 6 ? 0.4 : h < 10 ? 0.2 : h < 14 ? 0.15 : h < 18 ? 0.2 : h < 20 ? 0.35 : h < 22 ? 0.7 : 0.95,
};

function buildEnergyProfile(energyLevel, characterType = "steady_pacer") {
  const curve = ENERGY_CURVES[characterType] || ENERGY_CURVES.steady_pacer;
  return Array.from({ length: TOTAL_SLOTS }, (_, i) => {
    const hour = i / 2; // Slot 0 = 00:00, Slot 24 = 12:00, Slot 47 = 23:30
    const base = curve(hour);
    return Math.max(0.08, base * (energyLevel / 3));
  });
}

function managerAgent(energyProfile, occupied, currentSlot) {
  const blocks = [];
  const available = [];
  for (let s = 0; s < TOTAL_SLOTS; s++) available.push(!occupied.has(s) && s >= currentSlot);

  let blockStart = -1, runLength = 0;
  for (let s = 0; s <= TOTAL_SLOTS; s++) {
    if (s < TOTAL_SLOTS && available[s]) {
      if (blockStart === -1) blockStart = s;
      runLength++;
    } else {
      if (blockStart !== -1 && runLength >= 2) {
        let pos = blockStart;
        while (pos < blockStart + runLength) {
          const remaining = blockStart + runLength - pos;
          const blockSize = remaining >= 10 ? 8 : remaining >= 6 ? Math.min(8, remaining) : remaining;
          if (blockSize < 2) break;
          const avgEnergy = energyProfile.slice(pos, pos + blockSize).reduce((a, b) => a + b, 0) / blockSize;
          let type;
          if (avgEnergy > 0.7) type = "Deep work";
          else if (avgEnergy > 0.45) type = "Light work";
          else if (avgEnergy > 0.25) type = "Admin";
          else type = "Recovery";
          blocks.push({ type, startSlot: pos, endSlot: pos + blockSize, avgEnergy });
          pos += blockSize;
        }
      }
      blockStart = -1;
      runLength = 0;
    }
  }
  return blocks;
}

function workerAgent(tasks, managerBlocks, energyProfile, occupied) {
  const demandMap = { high: 3, medium: 2, low: 1 };
  const scheduled = [];
  const usedTasks = new Set();
  const localOccupied = new Set(occupied);

  const sortedBlocks = [...managerBlocks].sort((a, b) => b.avgEnergy - a.avgEnergy);
  const sortedTasks = [...tasks].filter(t => !t.is_archived && !t.is_fixed).sort((a, b) => {
    const sa = (a.priority || 3) * demandMap[a.cognitive_demand || "medium"];
    const sb = (b.priority || 3) * demandMap[b.cognitive_demand || "medium"];
    return sb - sa;
  });

  for (const block of sortedBlocks) {
    const candidates = sortedTasks.filter(t => {
      if (usedTasks.has(t.id)) return false;
      const td = t.cognitive_demand || "medium";
      if (block.type === "Deep work") return td === "high" || td === "medium";
      if (block.type === "Light work") return td === "medium" || td === "low";
      if (block.type === "Admin") return td === "low" || td === "medium";
      return td === "low";
    });

    for (const task of candidates) {
      if (usedTasks.has(task.id)) continue;
      const slotsNeeded = Math.max(1, Math.ceil((task.duration_estimate || 30) / 30));
      let bestStart = -1, bestScore = -Infinity;

      for (let s = block.startSlot; s <= block.endSlot - slotsNeeded; s++) {
        let valid = true;
        for (let k = 0; k < slotsNeeded; k++) if (localOccupied.has(s + k)) { valid = false; break; }
        if (!valid) continue;
        if (task.deadline) {
          const parts = task.deadline.split(":");
          const dlSlot = parseInt(parts[0]) * 2 + Math.floor(parseInt(parts[1]) / 30);
          if (s + slotsNeeded > dlSlot) continue;
        }
        let score = 0;
        for (let k = 0; k < slotsNeeded; k++) score += energyProfile[s + k] * demandMap[task.cognitive_demand || "medium"];
        score += (task.priority || 3) * 3;
        if (score > bestScore) { bestScore = score; bestStart = s; }
      }

      if (bestStart >= 0) {
        for (let k = 0; k < slotsNeeded; k++) localOccupied.add(bestStart + k);
        scheduled.push({ ...task, scheduled_start: bestStart, scheduled_slots: slotsNeeded, assigned_block: block.type });
        usedTasks.add(task.id);
      }
    }
  }

  for (const task of sortedTasks) {
    if (usedTasks.has(task.id)) continue;
    const slotsNeeded = Math.max(1, Math.ceil((task.duration_estimate || 30) / 30));
    for (let s = 0; s < TOTAL_SLOTS - slotsNeeded; s++) {
      let valid = true;
      for (let k = 0; k < slotsNeeded; k++) if (localOccupied.has(s + k)) { valid = false; break; }
      if (!valid) continue;
      for (let k = 0; k < slotsNeeded; k++) localOccupied.add(s + k);
      scheduled.push({ ...task, scheduled_start: s, scheduled_slots: slotsNeeded, assigned_block: "Overflow" });
      usedTasks.add(task.id);
      break;
    }
  }
  return scheduled;
}

function generateSchedule(tasks, fixedBlocks, energyLevel, characterType = "steady_pacer") {
  const now = new Date();
  const currentSlot = Math.min(TOTAL_SLOTS - 1, now.getHours() * 2 + (now.getMinutes() >= 30 ? 1 : 0));
  
  const occupied = new Set();
  fixedBlocks.forEach(fb => { for (let s = fb.startSlot; s < fb.endSlot; s++) occupied.add(s); });
  
  // Block out any slots that have already passed today
  for (let s = 0; s < currentSlot; s++) occupied.add(s);
  
  const energyProfile = buildEnergyProfile(energyLevel, characterType);
  const managerBlocks = managerAgent(energyProfile, occupied, currentSlot);
  const scheduled = workerAgent(tasks, managerBlocks, energyProfile, occupied);
  return { scheduled, managerBlocks };
}

const PriorityDots = ({ level }) => (
  <span style={{ display: "inline-flex", gap: 2 }}>
    {[1,2,3,4,5].map(i => <span key={i} style={{ width: 5, height: 5, borderRadius: "50%", background: i <= level ? "#f59e0b" : "#27272a" }} />)}
  </span>
);

export default function APUAHRLS() {
  const [tasks, setTasks] = useState([]);
  const [view, setView] = useState("dashboard");
  const [energyLevel, setEnergyLevel] = useState(3);
  const [schedule, setSchedule] = useState([]);
  const [managerBlocks, setManagerBlocks] = useState([]);
  const [fixedBlocks, setFixedBlocks] = useState([]);
  const [showAddTask, setShowAddTask] = useState(false);
  const [, setTick] = useState(0);
  const [form, setForm] = useState({ title: "", category: "", duration_estimate: 30, priority: 3, cognitive_demand: "medium", deadline: "", is_fixed: false, startTime: "08:00", endTime: "10:00" });
  const [characterType, setCharacterType] = useState("steady_pacer");
  const [nlInput, setNlInput] = useState("");
  const [nlLoading, setNlLoading] = useState(false);
  const [scheduleLoading, setScheduleLoading] = useState(false);
  const [useAI, setUseAI] = useState(true);
  const [backendStatus, setBackendStatus] = useState("unknown");
  const [totalReward, setTotalReward] = useState(0);
  const [toastMsg, setToastMsg] = useState("");
  const [nlError, setNlError] = useState("");

  useEffect(() => { const d = loadData(); if (d) { if (d.tasks) setTasks(d.tasks); if (d.energyLevel) setEnergyLevel(d.energyLevel); if (d.fixedBlocks?.length) setFixedBlocks(d.fixedBlocks); if (d.schedule) setSchedule(d.schedule); if (d.managerBlocks) setManagerBlocks(d.managerBlocks); } }, []);
  useEffect(() => { saveData({ tasks, energyLevel, fixedBlocks, schedule, managerBlocks }); }, [tasks, energyLevel, fixedBlocks, schedule, managerBlocks]);
  useEffect(() => { const iv = setInterval(() => setTick(t => t + 1), 1000); return () => clearInterval(iv); }, []);

  // Health check on mount
  useEffect(() => {
    fetch(`${API_URL}/health`).then(r => r.json()).then(() => setBackendStatus("ok")).catch(() => { setBackendStatus("error"); setUseAI(false); });
  }, []);

  // Auto-dismiss toast
  useEffect(() => { if (toastMsg) { const t = setTimeout(() => setToastMsg(""), 3000); return () => clearTimeout(t); } }, [toastMsg]);

  const showToast = useCallback((msg) => setToastMsg(msg), []);

  const addTask = () => {
    if (!form.title.trim()) return;
    if (form.is_fixed) {
      const sh = parseInt(form.startTime.split(":")[0]), sm = parseInt(form.startTime.split(":")[1] || "0");
      const eh = parseInt(form.endTime.split(":")[0]), em = parseInt(form.endTime.split(":")[1] || "0");
      const startSlot = Math.min(TOTAL_SLOTS - 1, sh * 2 + Math.floor(sm / 30));
      const endSlot = Math.min(TOTAL_SLOTS, eh * 2 + Math.floor(em / 30));
      if (endSlot <= startSlot) return;
      setFixedBlocks(prev => [...prev, { id: crypto.randomUUID(), title: form.title.toUpperCase(), startSlot, endSlot, color: "#4338ca" }]);
    } else {
      setTasks(prev => [{ id: crypto.randomUUID(), ...form, title: form.title.toUpperCase(), category: (form.category || "GENERAL").toUpperCase(), is_running: false, total_duration: 0, last_started_at: null, is_archived: false, is_fixed: false }, ...prev]);
    }
    setForm({ title: "", category: "", duration_estimate: 30, priority: 3, cognitive_demand: "medium", deadline: "", is_fixed: false, startTime: "08:00", endTime: "10:00" });
    setShowAddTask(false);
  };

  const toggleTask = (id) => setTasks(prev => prev.map(t => { if (t.id !== id) return t; const now = Date.now(); if (t.is_running) { const dur = t.last_started_at ? Math.floor((now - t.last_started_at) / 1000) : 0; return { ...t, is_running: false, total_duration: t.total_duration + dur, last_started_at: null }; } return { ...t, is_running: true, last_started_at: now }; }));
  const archiveTask = (id) => setTasks(prev => prev.map(t => t.id === id ? { ...t, is_archived: true, is_running: false } : t));
  const deleteTask = (id) => setTasks(prev => prev.filter(t => t.id !== id));
  const removeFixed = (id) => setFixedBlocks(prev => prev.filter(f => f.id !== id));

  const handleNlParse = async () => {
    if (!nlInput.trim()) return;
    const apiKey = process.env.REACT_APP_GEMINI_API_KEY;
    if (!apiKey) { setNlError("REACT_APP_GEMINI_API_KEY is missing in .env"); return; }
    
    setNlLoading(true); setNlError("");
    try {
      const payload = {
        systemInstruction: { parts: [{ text: "You are an intelligent natural language parser for an adaptive scheduling application.\nYour objective is to extract scheduling information from user input and categorize it.\n\nRules:\n1. 'fixed' schedules are events with explicitly stated start and end times (output as HH:MM like '08:00').\n2. 'tasks' are flexible items to be done.\n3. If duration is not provided, estimate a reasonable default in minutes.\n4. Infer priority (1-5) and cognitive demand (low, medium, high) based on context clues." }] },
        contents: [{ parts: [{ text: nlInput }] }],
        generationConfig: {
          temperature: 0.1,
          responseMimeType: "application/json",
          responseSchema: {
            type: "OBJECT",
            properties: {
              fixed: { type: "ARRAY", items: { type: "OBJECT", properties: { title: { type: "STRING" }, start_time: { type: "STRING" }, end_time: { type: "STRING" } }, required: ["title", "start_time", "end_time"] } },
              tasks: { type: "ARRAY", items: { type: "OBJECT", properties: { title: { type: "STRING" }, duration_mins: { type: "INTEGER" }, priority: { type: "INTEGER" }, demand: { type: "STRING" } }, required: ["title", "duration_mins", "priority", "demand"] } }
            },
            required: ["fixed", "tasks"]
          }
        }
      };

      const res = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=${apiKey}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      
      if (!res.ok) throw new Error("Parse failed");
      const apiRes = await res.json();
      const textResponse = apiRes.candidates?.[0]?.content?.parts?.[0]?.text;
      if (!textResponse) throw new Error("Empty response");
      
      const data = JSON.parse(textResponse);
      const newFixed = []; const newTasks = [];
      
      for (const e of data.fixed || []) {
        const sh = parseInt(e.start_time.split(":")[0]), sm = parseInt(e.start_time.split(":")[1] || "0");
        const eh = parseInt(e.end_time.split(":")[0]), em = parseInt(e.end_time.split(":")[1] || "0");
        const startSlot = Math.min(TOTAL_SLOTS - 1, sh * 2 + Math.floor(sm / 30));
        const endSlot = Math.min(TOTAL_SLOTS, eh * 2 + Math.floor(em / 30));
        if (endSlot > startSlot) {
          newFixed.push({ id: crypto.randomUUID(), title: (e.title || "EVENT").toUpperCase(), startSlot, endSlot, color: "#4338ca" });
        }
      }
      
      for (const e of data.tasks || []) {
        newTasks.push({ id: crypto.randomUUID(), title: (e.title || "TASK").toUpperCase(), category: "GENERAL", duration_estimate: e.duration_mins || 30, priority: e.priority || 3, cognitive_demand: ["low","medium","high"].includes(e.demand) ? e.demand : "medium", deadline: "", is_running: false, total_duration: 0, last_started_at: null, is_archived: false, is_fixed: false });
      }
      
      if (newFixed.length) setFixedBlocks(prev => [...prev, ...newFixed]);
      if (newTasks.length) setTasks(prev => [...newTasks, ...prev]);
      setNlInput("");
    } catch (e) { setNlError("Failed to parse: " + e.message); }
    setNlLoading(false);
  };

  const doGenerate = async () => {
    if (useAI && backendStatus === "ok") {
      setScheduleLoading(true);
      try {
        const taskPayload = activeTasks.filter(t => !t.is_fixed).map(t => ({ id: t.id, title: t.title, duration_minutes: t.duration_estimate || 30, priority: t.priority || 3, cognitive_demand: cogDemandStrToInt(t.cognitive_demand || "medium"), deadline_hour: deadlineToHour(t.deadline) }));
        const fixedPayload = fixedBlocks.map(fb => ({ start_hour: fb.startSlot / 2, end_hour: fb.endSlot / 2, title: fb.title }));
        const res = await fetch(`${API_URL}/schedule`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ character_type: characterType, tasks: taskPayload, fixed_blocks: fixedPayload, current_hour: null, energy_override: null }) });
        if (!res.ok) throw new Error("Schedule failed");
        const data = await res.json();
        const mapped = (data.scheduled || []).map(item => {
          const orig = tasks.find(t => t.id === item.id) || {};
          return { ...orig, ...item, scheduled_start: Math.round(item.start_hour * 2), scheduled_slots: Math.round((item.end_hour - item.start_hour) * 2), energy_at_start: item.energy_at_start, reward: item.reward, assigned_block: "AI" };
        });
        setSchedule(mapped); setManagerBlocks([]); setTotalReward(data.total_reward || 0); setView("schedule");
      } catch {
        showToast("AI unavailable, using local scheduler");
        const r = generateSchedule(tasks, fixedBlocks, energyLevel, characterType); setSchedule(r.scheduled); setManagerBlocks(r.managerBlocks); setTotalReward(0); setView("schedule");
      }
      setScheduleLoading(false);
    } else {
      const r = generateSchedule(tasks, fixedBlocks, energyLevel, characterType); setSchedule(r.scheduled); setManagerBlocks(r.managerBlocks); setTotalReward(0); setView("schedule");
    }
  };

  const activeTasks = tasks.filter(t => !t.is_archived);
  const archivedTasks = tasks.filter(t => t.is_archived);
  const runningTask = tasks.find(t => t.is_running);
  const getElapsed = (t) => !t.is_running || !t.last_started_at ? t.total_duration : t.total_duration + Math.floor((Date.now() - t.last_started_at) / 1000);

  const energyProfile = useMemo(() => buildEnergyProfile(energyLevel, characterType), [energyLevel, characterType]);
  const maxEnergy = Math.max(...energyProfile);
  const nowSlot = new Date().getHours() * 2 + (new Date().getMinutes() >= 30 ? 1 : 0);

  const S = {
    app: { display: "flex", height: "100vh", fontFamily: "'JetBrains Mono','SF Mono','Fira Code',monospace", background: "#09090b", color: "#e4e4e7", overflow: "hidden", fontSize: 13 },
    sidebar: { width: 220, minWidth: 220, background: "#0c0c0f", borderRight: "1px solid rgba(255,255,255,0.06)", display: "flex", flexDirection: "column", justifyContent: "space-between" },
    main: { flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" },
    topbar: { padding: "16px 24px", borderBottom: "1px solid rgba(255,255,255,0.06)", background: "rgba(9,9,11,0.8)", backdropFilter: "blur(12px)", display: "flex", alignItems: "center", justifyContent: "space-between", gap: 16, flexWrap: "wrap" },
    content: { flex: 1, overflow: "auto", padding: 24 },
    card: { background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.06)", borderRadius: 8, padding: "12px 16px", marginBottom: 8, display: "flex", alignItems: "center", justifyContent: "space-between", transition: "all 0.2s" },
    btn: { padding: "6px 14px", borderRadius: 6, border: "1px solid rgba(255,255,255,0.1)", background: "rgba(255,255,255,0.04)", color: "#a1a1aa", cursor: "pointer", fontSize: 12, fontFamily: "inherit" },
    btnP: { padding: "8px 18px", borderRadius: 6, border: "1px solid rgba(16,185,129,0.3)", background: "rgba(16,185,129,0.12)", color: "#34d399", cursor: "pointer", fontSize: 12, fontFamily: "inherit", fontWeight: 600 },
    input: { padding: "8px 12px", borderRadius: 6, border: "1px solid rgba(255,255,255,0.1)", background: "rgba(255,255,255,0.04)", color: "#e4e4e7", fontSize: 12, fontFamily: "inherit", outline: "none", width: "100%", boxSizing: "border-box" },
    select: { padding: "8px 12px", borderRadius: 6, border: "1px solid rgba(255,255,255,0.1)", background: "#18181b", color: "#e4e4e7", fontSize: 12, fontFamily: "inherit", outline: "none" },
    badge: (c) => ({ display: "inline-block", padding: "2px 8px", borderRadius: 4, fontSize: 10, fontWeight: 600, background: c.bg, color: c.text, border: `1px solid ${c.border}30` }),
    dot: (on) => ({ width: 8, height: 8, borderRadius: "50%", background: on ? "#34d399" : "#3f3f46", flexShrink: 0, boxShadow: on ? "0 0 8px rgba(52,211,153,0.5)" : "none" }),
  };

  const NavItem = ({ label, icon, active, onClick }) => (
    <button onClick={onClick} style={{ display: "flex", alignItems: "center", gap: 10, padding: "10px 20px", width: "100%", background: active ? "rgba(255,255,255,0.04)" : "transparent", borderLeft: `2px solid ${active ? "#34d399" : "transparent"}`, color: active ? "#34d399" : "#71717a", border: "none", cursor: "pointer", fontFamily: "inherit", fontSize: 12, fontWeight: active ? 600 : 400, textAlign: "left", borderRight: "none", borderTop: "none", borderBottom: "none", borderLeftWidth: 2, borderLeftStyle: "solid", borderLeftColor: active ? "#34d399" : "transparent" }}>
      <span style={{ fontSize: 14 }}>{icon}</span>{label}
    </button>
  );

  return (
    <div style={S.app}>
      <aside style={S.sidebar}>
        <div>
          <div style={{ padding: "20px 20px 16px", borderBottom: "1px solid rgba(255,255,255,0.04)" }}>
            <div style={{ fontSize: 18, fontWeight: 700, letterSpacing: -0.5, color: "#f4f4f5" }}>APUAHRLS</div>
            <div style={{ fontSize: 9, color: "#52525b", letterSpacing: 1.5, marginTop: 2 }}>ADAPTIVE SCHEDULER</div>
          </div>
          <div style={{ padding: "12px 0" }}>
            <div style={{ padding: "0 20px", marginBottom: 8, fontSize: 9, color: "#3f3f46", letterSpacing: 2, fontWeight: 600 }}>MODULES</div>
            <NavItem label="Dashboard" icon="◈" active={view === "dashboard"} onClick={() => setView("dashboard")} />
            <NavItem label="Schedule" icon="◫" active={view === "schedule"} onClick={() => setView("schedule")} />
            <NavItem label="Archive" icon="◰" active={view === "archive"} onClick={() => setView("archive")} />
          </div>
          {managerBlocks.length > 0 && (
            <div style={{ padding: "0 20px", marginTop: 8 }}>
              <div style={{ fontSize: 9, color: "#3f3f46", letterSpacing: 2, fontWeight: 600, marginBottom: 8 }}>MANAGER AGENT</div>
              {managerBlocks.map((b, i) => { const bc = BLOCK_COLORS[b.type] || BLOCK_COLORS["Admin"]; return (
                <div key={i} style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 4, fontSize: 10, color: bc.text }}>
                  <div style={{ width: 8, height: 8, borderRadius: 2, background: bc.border, opacity: 0.7 }} />
                  <span style={{ fontWeight: 600 }}>{b.type}</span>
                  <span style={{ color: "#3f3f46" }}>{slotToTime(b.startSlot)}-{slotToTime(b.endSlot)}</span>
                </div>
              ); })}
            </div>
          )}
        </div>
        <div style={{ padding: 16, borderTop: "1px solid rgba(255,255,255,0.04)" }}>
          <div style={{ fontSize: 9, color: "#3f3f46", letterSpacing: 2, fontWeight: 600, marginBottom: 10 }}>ENERGY STATE</div>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 6 }}>
            <span style={{ fontSize: 20 }}>{["","😵","😮‍💨","😐","😊","⚡"][energyLevel]}</span>
            <span style={{ fontSize: 11, color: "#a1a1aa", fontWeight: 600 }}>{ENERGY_LABELS[energyLevel]}</span>
          </div>
          <input type="range" min={1} max={5} value={energyLevel} onChange={e => setEnergyLevel(+e.target.value)} style={{ width: "100%", accentColor: "#34d399" }} />
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: 9, color: "#3f3f46", marginTop: 2 }}><span>1</span><span>2</span><span>3</span><span>4</span><span>5</span></div>
          {runningTask && (
            <div style={{ marginTop: 12, padding: "8px 10px", borderRadius: 6, background: "rgba(52,211,153,0.08)", border: "1px solid rgba(52,211,153,0.2)" }}>
              <div style={{ fontSize: 9, color: "#34d399", letterSpacing: 1, marginBottom: 4 }}>▶ ACTIVE</div>
              <div style={{ fontSize: 11, color: "#d4d4d8", fontWeight: 600 }}>{runningTask.title}</div>
              <div style={{ fontSize: 16, color: "#34d399", fontWeight: 700, marginTop: 2, fontVariantNumeric: "tabular-nums" }}>{fmt(getElapsed(runningTask))}</div>
            </div>
          )}
          <div style={{ marginTop: 16 }}>
            <div style={{ fontSize: 9, color: "#3f3f46", letterSpacing: 2, fontWeight: 600, marginBottom: 8 }}>CHARACTER PROFILE</div>
            {CHARACTER_TYPES.map(ct => (
              <button key={ct.key} onClick={() => setCharacterType(ct.key)} style={{ display: "flex", alignItems: "center", gap: 6, width: "100%", padding: "6px 8px", marginBottom: 3, borderRadius: 6, border: characterType === ct.key ? "1px solid rgba(52,211,153,0.4)" : "1px solid transparent", background: characterType === ct.key ? "rgba(52,211,153,0.08)" : "transparent", color: characterType === ct.key ? "#34d399" : "#71717a", cursor: "pointer", fontFamily: "inherit", fontSize: 10, fontWeight: characterType === ct.key ? 600 : 400, textAlign: "left" }}>
                <span style={{ fontSize: 14 }}>{ct.emoji}</span>
                <span>{ct.name}</span>
              </button>
            ))}
          </div>
          {backendStatus === "error" && <div style={{ marginTop: 10, padding: "6px 8px", borderRadius: 4, background: "rgba(239,68,68,0.1)", border: "1px solid rgba(239,68,68,0.2)", fontSize: 9, color: "#fca5a5", letterSpacing: 0.5 }}>⚠ Backend not connected</div>}
        </div>
      </aside>

      <div style={S.main}>
        <div style={S.topbar}>
          <div>
            <div style={{ fontSize: 9, color: "#52525b", letterSpacing: 2, marginBottom: 2 }}>SYS / {view.toUpperCase()}</div>
            <div style={{ fontSize: 18, fontWeight: 700, color: "#f4f4f5", letterSpacing: -0.5 }}>{view === "dashboard" ? "Active tasks" : view === "schedule" ? "Daily schedule" : "Completed tasks"}</div>
          </div>
          <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
            {view === "dashboard" && (<>
              <button style={S.btn} onClick={() => { setShowAddTask(true); setForm(f => ({...f, is_fixed: false})); }}>+ Task</button>
              <button style={{...S.btn, color: "#818cf8", borderColor: "rgba(99,102,241,0.3)"}} onClick={() => { setShowAddTask(true); setForm(f => ({...f, is_fixed: true})); }}>+ Fixed block</button>
              <div style={{ display: "flex", borderRadius: 6, border: "1px solid rgba(255,255,255,0.1)", overflow: "hidden" }}>
                <button onClick={() => setUseAI(true)} style={{ padding: "6px 10px", fontSize: 10, fontFamily: "inherit", border: "none", cursor: "pointer", background: useAI ? "rgba(52,211,153,0.15)" : "transparent", color: useAI ? "#34d399" : "#71717a", fontWeight: useAI ? 600 : 400 }}>AI (DQN)</button>
                <button onClick={() => setUseAI(false)} style={{ padding: "6px 10px", fontSize: 10, fontFamily: "inherit", border: "none", borderLeft: "1px solid rgba(255,255,255,0.1)", cursor: "pointer", background: !useAI ? "rgba(245,158,11,0.15)" : "transparent", color: !useAI ? "#f59e0b" : "#71717a", fontWeight: !useAI ? 600 : 400 }}>Heuristic</button>
              </div>
              <button style={scheduleLoading ? {...S.btnP, opacity: 0.6} : S.btnP} onClick={doGenerate} disabled={scheduleLoading}>{scheduleLoading ? "⏳ Scheduling..." : useAI ? "◈ AI Schedule" : "◈ Generate schedule"}</button>
            </>)}
            {view === "schedule" && <button style={S.btnP} onClick={doGenerate}>↻ Regenerate</button>}
          </div>
        </div>

        <div style={S.content}>
          {view === "dashboard" && (
            <div style={{ marginBottom: 16, padding: "12px 16px", borderRadius: 8, background: "rgba(255,255,255,0.03)", border: nlLoading ? "1px solid rgba(52,211,153,0.4)" : "1px solid rgba(255,255,255,0.06)", animation: nlLoading ? "pulse 1.5s ease-in-out infinite" : "none" }}>
              <div style={{ fontSize: 9, color: "#3f3f46", letterSpacing: 2, fontWeight: 600, marginBottom: 8 }}>NATURAL LANGUAGE INPUT</div>
              <div style={{ display: "flex", gap: 8 }}>
                <textarea rows={2} placeholder='Describe your day... (e.g. ada kelas jam 12, tugas ML deadline 23:59)' value={nlInput} onChange={e => setNlInput(e.target.value)} style={{ ...S.input, resize: "none" }} onKeyDown={e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleNlParse(); } }} />
                <button onClick={handleNlParse} disabled={nlLoading || !nlInput.trim()} style={{ ...S.btnP, minWidth: 100, opacity: nlLoading || !nlInput.trim() ? 0.5 : 1 }}>{nlLoading ? "⏳ Parsing..." : "Parse with AI"}</button>
              </div>
              {nlError && <div style={{ marginTop: 6, fontSize: 10, color: "#fca5a5" }}>{nlError}</div>}
            </div>
          )}

          {showAddTask && view === "dashboard" && (
            <div style={{ ...S.card, flexDirection: "column", alignItems: "stretch", gap: 12, marginBottom: 16, border: `1px solid ${form.is_fixed ? "rgba(99,102,241,0.3)" : "rgba(52,211,153,0.2)"}`, background: form.is_fixed ? "rgba(99,102,241,0.04)" : "rgba(52,211,153,0.03)" }}>
              <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                <button onClick={() => setForm(f => ({...f, is_fixed: false}))} style={{ ...S.btn, background: !form.is_fixed ? "rgba(52,211,153,0.15)" : "transparent", color: !form.is_fixed ? "#34d399" : "#71717a" }}>Flexible task</button>
                <button onClick={() => setForm(f => ({...f, is_fixed: true}))} style={{ ...S.btn, background: form.is_fixed ? "rgba(99,102,241,0.15)" : "transparent", color: form.is_fixed ? "#818cf8" : "#71717a" }}>Fixed block</button>
              </div>
              <div style={{ display: "grid", gridTemplateColumns: form.is_fixed ? "2fr 1fr 1fr" : "2fr 1fr", gap: 8 }}>
                <input placeholder={form.is_fixed ? "Block name (e.g. Network Programming Class)" : "Task title..."} value={form.title} onChange={e => setForm({...form, title: e.target.value})} style={S.input} onKeyDown={e => e.key === "Enter" && addTask()} />
                {form.is_fixed ? (<>
                  <div><label style={{ fontSize: 9, color: "#52525b" }}>START</label><input type="time" value={form.startTime} onChange={e => setForm({...form, startTime: e.target.value})} style={S.input} /></div>
                  <div><label style={{ fontSize: 9, color: "#52525b" }}>END</label><input type="time" value={form.endTime} onChange={e => setForm({...form, endTime: e.target.value})} style={S.input} /></div>
                </>) : <input placeholder="Category" value={form.category} onChange={e => setForm({...form, category: e.target.value})} style={S.input} />}
              </div>
              {!form.is_fixed && (
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr 1fr", gap: 8 }}>
                  <div><label style={{ fontSize: 9, color: "#52525b", letterSpacing: 1 }}>DURATION (min)</label><input type="number" min={5} step={5} value={form.duration_estimate} onChange={e => setForm({...form, duration_estimate: +e.target.value})} style={S.input} /></div>
                  <div><label style={{ fontSize: 9, color: "#52525b", letterSpacing: 1 }}>PRIORITY</label><select value={form.priority} onChange={e => setForm({...form, priority: +e.target.value})} style={{...S.select, width: "100%"}}>{[1,2,3,4,5].map(i => <option key={i} value={i}>{i} - {["Lowest","Low","Medium","High","Critical"][i-1]}</option>)}</select></div>
                  <div><label style={{ fontSize: 9, color: "#52525b", letterSpacing: 1 }}>COGNITIVE DEMAND</label><select value={form.cognitive_demand} onChange={e => setForm({...form, cognitive_demand: e.target.value})} style={{...S.select, width: "100%"}}><option value="low">Low</option><option value="medium">Medium</option><option value="high">High</option></select></div>
                  <div><label style={{ fontSize: 9, color: "#52525b", letterSpacing: 1 }}>DEADLINE</label><input type="time" value={form.deadline} onChange={e => setForm({...form, deadline: e.target.value})} style={S.input} /></div>
                </div>
              )}
              <div style={{ display: "flex", gap: 8, justifyContent: "flex-end" }}>
                <button style={S.btn} onClick={() => setShowAddTask(false)}>Cancel</button>
                <button style={form.is_fixed ? {...S.btnP, borderColor: "rgba(99,102,241,0.3)", background: "rgba(99,102,241,0.15)", color: "#818cf8"} : S.btnP} onClick={addTask}>{form.is_fixed ? "Add fixed block" : "Add task"}</button>
              </div>
            </div>
          )}

          {view === "dashboard" && (<>
            {fixedBlocks.length > 0 && (
              <div style={{ marginBottom: 16 }}>
                <div style={{ fontSize: 9, color: "#3f3f46", letterSpacing: 2, fontWeight: 600, marginBottom: 8 }}>FIXED BLOCKS</div>
                <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                  {fixedBlocks.map(fb => (
                    <div key={fb.id} style={{ display: "inline-flex", alignItems: "center", gap: 6, padding: "4px 10px", borderRadius: 6, background: "rgba(99,102,241,0.08)", border: "1px solid rgba(99,102,241,0.2)", fontSize: 11, color: "#a5b4fc" }}>
                      <span style={{ fontWeight: 600 }}>{fb.title}</span>
                      <span style={{ color: "#6366f1", fontSize: 10 }}>{slotToTime(fb.startSlot)}-{slotToTime(fb.endSlot)}</span>
                      <button onClick={() => removeFixed(fb.id)} style={{ background: "none", border: "none", color: "#4338ca", cursor: "pointer", fontSize: 10, fontFamily: "inherit", padding: 0 }}>✕</button>
                    </div>
                  ))}
                </div>
              </div>
            )}
            {activeTasks.length === 0 && <div style={{ textAlign: "center", padding: "80px 0", color: "#3f3f46" }}><div style={{ fontSize: 32, marginBottom: 8 }}>◇</div><div style={{ fontSize: 11, letterSpacing: 2 }}>NO ACTIVE TASKS</div></div>}
            {activeTasks.map(task => { const elapsed = getElapsed(task); const dc = DEMAND_COLORS[task.cognitive_demand || "medium"]; return (
              <div key={task.id} style={{ ...S.card, borderColor: task.is_running ? "rgba(52,211,153,0.3)" : undefined, boxShadow: task.is_running ? "0 0 20px rgba(52,211,153,0.06)" : "none" }}>
                <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                  <div style={S.dot(task.is_running)} />
                  <div>
                    <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
                      <span style={{ fontWeight: 700, color: task.is_running ? "#f4f4f5" : "#a1a1aa", fontSize: 13 }}>{task.title}</span>
                      <span style={S.badge(dc)}>{task.cognitive_demand || "med"}</span>
                      <PriorityDots level={task.priority || 3} />
                    </div>
                    <div style={{ display: "flex", gap: 12, marginTop: 4, fontSize: 10, color: "#52525b" }}>
                      <span>{task.category}</span><span>{task.duration_estimate}min</span>
                      {task.deadline && <span>⏰ {task.deadline}</span>}
                    </div>
                  </div>
                </div>
                <div style={{ display: "flex", alignItems: "center", gap: 10, flexShrink: 0 }}>
                  <span style={{ fontVariantNumeric: "tabular-nums", fontSize: 16, fontWeight: 700, color: task.is_running ? "#34d399" : "#52525b", minWidth: 80, textAlign: "right" }}>{fmt(elapsed)}</span>
                  <button onClick={() => toggleTask(task.id)} style={{ ...S.btn, minWidth: 70, textAlign: "center", background: task.is_running ? "rgba(52,211,153,0.15)" : undefined, color: task.is_running ? "#34d399" : "#a1a1aa" }}>{task.is_running ? "STOP" : "START"}</button>
                  <button onClick={() => archiveTask(task.id)} style={{ ...S.btn, color: "#22c55e", padding: "6px 10px" }}>✓</button>
                  <button onClick={() => deleteTask(task.id)} style={{ ...S.btn, color: "#ef4444", padding: "6px 10px" }}>✕</button>
                </div>
              </div>
            ); })}
          </>)}

          {view === "schedule" && (
            <div style={{ display: "grid", gridTemplateColumns: "50px 1fr 60px", gap: 0 }}>
              <div style={{ fontSize: 9, color: "#3f3f46", padding: "0 0 8px", letterSpacing: 1 }}>TIME</div>
              <div style={{ fontSize: 9, color: "#3f3f46", padding: "0 0 8px", letterSpacing: 1 }}>TASKS</div>
              <div style={{ fontSize: 9, color: "#3f3f46", padding: "0 0 8px", letterSpacing: 1, textAlign: "center" }}>ENERGY</div>
              {Array.from({ length: TOTAL_SLOTS }, (_, slot) => {
                const time = slotToTime(slot); const isHour = slot % 2 === 0;
                const fixedHere = fixedBlocks.find(fb => slot >= fb.startSlot && slot < fb.endSlot);
                const scheduledHere = schedule.find(s => slot >= s.scheduled_start && slot < s.scheduled_start + s.scheduled_slots);
                const isFixedStart = fixedHere && slot === fixedHere.startSlot;
                const isSchedStart = scheduledHere && slot === scheduledHere.scheduled_start;
                const ep = energyProfile[slot]; const barH = (ep / maxEnergy) * 100;
                const isCurrent = slot === nowSlot; const isPast = slot < nowSlot;
                const mBlock = managerBlocks.find(b => slot >= b.startSlot && slot < b.endSlot);
                const mBlockStart = mBlock && slot === mBlock.startSlot;
                const mbc = mBlock ? (BLOCK_COLORS[mBlock.type] || BLOCK_COLORS["Admin"]) : null;

                return (
                  <div key={slot} style={{ display: "contents" }}>
                    <div style={{ fontSize: 10, color: isCurrent ? "#34d399" : isHour ? "#52525b" : "#27272a", padding: "4px 0", borderTop: isHour ? "1px solid rgba(255,255,255,0.04)" : "none", fontWeight: isCurrent ? 700 : 400, opacity: isPast ? 0.35 : 1 }}>{isHour ? time : ""}</div>
                    <div style={{ padding: "2px 0", borderTop: isHour ? "1px solid rgba(255,255,255,0.04)" : "none", minHeight: 28, position: "relative", background: isCurrent ? "rgba(52,211,153,0.03)" : isPast ? "rgba(0,0,0,0.15)" : "transparent", borderLeft: mbc ? `2px solid ${mbc.border}25` : "2px solid transparent" }}>
                      {isCurrent && <div style={{ position: "absolute", left: 0, top: 0, bottom: 0, width: 2, background: "#34d399", borderRadius: 1 }} />}
                      {mBlockStart && <div style={{ position: "absolute", top: 2, right: 8, fontSize: 9, color: mbc.text, opacity: 0.5, letterSpacing: 1, fontWeight: 600 }}>{mBlock.type.toUpperCase()}</div>}
                      {isFixedStart && (
                        <div style={{ background: "rgba(99,102,241,0.1)", border: "1px solid rgba(99,102,241,0.25)", borderRadius: 6, padding: "6px 10px", height: (fixedHere.endSlot - fixedHere.startSlot) * 28 - 4, display: "flex", alignItems: "flex-start", justifyContent: "space-between" }}>
                          <div><span style={{ fontSize: 10, color: "#818cf8", fontWeight: 700 }}>⬒ {fixedHere.title}</span><span style={{ fontSize: 9, color: "#4338ca", marginLeft: 8 }}>{slotToTime(fixedHere.startSlot)}-{slotToTime(fixedHere.endSlot)}</span></div>
                        </div>
                      )}
                      {isSchedStart && (() => { const dc = DEMAND_COLORS[scheduledHere.cognitive_demand || "medium"]; return (
                        <div onClick={() => toggleTask(scheduledHere.id)} style={{ background: dc.bg, border: `1px solid ${dc.border}30`, borderRadius: 6, padding: "6px 10px", height: scheduledHere.scheduled_slots * 28 - 4, display: "flex", flexDirection: "column", cursor: "pointer", borderLeftWidth: 3, borderLeftColor: dc.border, position: "relative" }}>
                          <div style={{ display: "flex", alignItems: "center", gap: 6, flexWrap: "wrap" }}>
                            <span style={{ fontSize: 11, fontWeight: 700, color: dc.text }}>{scheduledHere.title}</span>
                            <span style={S.badge(dc)}>{scheduledHere.cognitive_demand}</span>
                            <PriorityDots level={scheduledHere.priority || 3} />
                            {scheduledHere.assigned_block && <span style={{ fontSize: 9, color: "#52525b" }}>({scheduledHere.assigned_block})</span>}
                            {scheduledHere.energy_at_start != null && <span style={{ width: 7, height: 7, borderRadius: "50%", display: "inline-block", background: scheduledHere.energy_at_start > 0.6 ? "#22c55e" : scheduledHere.energy_at_start > 0.3 ? "#f59e0b" : "#ef4444", boxShadow: `0 0 4px ${scheduledHere.energy_at_start > 0.6 ? "rgba(34,197,94,0.5)" : scheduledHere.energy_at_start > 0.3 ? "rgba(245,158,11,0.5)" : "rgba(239,68,68,0.5)"}` }} />}
                          </div>
                          <div style={{ fontSize: 9, color: "#52525b", marginTop: 2 }}>{slotToTime(scheduledHere.scheduled_start)}-{slotToTime(scheduledHere.scheduled_start + scheduledHere.scheduled_slots)} · {scheduledHere.duration_estimate}min · {scheduledHere.category}</div>
                          {scheduledHere.reward != null && <span style={{ position: "absolute", top: 4, right: 8, fontSize: 9, fontWeight: 600, color: scheduledHere.reward >= 0 ? "#34d399" : "#ef4444" }}>{scheduledHere.reward >= 0 ? "+" : ""}{scheduledHere.reward.toFixed(2)}</span>}
                        </div>
                      ); })()}
                    </div>
                    <div style={{ padding: "4px 8px", borderTop: isHour ? "1px solid rgba(255,255,255,0.04)" : "none", display: "flex", alignItems: "center" }}>
                      <div style={{ width: "100%", height: 6, background: "#18181b", borderRadius: 3, overflow: "hidden" }}>
                        <div style={{ width: `${barH}%`, height: "100%", borderRadius: 3, background: ep / maxEnergy > 0.7 ? "#34d399" : ep / maxEnergy > 0.4 ? "#f59e0b" : "#ef4444", opacity: isPast ? 0.25 : 1 }} />
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          )}

          {view === "archive" && (<>
            {archivedTasks.length === 0 && <div style={{ textAlign: "center", padding: "80px 0", color: "#3f3f46" }}><div style={{ fontSize: 32, marginBottom: 8 }}>◰</div><div style={{ fontSize: 11, letterSpacing: 2 }}>NO COMPLETED TASKS</div></div>}
            {archivedTasks.map(task => (
              <div key={task.id} style={{ ...S.card, opacity: 0.6 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 10 }}><span style={{ color: "#22c55e", fontSize: 14 }}>✓</span><div><span style={{ fontWeight: 600, color: "#71717a", textDecoration: "line-through", fontSize: 12 }}>{task.title}</span><div style={{ fontSize: 10, color: "#3f3f46", marginTop: 2 }}>{task.category} · {fmt(task.total_duration)}</div></div></div>
                <button onClick={() => deleteTask(task.id)} style={{ ...S.btn, color: "#ef4444", padding: "4px 8px", fontSize: 11 }}>✕</button>
              </div>
            ))}
          </>)}
        </div>

        <div style={{ padding: "6px 24px", borderTop: "1px solid rgba(255,255,255,0.04)", display: "flex", justifyContent: "space-between", fontSize: 10, color: "#3f3f46" }}>
          <span>{activeTasks.length} active · {archivedTasks.length} archived · {schedule.length} scheduled · {managerBlocks.length} blocks</span>
          <span>Energy: {ENERGY_LABELS[energyLevel]} ({energyLevel}/5) · Character: {(CHARACTER_TYPES.find(c => c.key === characterType) || {}).name || characterType} · Algorithm: {useAI ? "DQN (AI)" : "Manager-Worker HRL"}{useAI && totalReward ? ` · Reward: +${totalReward.toFixed(2)}` : ""}</span>
        </div>
      </div>
      {toastMsg && <div style={{ position: "fixed", bottom: 24, right: 24, padding: "10px 18px", borderRadius: 8, background: "rgba(30,30,30,0.95)", border: "1px solid rgba(255,255,255,0.1)", color: "#fbbf24", fontSize: 12, fontFamily: "inherit", zIndex: 999, boxShadow: "0 4px 20px rgba(0,0,0,0.5)", backdropFilter: "blur(8px)" }}>{toastMsg}</div>}
      <style>{`@keyframes pulse { 0%,100% { border-color: rgba(52,211,153,0.2); } 50% { border-color: rgba(52,211,153,0.6); } }`}</style>
    </div>
  );
}
