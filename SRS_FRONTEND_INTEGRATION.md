# SRS: APUAHRLS Frontend Integration with FastAPI Backend

## 1. Overview

The existing React app (`APUAHRLS.jsx`) currently uses a client-side heuristic scheduler (Manager-Worker agents). This SRS describes changes to integrate it with a FastAPI backend that provides:

- **LLM-powered natural language task parsing** (Gemini)
- **CatBoost energy prediction** per character profile
- **DQN-based scheduling** (trained RL model, replaces heuristic scheduler)

The backend is already built and running at `http://localhost:8000`. The frontend needs to be updated to call it.

---

## 2. Backend API Contract

Base URL: `http://localhost:8000` (configurable via env var `REACT_APP_API_URL`)

### 2.1 `POST /parse`

Parses natural language input into structured tasks.

**Request:**
```json
{
  "user_input": "Aku ada kelas jam 12-13, harus belajar presentasi, nge gym jam 15-16...",
  "categories": ["Education", "Health", "Errands", "Work", "Leisure"]
}
```

**Response:**
```json
{
  "entries": [
    {
      "timestamp": "2026-04-11T17:03:21.622449",
      "number_key": 101,
      "type": "fixed",
      "title": "Kelas",
      "category": "Education",
      "duration": "1h",
      "priority": 4,
      "cognitive_demand": 3,
      "deadline": null,
      "start": "2026-04-11T12:00:00",
      "end": "2026-04-11T13:00:00"
    },
    {
      "type": "flexible",
      "title": "Belajar Presentasi",
      "category": "Education",
      "duration": "1h30m",
      "priority": 5,
      "cognitive_demand": 5,
      "deadline": "2026-04-11T12:00:00",
      "start": null,
      "end": null
    }
  ],
  "energy_forecast": [
    { "time": "11:00", "potential_energy_level": -1 },
    { "time": "17:00", "potential_energy_level": 1 }
  ]
}
```

### 2.2 `POST /energy`

Predicts energy level using CatBoost when user doesn't specify mood.

**Request:**
```json
{
  "character_type": "early_grinder",
  "time_of_day": 14.5,
  "tasks_completed": 3,
  "cognitive_load_spent": 2.0
}
```

**Response:**
```json
{
  "energy": 0.42,
  "character_type": "early_grinder"
}
```

### 2.3 `POST /schedule`

Generates optimal task schedule using the trained DQN model.

**Request:**
```json
{
  "character_type": "early_grinder",
  "tasks": [
    {
      "id": "uuid-1",
      "title": "Belajar Presentasi",
      "duration_minutes": 90,
      "priority": 5,
      "cognitive_demand": 5,
      "deadline_hour": 12.0
    },
    {
      "id": "uuid-2",
      "title": "Beli Baygon",
      "duration_minutes": 30,
      "priority": 2,
      "cognitive_demand": 1,
      "deadline_hour": null
    }
  ],
  "fixed_blocks": [
    { "start_hour": 12.0, "end_hour": 13.0, "title": "Kelas" },
    { "start_hour": 15.0, "end_hour": 16.0, "title": "Gym" }
  ],
  "current_hour": null,
  "energy_override": null
}
```

**Response:**
```json
{
  "scheduled": [
    {
      "id": "uuid-1",
      "title": "Belajar Presentasi",
      "start_hour": 9.0,
      "end_hour": 10.5,
      "priority": 5,
      "cognitive_demand": 5,
      "energy_at_start": 0.92,
      "reward": 1.35
    },
    {
      "id": "uuid-2",
      "title": "Beli Baygon",
      "start_hour": 13.0,
      "end_hour": 13.5,
      "priority": 2,
      "cognitive_demand": 1,
      "energy_at_start": 0.38,
      "reward": 0.45
    }
  ],
  "total_reward": 1.80,
  "character_type": "early_grinder"
}
```

### 2.4 `GET /health`

Returns server status and loaded models. No request body needed.

---

## 3. Character Types

The system supports 5 fixed character types. Each has a unique energy curve and behavioral parameters.

| Key | Display Name | Chronotype | Peak Window |
|-----|-------------|------------|-------------|
| `early_grinder` | Early Grinder | Morning | 6am–10am |
| `slow_starter` | Slow Starter | Evening | 2pm–6pm |
| `sprinter` | Sprinter | Intermediate | Burst 90min on/off |
| `steady_pacer` | Steady Pacer | Intermediate | 9am–3pm flat |
| `night_owl` | Night Owl | Night | 8pm–1am |

---

## 4. Changes Required to `APUAHRLS.jsx`

### 4.1 New State Variables

Add these to the main component:

```
characterType        — string, one of the 5 character keys. Default: "steady_pacer"
nlInput              — string, the natural language text input
nlLoading            — boolean, loading state while Gemini parses
scheduleLoading      — boolean, loading state while DQN generates
useAI                — boolean, toggle between AI schedule (DQN) vs local heuristic
```

### 4.2 New UI Component: Natural Language Input Bar

**Location:** Top of the Dashboard view, above the task list and above the "Add Task" form.

**Design:**
- A text input field (or textarea, 2 rows) with placeholder: `"Describe your day... (e.g. ada kelas jam 12, tugas ML deadline 23:59)"`
- A submit button labeled "Parse with AI" next to it
- Uses the same dark theme / monospace styling as the existing app
- When loading, show a subtle spinner or pulsing border
- On success, parsed tasks auto-populate into the task list AND fixed blocks list

**Behavior on submit:**
1. Call `POST /parse` with the text input
2. On response, for each entry in `entries`:
   - If `entry.type === "fixed"`: add to `fixedBlocks` state. Convert `start`/`end` ISO strings to `startSlot`/`endSlot` using the existing slot system (slot = (hour - 6) * 2 + floor(minute / 30))
   - If `entry.type === "flexible"`: add to `tasks` state. Map fields:
     - `title` → `title` (uppercase it)
     - `category` → `category` (uppercase it)
     - `duration` → `duration_estimate` (parse "1h30m" → 90, "2h" → 120, "30m" → 30)
     - `priority` → `priority` (1-5, keep as-is)
     - `cognitive_demand` → `cognitive_demand` (convert int 1-5 to string: 1-2="low", 3="medium", 4-5="high")
     - `deadline` → `deadline` (extract HH:MM from ISO string if present)
     - Generate a UUID for `id`
3. If `energy_forecast` has entries, map the highest absolute `potential_energy_level` to the existing `energyLevel` slider (1-5 scale). Mapping: -2→1, -1→2, 0→3, +1→4, +2→5
4. Clear the text input after successful parse
5. Show error toast if parse fails

### 4.3 New UI Component: Character Selector

**Location:** Sidebar, below the "ENERGY STATE" section.

**Design:**
- Section header: "CHARACTER PROFILE" (same style as other sidebar headers: 9px, letterSpacing 2, gray)
- 5 selectable buttons/pills, one for each character type
- Each shows the display name and a small icon/emoji:
  - Early Grinder: 🌅
  - Slow Starter: 🌙
  - Sprinter: ⚡
  - Steady Pacer: 🔄
  - Night Owl: 🦉
- Active character has the green accent border/highlight (same style as active NavItem)
- Selecting a character updates `characterType` state
- The existing `energyProfile` builder (`buildEnergyProfile`) should be updated to use character-specific curves instead of the hardcoded one. Alternatively, call `POST /energy` to get the energy value for display.

### 4.4 Modified: "Generate Schedule" Button

The existing `doGenerate` function currently calls the local `generateSchedule()` heuristic. Update it to:

**When `useAI` is true (default):**
1. Collect all flexible tasks and fixed blocks from state
2. Convert tasks to the `/schedule` request format:
   - `id` → task.id
   - `title` → task.title
   - `duration_minutes` → task.duration_estimate (already in minutes)
   - `priority` → task.priority
   - `cognitive_demand` → convert "low"→2, "medium"→3, "high"→5
   - `deadline_hour` → parse task.deadline "HH:MM" to decimal hour (e.g. "23:59" → 23.983)
3. Convert fixed blocks to the `/schedule` request format:
   - `start_hour` → fixedBlock.startSlot / 2 + 6
   - `end_hour` → fixedBlock.endSlot / 2 + 6
   - `title` → fixedBlock.title
4. Call `POST /schedule` with `character_type` from state
5. On response, convert `scheduled` items back to the existing schedule format:
   - `scheduled_start` → (item.start_hour - 6) * 2 (convert to slot)
   - `scheduled_slots` → (item.end_hour - item.start_hour) * 2
   - Keep all other fields (id, title, priority, cognitive_demand)
   - Store `energy_at_start` and `reward` for display
6. Update `schedule` and `managerBlocks` state (managerBlocks can be empty or auto-generated from the DQN output for visual consistency)

**When `useAI` is false:**
- Use the existing local `generateSchedule()` function (no change)

### 4.5 New UI Component: AI/Heuristic Toggle

**Location:** Top bar, near the "Generate schedule" button.

**Design:**
- Small toggle switch or segmented button: "AI (DQN)" | "Heuristic"
- When AI is active, the generate button label changes to "◈ AI Schedule"
- When heuristic is active, button label stays "◈ Generate schedule"

### 4.6 Modified: Schedule View — Show Energy & Reward

In the schedule grid view, for each scheduled task block, add:
- A small energy indicator showing `energy_at_start` (e.g. a colored dot: green > 0.6, yellow 0.3-0.6, red < 0.3)
- The `reward` value as a subtle number in the corner (e.g. "+1.35" in green or "-0.42" in red)

This shows the user WHY the AI placed a task in that slot — "your energy was 0.92 here, so it put your hardest task here."

### 4.7 Modified: Footer Bar

Update the footer text to show:
- Current character type name
- Algorithm in use: "DQN (AI)" or "Manager-Worker HRL"
- Total schedule reward (if AI schedule is active)

Current footer:
```
Energy: Moderate (3/5) · Algorithm: Manager-Worker HRL + CSP
```

Updated footer:
```
Energy: Moderate (3/5) · Character: Early Grinder · Algorithm: DQN (AI) · Reward: +3.42
```

---

## 5. Data Flow Summary

```
User types natural language
        │
        ▼
POST /parse (Gemini)
        │
        ▼
Tasks + Fixed Blocks auto-populate in UI
        │
        ▼
User selects Character Profile
        │
        ▼
User clicks "AI Schedule"
        │
        ▼
POST /schedule (DQN)
        │
        ▼
Schedule renders in timeline view with energy + reward indicators
```

---

## 6. Duration Parsing Utility

The `/parse` endpoint returns duration as strings like "30m", "1h", "1h30m", "2h". The frontend needs a parser to convert these to minutes (integer). Example:

```javascript
function parseDuration(str) {
  let minutes = 0;
  const hMatch = str.match(/(\d+)h/);
  const mMatch = str.match(/(\d+)m/);
  if (hMatch) minutes += parseInt(hMatch[1]) * 60;
  if (mMatch) minutes += parseInt(mMatch[1]);
  return minutes || 30; // default 30 min
}
```

---

## 7. Error Handling

- If `/parse` fails: show inline error message below the text input, keep user text so they can retry
- If `/schedule` fails: fall back to local heuristic scheduler, show a toast "AI unavailable, using local scheduler"
- If `/health` shows missing models: disable the AI toggle, force heuristic mode
- Network errors: show "Backend not connected" warning in the sidebar

---

## 8. Environment Variable

Add to `.env` (React app):
```
REACT_APP_API_URL=http://localhost:8000
```

All fetch calls use `process.env.REACT_APP_API_URL` as base URL.

---

## 9. Files to Modify

| File | Change |
|------|--------|
| `src/APUAHRLS.jsx` | All UI changes described above |
| `.env` | Add `REACT_APP_API_URL` |

No new files or dependencies needed — all API calls use native `fetch()`.

---

## 10. Out of Scope

- User authentication / login
- Persistent database (localStorage is fine for now)
- Character quiz/onboarding flow (user manually selects character)
- Mobile responsive layout
- Deployment / hosting configuration
