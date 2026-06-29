from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import webbrowser
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import psutil

from .app import build_service
from .asr_models import ranked_asr_candidates
from .config import CachePaths, ModelConfig, load_config, save_config
from .domain import SceneContextBlock
from .integrations import OllamaClient
from .queue import QueueError
from .utils import VIDEO_EXTENSIONS, no_window_creationflags

RECOMMENDED_TRANSLATION_MODEL = "fredrezones55/Gemma-4-Uncensored-HauhauCS-Aggressive:e2b"


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Fast Multilanguage Transcriber</title>
  <script crossorigin src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
  <script crossorigin src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>
  <style>
    :root {
      color-scheme: dark;
      font-family: "Segoe UI", system-ui, sans-serif;
      background: #101113;
      color: #f7f9fb;
      --bg: #101113;
      --surface: #171b1f;
      --surface-2: #20272d;
      --line: rgba(232,236,238,.18);
      --muted: #b3bdc3;
      --soft: #f0f3f4;
      --accent: #ffd84d;
      --danger: #ffd1d1;
      --danger-text: #5f1515;
      --shadow: 0 18px 60px rgba(0,0,0,.28);
    }
    * { box-sizing: border-box; }
    html { min-width: 320px; }
    body {
      margin: 0;
      min-height: 100vh;
      background:
        radial-gradient(circle at top left, rgba(236,226,181,.10), transparent 34rem),
        linear-gradient(135deg, #101113, #171b1f 68%);
      overflow-y: auto;
    }
    button, select, input { font: inherit; }
    button {
      min-height: 38px;
      border: 1px solid transparent;
      border-radius: 6px;
      background: var(--accent);
      color: #111827;
      cursor: pointer;
      font-weight: 700;
      padding: 9px 12px;
      white-space: nowrap;
    }
    button.secondary { background: transparent; color: var(--soft); border-color: var(--line); }
    button.danger { background: var(--danger); color: var(--danger-text); }
    button:disabled { cursor: not-allowed; opacity: .45; }
    input, select {
      min-height: 38px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: rgba(255,255,255,.05);
      color: var(--soft);
      padding: 9px 10px;
      min-width: 0;
    }
    input::placeholder { color: #8797a4; }
    main {
      width: min(1480px, 100%);
      margin: 0 auto;
      padding: clamp(14px, 2vw, 24px);
    }
    .topbar {
      position: sticky;
      top: 0;
      z-index: 5;
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 16px;
      align-items: center;
      padding: 14px 0 16px;
      background: linear-gradient(180deg, rgba(16,17,19,.98), rgba(16,17,19,.86) 72%, transparent);
      backdrop-filter: blur(10px);
    }
    h1 { margin: 0; font-size: clamp(24px, 3vw, 38px); letter-spacing: 0; line-height: 1.08; }
    p { color: var(--muted); line-height: 1.5; margin: 6px 0 0; }
    .layout {
      display: grid;
      grid-template-columns: minmax(360px, .9fr) minmax(420px, 1.35fr);
      gap: 14px;
      align-items: start;
    }
    .stack { display: grid; gap: 14px; min-width: 0; }
    .panel {
      background: color-mix(in srgb, var(--surface) 86%, transparent);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      min-width: 0;
      overflow: clip;
    }
    .panel-head {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      padding: 14px 16px;
      border-bottom: 1px solid var(--line);
      background: rgba(255,255,255,.03);
    }
    .panel-head strong { font-size: 15px; }
    .panel-body { padding: 16px; }
    .controls {
      display: grid;
      grid-template-columns: repeat(12, minmax(0, 1fr));
      gap: 10px;
      align-items: end;
    }
    .control-span-12 { grid-column: span 12; }
    .control-span-6 { grid-column: span 6; }
    .control-span-4 { grid-column: span 4; }
    .control-span-3 { grid-column: span 3; }
    .field { display: grid; gap: 6px; min-width: 0; }
    .field label, .check-label { color: var(--muted); font-size: 12px; font-weight: 600; }
    .check-label {
      display: flex;
      gap: 8px;
      align-items: center;
      min-height: 38px;
      padding: 0 4px;
    }
    .button-row { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
    .path { font-family: Consolas, "Cascadia Mono", monospace; color: #d8e4e6; overflow-wrap: anywhere; }
    .target-list {
      display: grid;
      gap: 8px;
      max-height: min(34vh, 340px);
      overflow: auto;
      padding-right: 2px;
    }
    .item {
      display: grid;
      gap: 2px;
      padding: 10px 12px;
      background: rgba(0,0,0,.18);
      border: 1px solid rgba(255,255,255,.06);
      border-radius: 6px;
    }
    .empty {
      border: 1px dashed var(--line);
      border-radius: 8px;
      color: var(--muted);
      padding: 18px;
      text-align: center;
    }
    .drop-zone {
      border: 1px dashed rgba(255,216,77,.55);
      border-radius: 8px;
      background: rgba(255,216,77,.07);
      color: var(--soft);
      padding: 18px;
      text-align: center;
    }
    .drop-zone strong { display: block; }
    .drop-zone span { color: var(--muted); display: block; font-size: 12px; margin-top: 4px; }
    .status-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
    }
    .metric {
      border: 1px solid rgba(255,255,255,.08);
      border-radius: 8px;
      padding: 12px;
      background: rgba(0,0,0,.14);
      min-width: 0;
    }
    .metric span { color: var(--muted); display: block; font-size: 12px; }
    .metric strong { display: block; font-size: clamp(20px, 3vw, 32px); margin-top: 2px; }
    .model-list { display: grid; gap: 8px; }
    .model-row {
      display: grid;
      grid-template-columns: 150px minmax(0, 1fr) auto;
      gap: 10px;
      align-items: center;
      padding: 8px 0;
      border-top: 1px solid rgba(255,255,255,.06);
    }
    .model-row:first-child { border-top: 0; padding-top: 0; }
    .model-row span:first-child { color: var(--muted); font-size: 12px; font-weight: 700; }
    .model-detail-row {
      display: grid;
      gap: 8px;
      padding: 10px 0;
      border-top: 1px solid rgba(255,255,255,.06);
    }
    .model-detail-head { display: flex; justify-content: space-between; gap: 12px; align-items: start; }
    .model-detail-head strong { overflow-wrap: anywhere; }
    .model-locations {
      display: grid;
      gap: 5px;
      padding: 8px;
      border: 1px solid rgba(255,255,255,.08);
      border-radius: 6px;
      background: rgba(0,0,0,.12);
    }
    .model-locations span { color: var(--muted); font-size: 12px; overflow-wrap: anywhere; }
    .job-list, .preview-list, .note-list { display: grid; gap: 8px; max-height: min(42vh, 480px); overflow: auto; }
    .preview-list { max-height: min(58vh, 680px); }
    .job-row, .preview-row, .note-row {
      width: 100%;
      text-align: left;
      display: grid;
      gap: 4px;
      background: rgba(0,0,0,.14);
      color: var(--soft);
      border-color: rgba(255,255,255,.08);
      white-space: normal;
    }
    .job-row.active, .preview-row.active { border-color: var(--accent); }
    .preview-row.selected { background: rgba(255,216,77,.12); border-color: rgba(255,216,77,.55); }
    .range-card {
      display: grid;
      gap: 8px;
      padding: 10px;
      border: 1px solid rgba(255,255,255,.1);
      border-radius: 8px;
      background: rgba(0,0,0,.14);
    }
    .range-card textarea { min-height: 76px; }
    .range-card-head { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 8px; align-items: center; }
    .section-note { color: var(--muted); margin: 0; }
    textarea {
      min-height: 90px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: rgba(255,255,255,.05);
      color: var(--soft);
      padding: 9px 10px;
      resize: vertical;
      font: inherit;
    }
    .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    .tiny { color: var(--muted); font-size: 12px; }
    .guided-grid { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 8px; align-items: end; }
    .workflow {
      display: grid;
      grid-template-columns: repeat(6, minmax(0, 1fr));
      gap: 8px;
      margin-bottom: 14px;
    }
    .workflow-step {
      min-width: 0;
      padding: 10px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: rgba(255,255,255,.04);
    }
    .workflow-step strong { display: block; font-size: 13px; }
    .workflow-step span { display: block; color: var(--muted); font-size: 12px; line-height: 1.35; margin-top: 4px; }
    .review-box {
      border: 1px solid rgba(255,216,77,.38);
      background: rgba(255,216,77,.08);
      border-radius: 8px;
      padding: 10px 12px;
    }
    .action-section {
      display: grid;
      gap: 10px;
      padding-top: 12px;
      border-top: 1px solid var(--line);
    }
    .action-section strong { font-size: 14px; }
    progress { width: 100%; height: 18px; accent-color: var(--accent); margin-top: 14px; }
    pre {
      white-space: pre-wrap;
      overflow: auto;
      max-height: min(48vh, 560px);
      min-height: 260px;
      margin: 0;
      background: #07111a;
      border-top: 1px solid var(--line);
      color: #cbd5d7;
      padding: 14px 16px;
      line-height: 1.45;
      contain: content;
    }
    .error { border-color: rgba(255,209,209,.65); }
    .error .panel-head { color: var(--danger); }
    footer {
      margin: 18px 0 4px;
      color: #8ea0a4;
      font-size: 13px;
      display: flex;
      justify-content: space-between;
      gap: 12px;
      flex-wrap: wrap;
    }
    a { color: var(--accent); }
    @media (max-width: 980px) {
      .layout { grid-template-columns: 1fr; }
      .workflow { grid-template-columns: repeat(3, minmax(0, 1fr)); }
      .topbar { grid-template-columns: 1fr; }
      .topbar .button-row { justify-content: flex-start; }
    }
    @media (max-width: 640px) {
      main { padding: 12px; }
      .controls { grid-template-columns: 1fr; }
      .control-span-12, .control-span-6, .control-span-4, .control-span-3 { grid-column: 1; }
    .button-row button, .button-row select { flex: 1 1 150px; }
      .status-grid { grid-template-columns: 1fr; }
      .model-row { grid-template-columns: 1fr; gap: 4px; }
      .two-col { grid-template-columns: 1fr; }
      .guided-grid { grid-template-columns: 1fr; }
      .workflow { grid-template-columns: 1fr; }
      pre { min-height: 180px; }
    }
  </style>
</head>
<body>
  <main id="root"></main>
  <script>
    const e = React.createElement;
    const api = (path, body) => fetch(path, {
      method: body ? "POST" : "GET",
      headers: body ? {"Content-Type": "application/json"} : {},
      body: body ? JSON.stringify(body) : undefined
    }).then(async r => {
      const data = await r.json();
      if (!r.ok) throw new Error(data.error || r.statusText);
      return data;
    });
    const formatClock = seconds => {
      const total = Math.max(0, Math.floor(Number(seconds) || 0));
      const h = Math.floor(total / 3600);
      const m = Math.floor((total % 3600) / 60);
      const s = total % 60;
      return [h, m, s].map(v => String(v).padStart(2, "0")).join(":");
    };
    const timeToSeconds = value => {
      const parts = String(value || "").split(":").map(Number);
      if (parts.length === 3) return parts[0] * 3600 + parts[1] * 60 + parts[2];
      if (parts.length === 2) return parts[0] * 60 + parts[1];
      return Number(value) || 0;
    };
      const stageLabel = stage => ({
        extract: "Prepare audio",
        transcribe: "Transcribe source language",
        literal: "Make direct English translation",
        adapted: "Make context-applied English",
        translate_literal: "Make direct English translation",
        translate_adapted: "Make context-applied English",
        finalize: "Save subtitle files"
      })[stage] || (stage || "Not started");
    const jobStepText = row => {
      if (!row) return "";
      if (row.status === "completed" && row.stage === "finalize") return "Saved subtitle files";
      return row.step_text || "";
    };
    const selectedStageLabel = row => {
      if (row?.status === "completed" && row?.stage === "finalize") return "Saved subtitle files";
      return stageLabel(row?.stage);
    };
    const reviewHint = row => {
      if (!row) return "Start at Step 1: choose videos or an existing subtitle file.";
      if (row.status === "completed") return "Review output now: open the subtitle lines below, edit any bad line, or open the saved files.";
      if (row.status === "failed") return "This job failed. Read the latest message, then use Run this job again.";
      if (row.status === "queued") return "Queued. Press Start processing all jobs when ready.";
      return `Now doing: ${stageLabel(row.stage)}. Wait, or press Stop after current safe step.`;
    };

    function App() {
      const [targets, setTargets] = React.useState([]);
      const [profile, setProfile] = React.useState("conservative");
      const [recursive, setRecursive] = React.useState(true);
      const [includeAdapted, setIncludeAdapted] = React.useState(true);
      const [preferFast, setPreferFast] = React.useState(false);
      const [status, setStatus] = React.useState({jobs:[], settings:{profiles:["conservative"]}});
      const [models, setModels] = React.useState({storage:"", hf_cache:"", selected:[]});
      const [error, setError] = React.useState("");
      const [manualPath, setManualPath] = React.useState("");
      const [selectedJobId, setSelectedJobId] = React.useState("");
      const [job, setJob] = React.useState(null);
      const [line, setLine] = React.useState(null);
      const [selectedCueIndexes, setSelectedCueIndexes] = React.useState([]);
      const [dropRole, setDropRole] = React.useState("direct");
      const [batchLabel, setBatchLabel] = React.useState("");
      const [context, setContext] = React.useState("");
      const [noteStart, setNoteStart] = React.useState("");
      const [noteEnd, setNoteEnd] = React.useState("");
      const [noteText, setNoteText] = React.useState("");
      const [notes, setNotes] = React.useState([]);
      const [importDraft, setImportDraft] = React.useState({video:"", primary_subtitle:"", japanese:"", direct:"", easy:"", reference:""});
      const [settingsDraft, setSettingsDraft] = React.useState(null);
      const [health, setHealth] = React.useState(null);
      const currentSettings = settingsDraft || status.settings || {};

      const refresh = () => api("/api/status").then(data => { setStatus(data); if (!settingsDraft) setSettingsDraft(data.settings); }).catch(err => setError(err.message));
      const refreshModels = () => api("/api/models").then(setModels).catch(err => setError(err.message));
      React.useEffect(() => { refresh(); refreshModels(); const id = setInterval(refresh, 1500); return () => clearInterval(id); }, []);
      React.useEffect(() => {
        if (!selectedJobId) return;
        api(`/api/job?id=${encodeURIComponent(selectedJobId)}`).then(data => {
          setJob(data);
          const manifest = data.manifest || {};
          setBatchLabel(manifest.series || "");
          setContext(manifest.job_context || "");
          setNotes(manifest.scene_contexts || []);
          setIncludeAdapted(manifest.include_adapted_english !== false);
          setPreferFast(Boolean(manifest.prefer_fast_translation));
          setSelectedCueIndexes([]);
          setLine(null);
        }).catch(err => setError(err.message));
      }, [selectedJobId]);

      const pickFolder = () => { setError(""); api("/api/pick-folder").then(d => d.path && setTargets([d.path])).catch(err => setError(err.message)); };
      const pickFiles = () => { setError(""); api("/api/pick-files").then(d => d.paths?.length && setTargets(d.paths)).catch(err => setError(err.message)); };
      const addManual = () => {
        const value = manualPath.trim();
        if (!value) return;
        setTargets(current => current.includes(value) ? current : current.concat(value));
        setManualPath("");
        setError("");
      };
      const post = (path, body, after=refresh) => { setError(""); return api(path, body).then(data => { after && after(data); refreshModels(); return data; }).catch(err => setError(err.message)); };
      const selectNewJob = data => {
        if (data?.jobs) setStatus(data);
        const jobId = data?.queued?.[0] || data?.job_id || "";
        if (jobId) setSelectedJobId(jobId);
        refresh();
      };
      const enqueueAndStart = () => post("/api/start", {targets, profile, recursive, include_adapted_english:includeAdapted, prefer_fast_translation:preferFast, batch_label:batchLabel, context, scene_contexts:notes}, selectNewJob);
      const enqueueOnly = () => post("/api/enqueue", {targets, profile, recursive, include_adapted_english:includeAdapted, prefer_fast_translation:preferFast, batch_label:batchLabel, context, scene_contexts:notes}, selectNewJob);
      const addNote = () => {
        if (!noteStart || !noteEnd || !noteText.trim()) return;
        setNotes(current => current.concat({start_seconds:timeToSeconds(noteStart), end_seconds:timeToSeconds(noteEnd), notes:noteText.trim()}));
        setNoteStart(""); setNoteEnd(""); setNoteText("");
      };
      const saveNotes = () => selectedJobId && post("/api/job/notes", {job_id:selectedJobId, batch_label:batchLabel, context, scene_contexts:notes, include_adapted_english:includeAdapted, prefer_fast_translation:preferFast});
      const deleteSelectedJob = () => {
        if (!selectedJobId) return;
        if (!confirm("Delete this job from the queue list? Saved subtitle files in the output folder stay on disk.")) return;
        post("/api/job/delete", {job_id:selectedJobId}, () => {
          setSelectedJobId("");
          setJob(null);
          setLine(null);
          setSelectedCueIndexes([]);
          refresh();
        });
      };
      const saveLine = () => line && selectedJobId && post("/api/job/line", {
        job_id:selectedJobId,
        cue_index:line.cue_index,
        japanese_text:line.has_japanese ? line.japanese : null,
        literal_english_text:line.has_literal_english ? line.literal_english : null,
        adapted_english_text:line.has_adapted_english ? line.adapted_english : null,
        reference_text:line.has_reference ? line.reference : null
      }, () => api(`/api/job?id=${encodeURIComponent(selectedJobId)}`).then(setJob));
      const importExisting = () => post("/api/import-existing", {profile, ...importDraft, batch_label:batchLabel, context, scene_contexts:notes, include_adapted_english:includeAdapted, prefer_fast_translation:preferFast}, selectNewJob);
      const saveSettings = () => settingsDraft && post("/api/settings/save", settingsDraft, data => setSettingsDraft(data));
      const chooseSubtitle = key => api("/api/pick-subtitle").then(d => d.path && setImportDraft(current => ({...current, [key]:d.path}))).catch(err => setError(err.message));
      const chooseCacheFolder = () => api("/api/pick-folder").then(d => d.path && setSettingsDraft(current => ({...current, cache_paths:{...current.cache_paths, hf_hub_cache:d.path}}))).catch(err => setError(err.message));
      const chooseAsrFolder = () => api("/api/pick-folder").then(d => d.path && setSettingsDraft(current => ({...current, models:{...current.models, asr:d.path}}))).catch(err => setError(err.message));
      const selectedPreviewRows = () => (job?.preview || []).filter(row => selectedCueIndexes.includes(row.cue_index));
      const rangeFromRows = rows => rows.length ? {
        start: formatClock(Math.min(...rows.map(row => Number(row.start)))),
        end: formatClock(Math.max(...rows.map(row => Number(row.end))))
      } : null;
      const applyRange = range => {
        if (!range) return;
        setNoteStart(range.start);
        setNoteEnd(range.end);
      };
      const setRangeFromSelection = () => applyRange(rangeFromRows(selectedPreviewRows()));
      React.useEffect(() => {
        const rows = selectedPreviewRows();
        if (!rows.length) return;
        applyRange(rangeFromRows(rows));
      }, [selectedCueIndexes, job]);
      const addContextForSelection = () => {
        setRangeFromSelection();
        if (!noteText.trim()) setNoteText("");
      };
      const updateNote = (index, patch) => setNotes(current => current.map((item, noteIndex) => noteIndex === index ? {...item, ...patch} : item));
      const removeNote = index => setNotes(current => current.filter((_item, noteIndex) => noteIndex !== index));
      const toggleCue = (row, ev) => {
        if (ev.shiftKey && selectedCueIndexes.length) {
          const preview = job?.preview || [];
          const lastIndex = preview.findIndex(item => item.cue_index === selectedCueIndexes[selectedCueIndexes.length - 1]);
          const rowIndex = preview.findIndex(item => item.cue_index === row.cue_index);
          if (lastIndex >= 0 && rowIndex >= 0) {
            const start = Math.min(lastIndex, rowIndex);
            const end = Math.max(lastIndex, rowIndex);
            setSelectedCueIndexes(preview.slice(start, end + 1).map(item => item.cue_index));
          } else {
            setSelectedCueIndexes(current => current.includes(row.cue_index) ? current.filter(value => value !== row.cue_index) : current.concat(row.cue_index));
          }
        } else if (ev.ctrlKey || ev.metaKey) {
          setSelectedCueIndexes(current => current.includes(row.cue_index) ? current.filter(value => value !== row.cue_index) : current.concat(row.cue_index));
        } else {
          setSelectedCueIndexes([row.cue_index]);
        }
        setLine({...row});
      };
      const redoSelectedRange = () => {
        const rows = selectedPreviewRows();
        const start = rows.length ? formatClock(Math.min(...rows.map(row => Number(row.start)))) : noteStart;
        const end = rows.length ? formatClock(Math.max(...rows.map(row => Number(row.end)))) : noteEnd;
        return post("/api/job/rebuild", {job_id:selectedJobId, batch_label:batchLabel, context, scene_contexts:notes, start_timecode:start, end_timecode:end, include_adapted_english:includeAdapted, prefer_fast_translation:preferFast});
      };
      const dropSubtitle = async ev => {
        ev.preventDefault();
        setError("");
        const file = [...(ev.dataTransfer?.files || [])].find(item => item.name.toLowerCase().endsWith(".srt"));
        if (!file) { setError("Drop an .srt subtitle file."); return; }
        const data = await post("/api/upload-subtitle", {
          filename:file.name,
          content:await file.text(),
          role:dropRole,
          job_id:selectedJobId,
          profile
        }, null);
        if (data?.job_id) setSelectedJobId(data.job_id);
        await refresh();
      };
      const jobs = status.jobs || [];
      const selectedRow = jobs.find(row => row.job_id === selectedJobId);
      const modelText = selectedRow?.current_model || "No model active now";
      const outputButtons = [
        ["review", "Open review bundle in Subtitle Edit"],
        ["ja", "Open Japanese in Subtitle Edit"],
        ["direct", "Open direct English translation in Subtitle Edit"],
        ["easy", "Open context-applied English in Subtitle Edit"],
        ["direct-partial", "Open direct English draft in Subtitle Edit"],
        ["easy-partial", "Open context-applied draft in Subtitle Edit"]
      ];
      const workflowSteps = [
        ["1", "Choose input", "Pick videos, folder, or existing subtitles."],
        ["2", "Choose models", "Select listening and English models."],
        ["3", "Add context", "Tell the translator names, tone, slang, scene meaning."],
        ["4", "Start", "Queue work and let the app process it."],
        ["5", "Review", "Open subtitle lines, edit text, or add context to selected lines."],
        ["6", "Export", "Open saved subtitles or Subtitle Edit."]
      ];

      return e(React.Fragment, null,
        e("header", {className:"topbar"},
          e("div", null, e("h1", null, "Fast Multilanguage Transcriber"), e("p", null, "Local folder and file transcription using the existing adaptive runner.")),
          e("div", {className:"button-row"},
            e("button", {className:"secondary", onClick:refresh}, "Refresh"),
            e("button", {onClick:()=>post("/api/worker/start", {})}, "Start processing all jobs"),
            e("button", {className:"danger", onClick:()=>post("/api/worker/stop", {})}, "Stop after current step")
          )
        ),
        e("section", {className:"workflow"},
          workflowSteps.map(([number, title, text]) => e("div", {className:"workflow-step", key:number},
            e("strong", null, `${number}. ${title}`),
            e("span", null, text)
          ))
        ),
        e("div", {className:"layout"},
          e("div", {className:"stack"},
            e("section", {className:"panel"},
              e("div", {className:"panel-head"}, e("strong", null, "Queue inputs"), e("span", null, `${targets.length} selected`)),
              e("div", {className:"panel-body stack"},
                e("div", {className:"controls"},
                  e("div", {className:"field control-span-12"},
                    e("label", null, "Manual path"),
                    e("input", {value:manualPath, onChange:ev=>setManualPath(ev.target.value), onKeyDown:ev=>{ if (ev.key === "Enter") addManual(); }, disabled:status.running, placeholder:"Paste a folder or video path"})
                  ),
                  e("button", {className:"control-span-4", onClick:pickFolder, disabled:status.running}, "Select folder"),
                  e("button", {className:"control-span-4", onClick:pickFiles, disabled:status.running}, "Select files"),
                  e("button", {className:"secondary control-span-4", onClick:addManual, disabled:status.running || !manualPath.trim()}, "Add path"),
                  e("div", {className:"field control-span-6"},
                    e("label", null, "Profile"),
                    e("select", {value:profile, onChange:ev=>setProfile(ev.target.value)},
                      ((status.settings && status.settings.profiles) || ["conservative"]).map(v => e("option", {key:v, value:v}, v))
                    )
                  ),
                  e("label", {className:"check-label control-span-6"},
                    e("input", {type:"checkbox", checked:recursive, onChange:ev=>setRecursive(ev.target.checked)}),
                    "Search subfolders"
                  ),
                  e("label", {className:"check-label control-span-6"},
                    e("input", {type:"checkbox", checked:includeAdapted, onChange:ev=>setIncludeAdapted(ev.target.checked)}),
                    "Context-applied English"
                  ),
                  e("label", {className:"check-label control-span-6"},
                    e("input", {type:"checkbox", checked:preferFast, onChange:ev=>setPreferFast(ev.target.checked)}),
                    "Fast translation"
                  )
                ),
                e("div", {className:"button-row"},
                  e("button", {onClick:enqueueAndStart, disabled:!targets.length}, "Add files and start processing"),
                  e("button", {className:"secondary", onClick:enqueueOnly, disabled:!targets.length}, "Add files but do not start yet"),
                  e("button", {className:"secondary", onClick:()=>setTargets([]), disabled:!targets.length}, "Clear chosen files")
                ),
                targets.length
                  ? e("div", {className:"target-list"}, targets.map((path, index) => e("div", {className:"item", key:path}, e("span", null, `Source ${index + 1}`), e("span", {className:"path"}, path))))
                  : e("div", {className:"empty"}, "Select a folder, select files, or paste a path.")
              )
            ),
            e("section", {className:"panel"},
              e("div", {className:"panel-head"}, e("strong", null, "Jobs"), e("span", null, status.worker_running ? "Running" : (status.pause_requested ? "Stopping" : "Idle"))),
              e("div", {className:"panel-body job-list"},
                jobs.length ? jobs.map(row => e("button", {key:row.job_id, className:`job-row ${row.job_id===selectedJobId?"active":""}`, onClick:()=>setSelectedJobId(row.job_id)},
                  e("strong", null, row.source),
                  e("span", {className:"tiny"}, `${row.status} | ${selectedStageLabel(row)} | ${row.overall_progress_percent || 0}%`),
                  e("span", null, jobStepText(row))
                )) : e("div", {className:"empty"}, "No jobs yet.")
              )
            ),
            error ? e("section", {className:"panel error"}, e("div", {className:"panel-head"}, e("strong", null, "Error")), e("div", {className:"panel-body"}, e("p", null, error))) : null
          ),
          e("div", {className:"stack"},
            e("section", {className:"panel"},
              e("div", {className:"panel-head"}, e("strong", null, "Selected job"), e("span", null, selectedRow ? selectedRow.status : "None")),
              e("div", {className:"panel-body stack"},
                e("div", {className:"review-box"}, e("strong", null, "What to do now"), e("p", null, reviewHint(selectedRow))),
                selectedRow ? e("div", {className:"status-grid"},
                  e("div", {className:"metric"}, e("span", null, "Current step"), e("strong", null, selectedStageLabel(selectedRow))),
                  e("div", {className:"metric"}, e("span", null, "Progress"), e("strong", null, `${selectedRow.overall_progress_percent || 0}%`)),
                  e("div", {className:"metric"}, e("span", null, "Model in use"), e("strong", null, modelText))
                ) : e("div", {className:"empty"}, "Select a job."),
                e("div", {className:"button-row"},
                  e("button", {className:"secondary", disabled:!selectedJobId, onClick:()=>post("/api/job/retry", {job_id:selectedJobId})}, "Run this job again"),
                  e("button", {className:"secondary", disabled:!selectedJobId, onClick:()=>post("/api/open", {job_id:selectedJobId, action:"folder"})}, "Open subtitle folder"),
                  e("button", {className:"danger", disabled:!selectedJobId, onClick:deleteSelectedJob}, "Delete job from list")
                ),
                e("p", {className:"section-note"}, "These controls manage the job card. Delete removes this job card only; exported subtitle files stay in the output folder."),
                e("div", {className:"action-section"},
                  e("strong", null, "Open in Subtitle Edit"),
                  e("p", {className:"section-note"}, "Use these when you want Subtitle Edit to open saved subtitles or drafts."),
                  e("div", {className:"button-row"},
                    outputButtons.map(([kind, label]) => e("button", {key:kind, className:"secondary", disabled:!selectedJobId, onClick:()=>post("/api/open", {job_id:selectedJobId, action:kind})}, label))
                  )
                )
              )
            ),
            e("section", {className:"panel"},
              e("div", {className:"panel-head"}, e("strong", null, "Preview and line editor"), e("button", {className:"secondary", disabled:!selectedJobId, onClick:()=>api(`/api/job?id=${encodeURIComponent(selectedJobId)}`).then(setJob)}, "Reload")),
              e("div", {className:"panel-body stack"},
                e("p", {className:"section-note"}, "Click a line to edit it. Ctrl-click adds separate lines. Shift-click selects a continuous range and fills the time-range context below."),
                e("div", {className:"field"},
                  e("label", null, "Dropped subtitle type"),
                  e("select", {value:dropRole, onChange:ev=>setDropRole(ev.target.value)},
                    [["direct","Direct English translation"],["ja","Japanese"],["easy","Context-applied English"],["reference","Reference"]].map(([value,label]) => e("option", {key:value, value}, label))
                  )
                ),
                e("div", {className:"preview-list"}, job && job.preview && job.preview.length ? job.preview.map(row => e("button", {key:row.cue_index, className:`preview-row ${line && line.cue_index===row.cue_index?"active":""} ${selectedCueIndexes.includes(row.cue_index)?"selected":""}`, onClick:ev=>toggleCue(row, ev)},
                  e("strong", null, `#${row.cue_index} ${row.start}s-${row.end}s`),
                  e("span", null, row.japanese || row.literal_english || row.adapted_english || row.reference || "")
                )) : e("div", {className:"drop-zone", onDragOver:ev=>ev.preventDefault(), onDrop:dropSubtitle},
                  e("strong", null, "Drop an .srt file here to edit existing subtitles"),
                  e("span", null, selectedJobId ? "Dropped file attaches to the selected job." : "Dropped direct English translation/Japanese subtitles create a new editable job.")
                )),
                line ? e("div", {className:"stack"},
                  e("textarea", {value:line.japanese || "", onChange:ev=>setLine({...line, japanese:ev.target.value}), placeholder:"Japanese"}),
                  e("textarea", {value:line.literal_english || "", onChange:ev=>setLine({...line, literal_english:ev.target.value}), placeholder:"Direct English translation"}),
                  e("textarea", {value:line.adapted_english || "", onChange:ev=>setLine({...line, adapted_english:ev.target.value}), placeholder:"Context-applied English"}),
                  e("textarea", {value:line.reference || "", onChange:ev=>setLine({...line, reference:ev.target.value}), placeholder:"Reference"}),
                  e("button", {onClick:saveLine}, "Save line")
                ) : null
              )
            ),
            e("section", {className:"panel"},
              e("div", {className:"panel-head"}, e("strong", null, "Overall video context"), e("span", null, status.rebuild_running ? "Running" : "Idle")),
              e("div", {className:"panel-body stack"},
                e("p", {className:"section-note"}, "This text is added to every English translation prompt after the Japanese audio has been transcribed. Use it for names, relationships, tone, slang, and story so far."),
                e("input", {value:batchLabel, onChange:ev=>setBatchLabel(ev.target.value), placeholder:"Series or project name, e.g. MARA-018 or Show title"}),
                e("textarea", {value:context, onChange:ev=>setContext(ev.target.value), placeholder:"Overall video context used in every translation prompt: character names, relationships, tone, slang, setting, story so far"}),
                e("div", {className:"button-row"},
                  e("button", {className:"secondary", disabled:!selectedJobId, onClick:saveNotes}, "Save context"),
                  e("button", {disabled:!selectedJobId, onClick:()=>post("/api/job/rebuild", {job_id:selectedJobId, batch_label:batchLabel, context, scene_contexts:notes, include_adapted_english:includeAdapted, prefer_fast_translation:preferFast})}, "Retranslate whole job with context")
                )
              )
            ),
            e("section", {className:"panel"},
              e("div", {className:"panel-head"}, e("strong", null, "Time-range context"), e("span", null, selectedCueIndexes.length ? `${selectedCueIndexes.length} selected` : "Select subtitle lines")),
              e("div", {className:"panel-body stack"},
                e("p", {className:"section-note"}, "Select subtitle lines in the preview. Start and end times fill automatically. Add as many context ranges as needed, then retranslate the selected time range or save them for the full job."),
                e("div", {className:"button-row"},
                  e("button", {className:"secondary", disabled:!selectedCueIndexes.length, onClick:setRangeFromSelection}, "Fill times from selected lines"),
                  e("button", {className:"secondary", disabled:!selectedCueIndexes.length, onClick:addContextForSelection}, "New context block from selection")
                ),
                e("p", {className:"tiny"}, "Selected subtitle lines fill these times automatically. You can still type a time range by hand."),
                e("div", {className:"two-col"}, e("input", {value:noteStart, onChange:ev=>setNoteStart(ev.target.value), placeholder:"Start time, e.g. 00:12:04"}), e("input", {value:noteEnd, onChange:ev=>setNoteEnd(ev.target.value), placeholder:"End time, e.g. 00:12:22"})),
                e("textarea", {value:noteText, onChange:ev=>setNoteText(ev.target.value), placeholder:"What happens in this time range: who is speaking, intended meaning, local joke, relationship change, tone shift"}),
                e("div", {className:"button-row"},
                  e("button", {className:"secondary", onClick:addNote}, "Add time-range context"),
                  e("button", {className:"secondary", disabled:!selectedJobId, onClick:saveNotes}, "Save all context ranges"),
                  e("button", {disabled:!selectedJobId || (!selectedCueIndexes.length && (!noteStart || !noteEnd)), onClick:redoSelectedRange}, "Retranslate selected time range")
                ),
                e("div", {className:"note-list"}, notes.length ? notes.map((item, index) => e("div", {className:"range-card", key:index},
                  e("div", {className:"range-card-head"},
                    e("strong", null, `Context range ${index + 1}`),
                    e("button", {className:"danger", onClick:()=>removeNote(index)}, "Remove")
                  ),
                  e("div", {className:"two-col"},
                    e("input", {value:formatClock(Number(item.start_seconds || 0)), onChange:ev=>updateNote(index, {start_seconds:timeToSeconds(ev.target.value)}), placeholder:"Start time"}),
                    e("input", {value:formatClock(Number(item.end_seconds || 0)), onChange:ev=>updateNote(index, {end_seconds:timeToSeconds(ev.target.value)}), placeholder:"End time"})
                  ),
                  e("textarea", {value:item.notes || "", onChange:ev=>updateNote(index, {notes:ev.target.value}), placeholder:"Context for this range"})
                )) : e("div", {className:"empty"}, "No time-range context yet. Select subtitle lines above, add context, then save."))
              )
            ),
            e("section", {className:"panel"},
              e("div", {className:"panel-head"}, e("strong", null, "Models"), e("button", {className:"secondary", onClick:refreshModels}, "Refresh")),
              e("div", {className:"panel-body model-list"},
                e("div", {className:"model-row"}, e("span", null, "Ollama storage"), e("span", {className:"path"}, models.storage || "Unknown"), e("span", null, "")),
                (models.selected || []).map(item => e("div", {className:"model-detail-row", key:item.label},
                  e("div", {className:"model-detail-head"},
                    e("div", null,
                      e("span", {className:"tiny"}, item.label),
                      e("strong", {className:"path"}, item.name)
                    ),
                    e("strong", null, item.size || "Not installed")
                  ),
                  e("div", {className:"model-locations"},
                    e("span", null, `Stored on: ${models.storage || "Unknown"}`),
                    e("span", null, `Manifest file: ${item.manifest_path || "Not found"}`),
                    (item.blob_paths && item.blob_paths.length)
                      ? item.blob_paths.map((path, index) => e("span", {key:path}, `${index === 0 ? "Blob files" : "Blob file"}: ${path}`))
                      : e("span", null, "Blob files: Not found")
                  )
                )),
                e("div", {className:"model-row"}, e("span", null, "Japanese cache"), e("span", {className:"path"}, models.hf_cache || "Default Hugging Face cache"), e("span", null, ""))
              )
            ),
            e("section", {className:"panel"},
              e("div", {className:"panel-head"}, e("strong", null, "Transcription and translation models"), e("span", null, "App-wide")),
              e("div", {className:"panel-body stack"},
                e("p", {className:"section-note"}, "Choose Japanese listening model, English translation model, and cache folder. Pickers fill paths for you."),
                settingsDraft ? e("div", {className:"field"},
                  e("label", null, "Japanese listening model"),
                  e("select", {value:settingsDraft.models?.asr || "", onChange:ev => {
                    const chosen = (currentSettings.asr_options || []).find(item => item.model === ev.target.value);
                    setSettingsDraft({...settingsDraft, models:{...settingsDraft.models, asr:ev.target.value, asr_engine:chosen?.engine || settingsDraft.models.asr_engine}});
                  }},
                    (currentSettings.asr_options || []).map(item => e("option", {key:item.model, value:item.model}, `${item.label} | ${item.model}`))
                  )
                ) : null,
                settingsDraft ? e("div", {className:"guided-grid"},
                  e("div", {className:"field"},
                    e("label", null, "Local Japanese model folder"),
                    e("input", {value:settingsDraft.models?.asr || "", readOnly:true, placeholder:"Use dropdown or pick folder"})
                  ),
                  e("button", {className:"secondary", onClick:chooseAsrFolder}, "Pick ASR folder")
                ) : null,
                settingsDraft ? e("div", {className:"field"},
                  e("label", null, "Whisper speed profile"),
                  e("select", {value:settingsDraft.models?.faster_whisper_profile || "auto", onChange:ev=>setSettingsDraft({...settingsDraft, models:{...settingsDraft.models, faster_whisper_profile:ev.target.value}})},
                    (currentSettings.faster_whisper_profiles || []).map(item => e("option", {key:item.value, value:item.value}, item.label))
                  )
                ) : null,
                settingsDraft ? e("div", {className:"field"},
                  e("label", null, "Direct English translation model"),
                  e("select", {value:settingsDraft.models?.literal_translation || "", onChange:ev=>setSettingsDraft({...settingsDraft, models:{...settingsDraft.models, literal_translation:ev.target.value}})},
                    (currentSettings.translation_models || []).map(item => e("option", {key:item, value:item}, item || "Default"))
                  )
                ) : null,
                settingsDraft ? e("div", {className:"field"},
                  e("label", null, "Context-applied English model"),
                  e("select", {value:settingsDraft.models?.adapted_translation || "", onChange:ev=>setSettingsDraft({...settingsDraft, models:{...settingsDraft.models, adapted_translation:ev.target.value}})},
                    (currentSettings.translation_models || []).map(item => e("option", {key:item, value:item}, item || "Default"))
                  )
                ) : null,
                settingsDraft ? e("div", {className:"guided-grid"},
                  e("div", {className:"field"},
                    e("label", null, "Japanese model cache"),
                    e("select", {value:settingsDraft.cache_paths?.hf_hub_cache || "", onChange:ev=>setSettingsDraft({...settingsDraft, cache_paths:{...settingsDraft.cache_paths, hf_hub_cache:ev.target.value}})},
                      (currentSettings.cache_options || [""]).map(item => e("option", {key:item || "default", value:item}, item || "Default Hugging Face cache"))
                    )
                  ),
                  e("button", {className:"secondary", onClick:chooseCacheFolder}, "Pick cache folder")
                ) : null,
                e("div", {className:"button-row"},
                  e("button", {onClick:saveSettings}, "Save settings"),
                  e("button", {className:"secondary", onClick:()=>post("/api/settings/use-recommended", {}, data=>setSettingsDraft(data))}, "Use Gemma e2b"),
                  e("button", {className:"secondary", onClick:()=>post("/api/settings/download-recommended", {})}, "Download Gemma"),
                  e("button", {className:"danger", onClick:()=>post("/api/settings/reset", {}, data=>setSettingsDraft(data))}, "Defaults")
                )
              )
            ),
            e("section", {className:"panel"},
              e("div", {className:"panel-head"}, e("strong", null, "Load existing subtitles"), e("button", {className:"secondary", onClick:importExisting}, "Create job from these files")),
              e("div", {className:"panel-body stack"},
                e("p", {className:"section-note"}, "Use this when subtitles already exist. Add a video plus Japanese/English SRT files, or attach one SRT to the selected job."),
                e("div", {className:"guided-grid"},
                  e("input", {value:importDraft.video || "", readOnly:true, placeholder:"Video file to link"}),
                  e("button", {className:"secondary", onClick:()=>api("/api/pick-files").then(d => d.paths?.[0] && setImportDraft({...importDraft, video:d.paths[0]})).catch(err=>setError(err.message))}, "Pick video")
                ),
                e("div", {className:"guided-grid"},
                  e("input", {value:importDraft.japanese || "", readOnly:true, placeholder:"Japanese SRT"}),
                  e("button", {className:"secondary", onClick:()=>chooseSubtitle("japanese")}, "Pick Japanese")
                ),
                e("div", {className:"guided-grid"},
                  e("input", {value:importDraft.direct || "", readOnly:true, placeholder:"Direct English translation SRT"}),
                  e("button", {className:"secondary", onClick:()=>chooseSubtitle("direct")}, "Pick direct")
                ),
                e("div", {className:"guided-grid"},
                  e("input", {value:importDraft.easy || "", readOnly:true, placeholder:"Context-applied English SRT"}),
                    e("button", {className:"secondary", onClick:()=>chooseSubtitle("easy")}, "Pick context-applied")
                ),
                e("div", {className:"guided-grid"},
                  e("input", {value:importDraft.reference || "", readOnly:true, placeholder:"Reference SRT"}),
                  e("button", {className:"secondary", onClick:()=>chooseSubtitle("reference")}, "Pick reference")
                ),
                e("div", {className:"button-row"}, [
                  ["ja", "japanese", "Attach Japanese"],
                  ["direct", "direct", "Attach direct English translation"],
                  ["easy", "easy", "Attach context-applied English"],
                  ["reference", "reference", "Attach reference"]
                ].map(([role, key, label]) => e("button", {key:role, className:"secondary", disabled:!selectedJobId || !importDraft[key], onClick:()=>post("/api/job/attach", {job_id:selectedJobId, role, path:importDraft[key]})}, label)))
              )
            ),
            e("section", {className:"panel"},
              e("div", {className:"panel-head"}, e("strong", null, "Health"), e("button", {className:"secondary", onClick:()=>api("/api/health").then(setHealth).catch(err=>setError(err.message))}, "Check setup")),
              e("pre", null, health ? [health.summary, ...(health.checks || []).map(item => `[${item.status}] ${item.name}: ${item.detail}`)].join("\\n") : "Health check output appears here.")
            ),
            e("section", {className:"panel"},
              e("div", {className:"panel-head"}, e("strong", null, "Redo log"), e("span", null, status.rebuild_running ? "Live" : "Idle")),
              e("pre", null, status.rebuild_log || "Redo output appears here.")
            )
          )
        ),
        e("footer", null, e("span", null, "Built by Amir. Follow Amir at followamir.com."), e("a", {href:"https://www.paypal.com/donate/?hosted_button_id=2U2GXSKFJKJCA"}, "Donate"))
      );
    }
    ReactDOM.createRoot(document.getElementById("root")).render(e(App));
  </script>
</body>
</html>"""


def count_sources(targets: list[Path], recursive: bool) -> int:
    total = 0
    for target in targets:
        if target.is_file() and target.suffix.lower() in VIDEO_EXTENSIONS:
            total += 1
        elif target.is_dir():
            iterator = target.rglob("*") if recursive else target.iterdir()
            total += sum(1 for path in iterator if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS)
    return total


def count_completed_outputs(targets: list[Path], recursive: bool) -> int:
    total = 0
    for target in targets:
        if target.is_file():
            total += int((target.parent / f"{target.stem}.raw.meta.json").exists())
        elif target.is_dir():
            iterator = target.rglob("*.raw.meta.json") if recursive else target.glob("*.raw.meta.json")
            total += sum(1 for _ in iterator)
    return total


class WebTranscriberState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.process: subprocess.Popen[str] | None = None
        self.targets: list[Path] = []
        self.recursive = True
        self.total = 0
        self.log = ""
        self.state = "idle"
        self.current = ""

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            running = self.process is not None and self.process.poll() is None
            completed = count_completed_outputs(self.targets, self.recursive) if self.targets else 0
            if self.process is not None and not running and self.state == "running":
                self.state = "complete" if self.process.returncode == 0 else "failed"
            return {
                "running": running,
                "state": self.state,
                "completed": min(completed, self.total),
                "total": self.total,
                "current": self.current,
                "log": self.log[-6000:],
            }

    def start(self, targets: list[Path], profile: str, recursive: bool) -> dict[str, Any]:
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                raise RuntimeError("A transcription run is already active.")
            missing = [str(path) for path in targets if not path.exists()]
            if missing:
                raise RuntimeError(f"Path not found: {missing[0]}")
            self.targets = targets
            self.recursive = recursive
            self.total = count_sources(targets, recursive)
            if self.total == 0:
                raise RuntimeError("No supported video files found.")
            self.log = ""
            self.state = "running"
            self.current = str(targets[0]) if len(targets) == 1 else f"{len(targets)} selected files"
            commands = [self._command(path, profile, recursive) for path in targets]
            thread = threading.Thread(target=self._run_commands, args=(commands,), daemon=True)
            thread.start()
        return self.snapshot()

    def cancel(self) -> dict[str, Any]:
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                self.process.terminate()
                self.state = "cancelled"
        return self.snapshot()

    def _command(self, target: Path, profile: str, recursive: bool) -> list[str]:
        output_root = target if target.is_dir() else target.parent
        command = [
            sys.executable,
            "-m",
            "local_subtitle_stack.cli",
            "transcribe-batch",
            "--input-dir",
            str(target),
            "--output-dir",
            str(output_root),
            "--language",
            "en",
            "--profile",
            profile,
            "--low-memory-policy",
            "downgrade",
        ]
        if recursive and target.is_dir():
            command.append("--recursive")
        return command

    def _run_commands(self, commands: list[list[str]]) -> None:
        return_code = 0
        for command in commands:
            with self.lock:
                if self.state == "cancelled":
                    return
                self.current = command[command.index("--input-dir") + 1]
                self.process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    creationflags=no_window_creationflags(),
                )
                process = self.process
            assert process.stdout is not None
            for line in process.stdout:
                with self.lock:
                    self.log += line
            return_code = process.wait()
            if return_code != 0:
                break
        with self.lock:
            self.state = "complete" if return_code == 0 and self.state != "cancelled" else "failed"


STATE = WebTranscriberState()


def _scene_contexts(data: list[dict[str, Any]] | None) -> list[SceneContextBlock]:
    return [
        SceneContextBlock(
            start_seconds=float(item.get("start_seconds", 0)),
            end_seconds=float(item.get("end_seconds", 0)),
            notes=str(item.get("notes", "")).strip(),
        )
        for item in (data or [])
        if str(item.get("notes", "")).strip()
    ]


class WebServiceState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.service = build_service()
        self.worker_process: subprocess.Popen[str] | None = None
        self.rebuild_process: subprocess.Popen[str] | None = None
        self.rebuild_log = ""

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "jobs": self.service.status_rows(),
                "pause_requested": self.service.store.pause_requested(),
                "worker_running": self.worker_pid() is not None,
                "rebuild_running": self._running(self.rebuild_process),
                "rebuild_log": self.rebuild_log[-6000:],
                "settings": self.settings(),
            }

    def settings(self) -> dict[str, Any]:
        config = self.service.config
        ollama_models = self._ollama_model_names()
        asr_options = [
            {"engine": "kotoba", "model": "kotoba-tech/kotoba-whisper-v2.2", "label": "Kotoba Japanese quality"},
            {"engine": config.models.asr_engine, "model": config.models.asr, "label": "Current Japanese model"},
            *[
                {"engine": item.engine, "model": item.model_id, "label": item.label}
                for item in ranked_asr_candidates()
            ],
            {"engine": "faster-whisper", "model": "large-v3", "label": "Fast local Whisper large-v3"},
        ]
        return {
            "profiles": sorted(config.profiles),
            "default_profile": config.default_profile,
            "models": asdict(config.models),
            "cache_paths": asdict(config.cache_paths),
            "recommended_translation_model": RECOMMENDED_TRANSLATION_MODEL,
            "asr_options": _unique_options(asr_options, "model"),
            "faster_whisper_profiles": [
                {"value": "auto", "label": "Auto choose best fit"},
                {"value": "high", "label": "Highest accuracy GPU"},
                {"value": "balanced", "label": "Balanced GPU"},
                {"value": "low_gpu", "label": "Low VRAM GPU"},
                {"value": "cpu_fallback", "label": "CPU fallback"},
            ],
            "translation_models": _unique_strings(
                [
                    config.models.literal_translation,
                    config.models.adapted_translation,
                    RECOMMENDED_TRANSLATION_MODEL,
                    *ollama_models,
                ]
            ),
            "cache_options": _unique_strings(
                [
                    config.cache_paths.hf_hub_cache,
                    str(Path.cwd() / ".cache" / "whisper-models"),
                    "",
                ]
            ),
        }

    def _ollama_model_names(self) -> list[str]:
        try:
            return list(self.service.ollama.list_model_details())
        except Exception:
            return []

    def job(self, job_id: str) -> dict[str, Any]:
        _job_dir, manifest = self.service.load_job(job_id)
        subtitle_error = None
        try:
            subtitle_files = {key: str(value) for key, value in self.service.subtitle_file_paths(job_id).items()}
        except Exception as exc:
            subtitle_files = {}
            subtitle_error = str(exc)
        return {
            "manifest": manifest.to_dict(),
            "preview": self.service.preview_rows(job_id),
            "subtitle_files": subtitle_files,
            "subtitle_file_error": subtitle_error,
        }

    def enqueue(self, payload: dict[str, Any]) -> dict[str, Any]:
        targets = [Path(value).expanduser() for value in payload.get("targets", [])]
        profile = str(payload.get("profile") or self.service.config.default_profile)
        recursive = bool(payload.get("recursive", True))
        include_adapted = bool(payload.get("include_adapted_english", True))
        prefer_fast = bool(payload.get("prefer_fast_translation", False))
        batch_label = _optional_str(payload.get("batch_label"))
        context = _optional_str(payload.get("context"))
        scene_contexts = _scene_contexts(payload.get("scene_contexts"))
        manifests = []
        skipped: list[Path] = []
        for target in targets:
            if target.is_dir():
                added, folder_skipped = self.service.enqueue_folder(
                    folder=target,
                    profile=profile,
                    series=batch_label,
                    context=context,
                    scene_contexts=scene_contexts,
                    recursive=recursive,
                    include_adapted_english=include_adapted,
                    prefer_fast_translation=prefer_fast,
                )
                manifests.extend(added)
                skipped.extend(folder_skipped)
            else:
                added, file_skipped = self.service.enqueue_many(
                    [target],
                    profile=profile,
                    series=batch_label,
                    context=context,
                    scene_contexts=scene_contexts,
                    include_adapted_english=include_adapted,
                    prefer_fast_translation=prefer_fast,
                )
                manifests.extend(added)
                skipped.extend(file_skipped)
        return {"queued": [item.job_id for item in manifests], "skipped": [str(path) for path in skipped]}

    def import_existing(self, payload: dict[str, Any]) -> dict[str, Any]:
        manifest = self.service.import_existing(
            profile=str(payload.get("profile") or self.service.config.default_profile),
            video=_path_or_none(payload.get("video")),
            primary_subtitle=_path_or_none(payload.get("primary_subtitle")),
            japanese=_path_or_none(payload.get("japanese")),
            direct=_path_or_none(payload.get("direct")),
            easy=_path_or_none(payload.get("easy")),
            reference=_path_or_none(payload.get("reference")),
            series=_optional_str(payload.get("batch_label")),
            context=_optional_str(payload.get("context")),
            scene_contexts=_scene_contexts(payload.get("scene_contexts")),
            include_adapted_english=bool(payload.get("include_adapted_english", True)),
            prefer_fast_translation=bool(payload.get("prefer_fast_translation", False)),
        )
        return {"job_id": manifest.job_id}

    def start_worker(self) -> dict[str, Any]:
        self.service.store.set_pause(False)
        pid = self.worker_pid()
        if pid:
            return {"message": "Worker already running", "pid": pid}
        self.worker_process = subprocess.Popen(
            [sys.executable, "-m", "local_subtitle_stack.cli", "worker"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            creationflags=no_window_creationflags(),
        )
        return {"message": "Worker started", "pid": self.worker_process.pid}

    def worker_pid(self) -> int | None:
        pid = self.service.store.active_worker_pid()
        if pid:
            return pid
        if self._running(self.worker_process):
            return self.worker_process.pid
        return None

    def stop_worker(self) -> dict[str, Any]:
        self.service.store.set_pause(True)
        return {"message": "Worker will stop after the next safe step"}

    def retry(self, job_id: str) -> dict[str, Any]:
        manifest = self.service.resume(job_id)
        self.start_worker()
        return {"job_id": manifest.job_id}

    def delete_job(self, job_id: str) -> dict[str, Any]:
        job_dir, manifest = self.service.store.find_job(job_id)
        if manifest.status == "working":
            raise QueueError("Stop processing before deleting a running job.")
        shutil.rmtree(job_dir)
        return {"deleted": job_id}

    def save_notes(self, payload: dict[str, Any]) -> dict[str, Any]:
        manifest = self.service.save_job_notes(
            str(payload["job_id"]),
            batch_label=_optional_str(payload.get("batch_label")),
            overall_context=_optional_str(payload.get("context")),
            scene_contexts=_scene_contexts(payload.get("scene_contexts")),
            include_adapted_english=payload.get("include_adapted_english"),
            prefer_fast_translation=payload.get("prefer_fast_translation"),
        )
        return {"job_id": manifest.job_id}

    def save_line(self, payload: dict[str, Any]) -> dict[str, Any]:
        manifest = self.service.update_subtitle_line(
            str(payload["job_id"]),
            cue_index=int(payload["cue_index"]),
            japanese_text=payload.get("japanese_text"),
            literal_english_text=payload.get("literal_english_text"),
            adapted_english_text=payload.get("adapted_english_text"),
            reference_text=payload.get("reference_text"),
        )
        return {"job_id": manifest.job_id}

    def attach(self, payload: dict[str, Any]) -> dict[str, Any]:
        manifest = self.service.attach_existing_subtitle(
            str(payload["job_id"]),
            role=str(payload["role"]),
            subtitle_path=Path(str(payload["path"])).expanduser(),
        )
        return {"job_id": manifest.job_id}

    def rebuild(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._running(self.rebuild_process):
            raise RuntimeError("Redo English is already running.")
        job_id = str(payload["job_id"])
        self.save_notes(payload)
        command = [sys.executable, "-m", "local_subtitle_stack.cli", "rebuild-english", job_id]
        start = _optional_str(payload.get("start_timecode"))
        end = _optional_str(payload.get("end_timecode"))
        if start and end:
            command = [
                sys.executable,
                "-m",
                "local_subtitle_stack.cli",
                "rebuild-english-range",
                job_id,
                "--start",
                start,
                "--end",
                end,
            ]
        self.rebuild_log = ""
        self.rebuild_process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
            creationflags=no_window_creationflags(),
        )
        threading.Thread(target=self._drain_rebuild_log, daemon=True).start()
        return {"message": "Redo English started", "pid": self.rebuild_process.pid}

    def open_target(self, payload: dict[str, Any]) -> dict[str, Any]:
        job_id = str(payload["job_id"])
        action = str(payload["action"])
        if action == "review":
            paths = self.service.open_review(job_id)
            return {"opened": [str(path) for path in paths]}
        if action == "folder":
            return {"opened": str(self.service.open_output_folder(job_id))}
        path = self.service.open_subtitle_file(job_id, action)
        return {"opened": str(path)}

    def health(self) -> dict[str, Any]:
        return self.service.health_check()

    def save_settings(self, payload: dict[str, Any]) -> dict[str, Any]:
        models = payload.get("models", {})
        cache_paths = payload.get("cache_paths", {})
        self.service.config.models.asr_engine = _optional_str(models.get("asr_engine")) or ModelConfig().asr_engine
        self.service.config.models.asr = _optional_str(models.get("asr")) or ModelConfig().asr
        self.service.config.models.faster_whisper_profile = (
            _optional_str(models.get("faster_whisper_profile")) or ModelConfig().faster_whisper_profile
        )
        self.service.config.models.literal_translation = (
            _optional_str(models.get("literal_translation")) or ModelConfig().literal_translation
        )
        self.service.config.models.adapted_translation = (
            _optional_str(models.get("adapted_translation")) or ModelConfig().adapted_translation
        )
        self.service.config.cache_paths.hf_hub_cache = (
            _optional_str(cache_paths.get("hf_hub_cache")) or CachePaths().hf_hub_cache
        )
        save_config(self.service.config)
        return self.settings()

    def reset_settings(self) -> dict[str, Any]:
        self.service.config.models = ModelConfig()
        self.service.config.cache_paths = CachePaths()
        save_config(self.service.config)
        return self.settings()

    def use_recommended_model(self) -> dict[str, Any]:
        self.service.config.models.literal_translation = RECOMMENDED_TRANSLATION_MODEL
        self.service.config.models.adapted_translation = RECOMMENDED_TRANSLATION_MODEL
        save_config(self.service.config)
        return self.settings()

    def pull_recommended_model(self) -> dict[str, Any]:
        threading.Thread(target=self.service.ollama.pull_model, args=(RECOMMENDED_TRANSLATION_MODEL,), daemon=True).start()
        return {"message": f"Downloading {RECOMMENDED_TRANSLATION_MODEL}"}

    def upload_subtitle(self, payload: dict[str, Any]) -> dict[str, Any]:
        filename = Path(str(payload.get("filename") or "dropped.srt")).name
        if not filename.lower().endswith(".srt"):
            raise QueueError("Drop an .srt subtitle file.")
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", filename)
        content = str(payload.get("content") or "")
        if not content.strip():
            raise QueueError("Dropped subtitle file is empty.")
        upload_dir = self.service.config.queue_root_path / "web-imports"
        upload_dir.mkdir(parents=True, exist_ok=True)
        target = upload_dir / safe_name
        if target.exists():
            target = upload_dir / f"{Path(safe_name).stem}-{len(list(upload_dir.glob(Path(safe_name).stem + '*')))}.srt"
        target.write_text(content, encoding="utf-8-sig")
        role = str(payload.get("role") or "direct")
        job_id = _optional_str(payload.get("job_id"))
        if job_id:
            manifest = self.service.attach_existing_subtitle(job_id, role=role, subtitle_path=target)
            return {"mode": "attached", "job_id": manifest.job_id, "path": str(target)}
        if role not in {"ja", "direct"}:
            raise QueueError("Select an existing job before dropping context-applied English or reference subtitles.")
        kwargs = {"primary_subtitle": target, "profile": str(payload.get("profile") or self.service.config.default_profile)}
        if role == "ja":
            kwargs["japanese"] = target
        else:
            kwargs["direct"] = target
        manifest = self.service.import_existing(**kwargs)
        return {"mode": "created", "job_id": manifest.job_id, "path": str(target)}

    def _drain_rebuild_log(self) -> None:
        process = self.rebuild_process
        if process is None or process.stdout is None:
            return
        for line in process.stdout:
            with self.lock:
                self.rebuild_log += line

    def _running(self, process: subprocess.Popen[str] | None) -> bool:
        return process is not None and process.poll() is None


APP_STATE = WebServiceState()


def _optional_str(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _path_or_none(value: Any) -> Path | None:
    text = _optional_str(value)
    return Path(text).expanduser() if text else None


def _unique_strings(values: list[str | None]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _unique_options(values: list[dict[str, str]], key: str) -> list[dict[str, str]]:
    seen: set[str] = set()
    result: list[dict[str, str]] = []
    for item in values:
        text = item.get(key, "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(item)
    return result


def format_bytes(value: int) -> str:
    size = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{value} B"
        size /= 1024
    return f"{value} B"


def ollama_manifest_path(storage_root: str, model_name: str) -> str:
    if storage_root.startswith("Remote Ollama host:"):
        return ""
    model, _, tag = model_name.partition(":")
    tag = tag or "latest"
    parts = [part for part in model.split("/") if part]
    if len(parts) == 1:
        host, namespace, name = "registry.ollama.ai", "library", parts[0]
    elif len(parts) == 2:
        host, namespace, name = "registry.ollama.ai", parts[0], parts[1]
    else:
        host, namespace, name = parts[0], parts[-2], parts[-1]
    return str(Path(storage_root) / "manifests" / host / namespace / name / tag)


def ollama_blob_path(storage_root: str, digest: str) -> str:
    if storage_root.startswith("Remote Ollama host:"):
        return ""
    if not digest.startswith("sha256:"):
        return ""
    return str(Path(storage_root) / "blobs" / digest.replace(":", "-", 1))


def ollama_blob_paths(storage_root: str, manifest_path: str, fallback_digest: str) -> list[str]:
    paths: list[str] = []
    if manifest_path and Path(manifest_path).exists():
        try:
            manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = {}
        for item in manifest.get("layers", []):
            path = ollama_blob_path(storage_root, str(item.get("digest") or ""))
            if path:
                paths.append(path)
    fallback = ollama_blob_path(storage_root, fallback_digest)
    return _unique_strings([path for path in [*paths, fallback] if path])


def model_storage_snapshot() -> dict[str, Any]:
    config = APP_STATE.service.config
    ollama = APP_STATE.service.ollama
    try:
        details = ollama.list_model_details()
    except Exception:
        details = {}

    def selected(label: str, name: str) -> dict[str, Any]:
        detail = details.get(name)
        size = detail.get("size") if detail else None
        digest = str(detail.get("digest") or "") if detail else ""
        storage = ollama.model_storage_root()
        manifest_path = ollama_manifest_path(storage, name) if detail else ""
        return {
            "label": label,
            "name": name,
            "size": format_bytes(int(size)) if isinstance(size, int) else "",
            "manifest_path": manifest_path,
            "blob_paths": ollama_blob_paths(storage, manifest_path, digest),
        }

    return {
        "storage": ollama.model_storage_root(),
        "hf_cache": config.cache_paths.hf_hub_cache,
        "selected": [
            selected("Direct English translation", config.models.literal_translation),
            selected("Context-applied English", config.models.adapted_translation),
        ],
    }


def close_other_web_ui_processes() -> None:
    current_pid = os.getpid()
    parent_pids = {process.pid for process in psutil.Process(current_pid).parents()}
    old_processes = []
    for process in psutil.process_iter(["pid", "cmdline"]):
        pid = process.info.get("pid")
        if pid == current_pid or pid in parent_pids:
            continue
        command = " ".join(process.info.get("cmdline") or [])
        if "local_subtitle_stack" not in command or "web-ui" not in command:
            continue
        try:
            process.terminate()
            old_processes.append(process)
        except psutil.Error:
            continue
    _, alive = psutil.wait_procs(old_processes, timeout=2)
    for process in alive:
        try:
            process.kill()
        except psutil.Error:
            continue


def _run_windows_picker(script: str) -> list[str]:
    completed = subprocess.run(
        ["powershell.exe", "-NoProfile", "-STA", "-ExecutionPolicy", "Bypass", "-Command", script],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "Windows picker failed.")
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def pick_folder() -> str:
    paths = _run_windows_picker(
        """
        Add-Type -AssemblyName System.Windows.Forms
        $owner = New-Object System.Windows.Forms.Form
        $owner.TopMost = $true
        $owner.WindowState = 'Minimized'
        $owner.ShowInTaskbar = $false
        $owner.Show() | Out-Null
        $dialog = New-Object System.Windows.Forms.FolderBrowserDialog
        $dialog.Description = 'Select folder to transcribe'
        if ($dialog.ShowDialog($owner) -eq [System.Windows.Forms.DialogResult]::OK) {
            [Console]::Out.WriteLine($dialog.SelectedPath)
        }
        $owner.Dispose()
        """
    )
    return paths[0] if paths else ""


def pick_files() -> list[str]:
    return _run_windows_picker(
        """
        Add-Type -AssemblyName System.Windows.Forms
        $owner = New-Object System.Windows.Forms.Form
        $owner.TopMost = $true
        $owner.WindowState = 'Minimized'
        $owner.ShowInTaskbar = $false
        $owner.Show() | Out-Null
        $dialog = New-Object System.Windows.Forms.OpenFileDialog
        $dialog.Title = 'Select videos to transcribe'
        $dialog.Multiselect = $true
        $dialog.Filter = 'Video files|*.mp4;*.mkv;*.avi;*.mov;*.wmv;*.m4v;*.webm|All files|*.*'
        if ($dialog.ShowDialog($owner) -eq [System.Windows.Forms.DialogResult]::OK) {
            foreach ($name in $dialog.FileNames) {
                [Console]::Out.WriteLine($name)
            }
        }
        $owner.Dispose()
        """
    )


def pick_subtitle_file() -> str:
    paths = _run_windows_picker(
        """
        Add-Type -AssemblyName System.Windows.Forms
        $owner = New-Object System.Windows.Forms.Form
        $owner.TopMost = $true
        $owner.WindowState = 'Minimized'
        $owner.ShowInTaskbar = $false
        $owner.Show() | Out-Null
        $dialog = New-Object System.Windows.Forms.OpenFileDialog
        $dialog.Title = 'Select subtitle file'
        $dialog.Multiselect = $false
        $dialog.Filter = 'SRT subtitle files|*.srt|All files|*.*'
        if ($dialog.ShowDialog($owner) -eq [System.Windows.Forms.DialogResult]::OK) {
            [Console]::Out.WriteLine($dialog.FileName)
        }
        $owner.Dispose()
        """
    )
    return paths[0] if paths else ""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        try:
            parsed = urlparse(self.path)
            path = parsed.path
            if path == "/":
                self._send_html(HTML)
            elif path == "/api/status":
                self._send_json(APP_STATE.snapshot())
            elif path == "/api/models":
                self._send_json(model_storage_snapshot())
            elif path == "/api/health":
                self._send_json(APP_STATE.health())
            elif path == "/api/job":
                query = parse_qs(parsed.query)
                self._send_json(APP_STATE.job(query.get("id", [""])[0]))
            elif path == "/api/pick-folder":
                self._send_json({"path": pick_folder()})
            elif path == "/api/pick-files":
                self._send_json({"paths": pick_files()})
            elif path == "/api/pick-subtitle":
                self._send_json({"path": pick_subtitle_file()})
            else:
                self.send_error(404)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=400)

    def do_POST(self) -> None:
        try:
            payload = self._read_json()
            if self.path == "/api/start":
                targets = [Path(value).expanduser() for value in payload.get("targets", [])]
                result = APP_STATE.enqueue({**payload, "targets": [str(path) for path in targets]})
                APP_STATE.start_worker()
                self._send_json({**APP_STATE.snapshot(), **result})
            elif self.path == "/api/enqueue":
                self._send_json(APP_STATE.enqueue(payload))
            elif self.path == "/api/cancel":
                self._send_json(APP_STATE.stop_worker())
            elif self.path == "/api/import-existing":
                self._send_json(APP_STATE.import_existing(payload))
            elif self.path == "/api/upload-subtitle":
                self._send_json(APP_STATE.upload_subtitle(payload))
            elif self.path == "/api/worker/start":
                self._send_json(APP_STATE.start_worker())
            elif self.path == "/api/worker/stop":
                self._send_json(APP_STATE.stop_worker())
            elif self.path == "/api/job/retry":
                self._send_json(APP_STATE.retry(str(payload["job_id"])))
            elif self.path == "/api/job/delete":
                self._send_json(APP_STATE.delete_job(str(payload["job_id"])))
            elif self.path == "/api/job/notes":
                self._send_json(APP_STATE.save_notes(payload))
            elif self.path == "/api/job/line":
                self._send_json(APP_STATE.save_line(payload))
            elif self.path == "/api/job/attach":
                self._send_json(APP_STATE.attach(payload))
            elif self.path == "/api/job/rebuild":
                self._send_json(APP_STATE.rebuild(payload))
            elif self.path == "/api/open":
                self._send_json(APP_STATE.open_target(payload))
            elif self.path == "/api/settings/save":
                self._send_json(APP_STATE.save_settings(payload))
            elif self.path == "/api/settings/reset":
                self._send_json(APP_STATE.reset_settings())
            elif self.path == "/api/settings/use-recommended":
                self._send_json(APP_STATE.use_recommended_model())
            elif self.path == "/api/settings/download-recommended":
                self._send_json(APP_STATE.pull_recommended_model())
            else:
                self.send_error(404)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=400)

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _send_html(self, body: str) -> None:
        data = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, data: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main(argv: list[str] | None = None) -> int:
    port = int(argv[0]) if argv else int(os.environ.get("FAST_TRANSCRIBER_PORT", "8765"))
    close_other_web_ui_processes()
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"Fast Transcriber UI: {url}")
    webbrowser.open(url)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
