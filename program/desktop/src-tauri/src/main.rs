#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use anyhow::{anyhow, Context, Result};
use base64::Engine;
use chrono::{DateTime, Local};
use exif::{In, Reader, Tag};
use futures_util::StreamExt;
use minisign_verify::{PublicKey, Signature};
use reqwest::header::{HeaderValue, ACCEPT};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    fs,
    io::{BufRead, BufReader, Read},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::atomic::{AtomicBool, Ordering},
    sync::Mutex,
    sync::OnceLock,
    time::SystemTime,
};
use tauri::{AppHandle, Emitter, Manager};
use tauri_plugin_updater::{Update, UpdaterExt};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Candidate {
    method: String,
    score: f64,
    quad: Vec<[f64; 2]>,
    #[serde(rename = "originalQuad", default)]
    original_quad: Option<Vec<[f64; 2]>>,
    #[serde(rename = "manualQuad", default)]
    manual_quad: Option<Vec<[f64; 2]>>,
    metrics: serde_json::Value,
    #[serde(default)]
    source: Option<String>,
    #[serde(rename = "modelId", default)]
    model_id: Option<String>,
    #[serde(rename = "debugOnly", default)]
    debug_only: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PageRecord {
    id: String,
    name: String,
    path: String,
    #[serde(rename = "thumbPath")]
    thumb_path: Option<String>,
    #[serde(rename = "createdAt")]
    created_at: String,
    status: String,
    confidence: f64,
    #[serde(rename = "bestMethod")]
    best_method: Option<String>,
    #[serde(rename = "selectedCandidateIndex")]
    selected_candidate_index: usize,
    candidates: Vec<Candidate>,
    #[serde(rename = "activeQuad")]
    active_quad: Vec<[f64; 2]>,
    #[serde(rename = "manualQuad")]
    manual_quad: Option<Vec<[f64; 2]>>,
    #[serde(rename = "manualBaseCandidateIndex", default)]
    manual_base_candidate_index: Option<usize>,
    #[serde(rename = "previewPath")]
    preview_path: Option<String>,
    #[serde(default)]
    details: PageDetails,
    #[serde(rename = "eventSlug", default)]
    event_slug: Option<String>,
    #[serde(rename = "difficultyBucket", default)]
    difficulty_bucket: Option<String>,
    #[serde(rename = "failureTags", default)]
    failure_tags: Vec<String>,
    #[serde(rename = "bucketReason", default)]
    bucket_reason: Vec<String>,
    #[serde(rename = "reviewTags", default)]
    review_tags: Vec<String>,
    #[serde(rename = "tagVersion", default)]
    tag_version: Option<u32>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PageDetails {
    width: u32,
    height: u32,
    file_size_bytes: u64,
    captured_at: Option<String>,
    created_at: String,
    modified_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ProjectFile {
    version: u32,
    #[serde(rename = "dataStructureVersion", default)]
    data_structure_version: Option<u32>,
    name: String,
    #[serde(rename = "sourceDir")]
    source_dir: String,
    #[serde(rename = "projectPath")]
    project_path: Option<String>,
    #[serde(rename = "selectedPageId")]
    selected_page_id: Option<String>,
    #[serde(rename = "eventSlug", default)]
    event_slug: Option<String>,
    #[serde(rename = "eventName", default)]
    event_name: Option<String>,
    #[serde(rename = "tagVersion", default)]
    tag_version: Option<u32>,
    #[serde(rename = "tagSummary", default)]
    tag_summary: Option<serde_json::Value>,
    pages: Vec<PageRecord>,
}

const CURRENT_DATA_STRUCTURE_VERSION: u32 = 2;
const UPDATE_POLL_CHUNK_SIZE: usize = 64 * 1024;
const UPDATER_PUBKEY_BASE64: &str =
    "dW50cnVzdGVkIGNvbW1lbnQ6IG1pbmlzaWduIHB1YmxpYyBrZXk6IEQ1RDYwNjI4MUMzQkFDMTIKUldRU3JEc2NLQWJXMWExcTZrWDVtT2dLRmtCQURPSmVqZ0Q5cWd5bWhZd3F1cnRBbE5KWEQwS2EK";

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all(serialize = "snake_case", deserialize = "camelCase"))]
struct ExportOptions {
    output_path: String,
    export_all_pages: bool,
    include_auto_ready: bool,
    min_auto_ready_confidence: f64,
    jpeg_quality: u8,
    max_dimension: u32,
    ocr_enabled: bool,
    ocr_languages: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ExportPage {
    id: String,
    name: String,
    path: String,
    status: String,
    confidence: f64,
    quad: Vec<[f64; 2]>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all(serialize = "snake_case", deserialize = "camelCase"))]
struct ExportRequest {
    project_name: String,
    source_dir: String,
    output_path: String,
    work_dir: String,
    options: ExportOptions,
    pages: Vec<ExportPage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ExportedPage {
    id: String,
    name: String,
    image_path: String,
    pdf_path: String,
    ocr_text_path: Option<String>,
    warning: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ExportResult {
    output_path: String,
    report_path: String,
    page_count: usize,
    effective_ocr_languages: Option<String>,
    warnings: Vec<String>,
    pages: Vec<ExportedPage>,
}

#[derive(Debug, Clone, Deserialize)]
struct EngineBest {
    method: String,
    quad: Vec<[f64; 2]>,
    confidence: f64,
    #[serde(default)]
    source: Option<String>,
    #[serde(rename = "modelId", default)]
    model_id: Option<String>,
    #[serde(rename = "debugOnly", default)]
    debug_only: bool,
}

#[derive(Debug, Deserialize)]
struct RawEngineResult {
    best: Option<EngineBest>,
    candidates: Vec<Candidate>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawBatchEngineResult {
    image_path: String,
    best: Option<EngineBest>,
    #[serde(default)]
    candidates: Vec<Candidate>,
    error: Option<String>,
}

#[derive(Debug, Clone)]
struct EngineResult {
    best: EngineBest,
    candidates: Vec<Candidate>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct ScanProgressEvent {
    scan_id: u64,
    phase: String,
    processed: usize,
    total: usize,
    current_name: Option<String>,
    message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
struct UpdateInfoPayload {
    body: Option<String>,
    current_version: String,
    date: Option<String>,
    version: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
struct UpdateProgressPayload {
    percent: f64,
    total: u64,
    transferred: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
enum UpdateStatus {
    Idle,
    Checking,
    UpToDate,
    Available,
    Downloading,
    Ready,
    Unsupported,
    Error,
}

impl Default for UpdateStatus {
    fn default() -> Self {
        Self::Idle
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
struct UpdateStatePayload {
    status: UpdateStatus,
    update: Option<UpdateInfoPayload>,
    progress: Option<UpdateProgressPayload>,
    error: Option<String>,
}

#[derive(Default)]
struct UpdateManagerState {
    snapshot: UpdateStatePayload,
}

static UPDATE_MANAGER: OnceLock<Mutex<UpdateManagerState>> = OnceLock::new();

fn update_manager() -> &'static Mutex<UpdateManagerState> {
    UPDATE_MANAGER.get_or_init(|| Mutex::new(UpdateManagerState::default()))
}

fn supported_image(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|value| value.to_str()).map(|value| value.to_ascii_lowercase()),
        Some(ext) if ["jpg", "jpeg", "png", "heic"].contains(&ext.as_str())
    )
}

fn engine_dir(app: &AppHandle) -> Result<PathBuf> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let dev_dir = manifest_dir
        .parent()
        .map(|path| path.join("engine"))
        .ok_or_else(|| anyhow!("failed to resolve engine directory"))?;
    if dev_dir.exists() {
        return Ok(dev_dir);
    }
    let resource_dir = app
        .path()
        .resource_dir()
        .context("failed to resolve resource directory")?;
    if let Some(engine) = resolve_resource_path_from_base(&resource_dir, Path::new("engine")) {
        return Ok(engine);
    }
    Err(anyhow!(
        "failed to resolve engine directory inside resources rooted at {}",
        resource_dir.display()
    ))
}

fn resolve_resource_path_from_base(resource_dir: &Path, relative: &Path) -> Option<PathBuf> {
    let mut base = resource_dir.to_path_buf();
    for _ in 0..=8 {
        let candidate = base.join(relative);
        if candidate.exists() {
            return Some(candidate);
        }
        base = base.join("_up_");
    }
    None
}

fn bundled_python_candidates(engine: &Path) -> Vec<PathBuf> {
    if cfg!(target_os = "windows") {
        return vec![
            engine.join("vendor/windows/bin/python.exe"),
            engine.join("vendor/python.exe"),
        ];
    }
    vec![
        engine.join("vendor/macos/bin/python3"),
        engine.join("vendor/linux/bin/python3"),
        engine.join("vendor/bin/python3"),
    ]
}

fn python_probe_code() -> &'static str {
    "import importlib.util,sys;mods=('cv2','PIL','pypdf','torch');missing=[m for m in mods if importlib.util.find_spec(m) is None];sys.exit(0 if not missing else 1)"
}

fn explicit_python_candidates() -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    if let Ok(value) = std::env::var("SCREEN_PDF_PYTHON") {
        candidates.push(PathBuf::from(value));
    }
    if let Ok(home) = std::env::var("HOME") {
        let home = PathBuf::from(home);
        candidates.push(home.join("anaconda3/bin/python3"));
        candidates.push(home.join("anaconda3/bin/python"));
        candidates.push(home.join("miniconda3/bin/python3"));
        candidates.push(home.join("miniconda3/bin/python"));
    }
    candidates
}

fn fallback_python_commands() -> Vec<PathBuf> {
    if cfg!(target_os = "windows") {
        vec![PathBuf::from("python"), PathBuf::from("py")]
    } else {
        vec![
            PathBuf::from("python3"),
            PathBuf::from("python"),
            PathBuf::from("/usr/local/bin/python3"),
            PathBuf::from("/opt/homebrew/bin/python3"),
            PathBuf::from("/usr/bin/python3"),
        ]
    }
}

fn python_candidates(app: &AppHandle) -> Result<Vec<PathBuf>> {
    let engine = engine_dir(app)?;
    let mut candidates = Vec::new();
    candidates.extend(explicit_python_candidates());
    candidates.extend(bundled_python_candidates(&engine));
    candidates.extend(fallback_python_commands());
    candidates.retain(|candidate| !candidate.as_os_str().is_empty());
    let mut deduped = Vec::new();
    for candidate in candidates {
        if !deduped.iter().any(|value: &PathBuf| value == &candidate) {
            deduped.push(candidate);
        }
    }
    Ok(deduped)
}

fn python_supports_engine(python: &Path, engine: &Path) -> bool {
    let output = Command::new(python)
        .arg("-c")
        .arg(python_probe_code())
        .current_dir(engine)
        .output();

    match output {
        Ok(result) => result.status.success(),
        Err(_) => false,
    }
}

fn python_cmd(app: &AppHandle) -> Result<PathBuf> {
    static PYTHON_CACHE: OnceLock<Option<PathBuf>> = OnceLock::new();

    let resolved = PYTHON_CACHE.get_or_init(|| {
        let engine = match engine_dir(app) {
            Ok(value) => value,
            Err(_) => return None,
        };
        python_candidates(app).ok().and_then(|candidates| {
            candidates
                .into_iter()
                .find(|candidate| python_supports_engine(candidate, &engine))
        })
    });

    if let Some(path) = resolved {
        return Ok(path.clone());
    }

    let candidates = python_candidates(app)?
        .into_iter()
        .map(|value| value.display().to_string())
        .collect::<Vec<_>>()
        .join(", ");
    Err(anyhow!(
        "no usable python interpreter found for engine dependencies; checked: {candidates}"
    ))
}

fn configure_engine_command(command: &mut Command, app: &AppHandle) {
    command.env("SCREEN_PDF_DEBUG_DUAL_MODEL", "1");
    if let Ok(resource_dir) = app.path().resource_dir() {
        if let Some(model_dir) = resolve_resource_path_from_base(
            &resource_dir,
            Path::new("models").join("runtime").as_path(),
        ) {
            command.env("SCREEN_PDF_MODEL_DIR", &model_dir);
            let model_path = model_dir.join("deep_screen_v1_debug.pt");
            if model_path.exists() {
                command.env("SCREEN_PDF_DEEP_SCREEN_V1_MODEL", model_path);
            }
        }
    }
}

fn scan_cancel_flag() -> &'static AtomicBool {
    static SCAN_CANCEL: OnceLock<AtomicBool> = OnceLock::new();
    SCAN_CANCEL.get_or_init(|| AtomicBool::new(false))
}

fn clear_runtime_temp_artifacts() -> Result<()> {
    let runtime_dir = std::env::temp_dir().join("screen-pdf");
    if runtime_dir.exists() {
        fs::remove_dir_all(&runtime_dir)
            .with_context(|| format!("failed to clear {}", runtime_dir.display()))?;
    }
    Ok(())
}

fn emit_scan_progress(app: &AppHandle, event: ScanProgressEvent) {
    let _ = app.emit("scan-progress", event);
}

fn lock_update_manager() -> Result<std::sync::MutexGuard<'static, UpdateManagerState>, String> {
    update_manager()
        .lock()
        .map_err(|_| "update manager lock poisoned".to_string())
}

fn current_update_snapshot(current_version: &str) -> UpdateStatePayload {
    let snapshot = lock_update_manager()
        .map(|manager| manager.snapshot.clone())
        .unwrap_or_default();
    normalize_update_snapshot(snapshot, current_version)
}

fn normalize_update_snapshot(
    mut snapshot: UpdateStatePayload,
    current_version: &str,
) -> UpdateStatePayload {
    if let Some(update) = snapshot.update.as_mut() {
        if update.current_version.is_empty() {
            update.current_version = current_version.to_string();
        }
    }
    snapshot
}

async fn fetch_update<R: tauri::Runtime>(app: &AppHandle<R>) -> Result<Option<Update>, String> {
    app.updater()
        .map_err(|err| format!("build updater failed: {err}"))?
        .check()
        .await
        .map_err(|err| format!("check update failed: {err}"))
}

fn to_update_info_payload(current_version: &str, update: &Update) -> UpdateInfoPayload {
    UpdateInfoPayload {
        body: update.body.clone(),
        current_version: current_version.to_string(),
        date: update.date.map(|value| value.to_string()),
        version: update.version.clone(),
    }
}

fn update_download_progress(
    transferred: u64,
    total: u64,
    update: UpdateInfoPayload,
) -> Result<(), String> {
    let percent = if total > 0 {
        ((transferred as f64 / total as f64) * 100.0).clamp(0.0, 100.0)
    } else {
        0.0
    };
    let mut manager = lock_update_manager()?;
    manager.snapshot = UpdateStatePayload {
        status: UpdateStatus::Downloading,
        update: Some(update),
        progress: Some(UpdateProgressPayload {
            percent,
            total,
            transferred,
        }),
        error: None,
    };
    Ok(())
}

fn decode_base64_to_string(value: &str) -> Result<String, String> {
    let decoded = base64::engine::general_purpose::STANDARD
        .decode(value)
        .map_err(|err| format!("decode base64 failed: {err}"))?;
    std::str::from_utf8(&decoded)
        .map_err(|_| "decode utf-8 failed".to_string())
        .map(|text| text.to_string())
}

fn verify_update_signature(data: &[u8], release_signature: &str) -> Result<(), String> {
    let pub_key = decode_base64_to_string(UPDATER_PUBKEY_BASE64)?;
    let public_key = PublicKey::decode(&pub_key)
        .map_err(|err| format!("decode updater public key failed: {err}"))?;
    let signature_text = decode_base64_to_string(release_signature)?;
    let signature = Signature::decode(&signature_text)
        .map_err(|err| format!("decode update signature failed: {err}"))?;
    public_key
        .verify(data, &signature, true)
        .map_err(|err| format!("verify update signature failed: {err}"))?;
    Ok(())
}

async fn download_update_bytes(update: &Update, current_version: &str) -> Result<Vec<u8>, String> {
    let mut headers = update.headers.clone();
    if !headers.contains_key(ACCEPT) {
        headers.insert(ACCEPT, HeaderValue::from_static("application/octet-stream"));
    }

    let mut request = reqwest::ClientBuilder::new().user_agent("screen-pdf-desktop-updater");
    if let Some(timeout) = update.timeout {
        request = request.timeout(timeout);
    }
    if update.no_proxy {
        request = request.no_proxy();
    } else if let Some(ref proxy) = update.proxy {
        let parsed = reqwest::Proxy::all(proxy.as_str())
            .map_err(|err| format!("configure updater proxy failed: {err}"))?;
        request = request.proxy(parsed);
    }

    let response = request
        .build()
        .map_err(|err| format!("build updater client failed: {err}"))?
        .get(update.download_url.clone())
        .headers(headers)
        .send()
        .await
        .map_err(|err| format!("download update failed: {err}"))?;

    if !response.status().is_success() {
        return Err(format!(
            "download request failed with status: {}",
            response.status()
        ));
    }

    let total = response.content_length().unwrap_or(0);
    let mut transferred = 0_u64;
    let mut buffer = Vec::with_capacity(total.min(UPDATE_POLL_CHUNK_SIZE as u64) as usize);
    let mut stream = response.bytes_stream();

    while let Some(next) = stream.next().await {
        let chunk = next.map_err(|err| format!("read update chunk failed: {err}"))?;
        transferred += chunk.len() as u64;
        buffer.extend_from_slice(&chunk);
        update_download_progress(
            transferred,
            total,
            to_update_info_payload(current_version, update),
        )?;
    }

    verify_update_signature(&buffer, &update.signature)?;
    Ok(buffer)
}

async fn download_and_install_update(update: Update, current_version: &str) -> Result<(), String> {
    let bytes = download_update_bytes(&update, current_version).await?;
    update
        .install(bytes)
        .map_err(|err| format!("install update failed: {err}"))
}

fn candidate_export_quad(page: &PageRecord) -> Vec<[f64; 2]> {
    page.manual_quad
        .clone()
        .unwrap_or_else(|| page.active_quad.clone())
}

fn exportable_pages<'a>(pages: &'a [PageRecord], options: &ExportOptions) -> Vec<&'a PageRecord> {
    pages
        .iter()
        .filter(|page| {
            if options.export_all_pages {
                return page.status != "error";
            }
            if page.status == "reviewed" {
                return true;
            }
            if page.status == "needs_review" {
                return true;
            }
            if page.status == "auto_ready" && options.include_auto_ready {
                return page.confidence >= options.min_auto_ready_confidence;
            }
            false
        })
        .collect()
}

#[cfg_attr(not(test), allow(dead_code))]
fn resolve_ocr_languages(requested: &str, available: &[String]) -> String {
    let requested_values = requested
        .split('+')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .collect::<Vec<_>>();

    let mut resolved = requested_values
        .into_iter()
        .filter(|lang| available.iter().any(|value| value == lang))
        .map(str::to_string)
        .collect::<Vec<_>>();

    if resolved.is_empty() && available.iter().any(|value| value == "eng") {
        resolved.push("eng".to_string());
    }
    if resolved.is_empty() {
        if let Some(first) = available.first() {
            resolved.push(first.clone());
        }
    }

    resolved.join("+")
}

fn export_work_dir(output_path: &Path) -> PathBuf {
    let stem = output_path
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("screen-pdf-export");
    output_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(format!("{stem}_export"))
}

fn output_directory(output_path: &Path) -> PathBuf {
    output_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .to_path_buf()
}

fn runtime_temp_dir() -> PathBuf {
    std::env::temp_dir().join("screen-pdf")
}

fn thumbnail_dir() -> PathBuf {
    runtime_temp_dir().join("thumbs")
}

fn thumbnail_file_name(path: &Path, stamp: u64) -> String {
    let stem = path
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("page");
    format!("{stem}-{stamp}.jpg")
}

fn generate_thumbnail(path: &Path) -> Option<String> {
    let stamp = modified_time(path)
        .duration_since(SystemTime::UNIX_EPOCH)
        .ok()
        .map(|value| value.as_secs())
        .unwrap_or(0);
    let dir = thumbnail_dir();
    if fs::create_dir_all(&dir).is_err() {
        return None;
    }
    let output_path = dir.join(thumbnail_file_name(path, stamp));
    if !output_path.exists() {
        let image = image::open(path).ok()?;
        let thumbnail = image.thumbnail(640, 360).to_rgb8();
        if thumbnail.save(&output_path).is_err() {
            return None;
        }
    }
    Some(output_path.to_string_lossy().to_string())
}

fn open_output_directory(output_path: &Path) -> Result<()> {
    let directory = output_directory(output_path);

    #[cfg(target_os = "macos")]
    let status = Command::new("open")
        .arg(&directory)
        .status()
        .with_context(|| format!("failed to launch Finder for {}", directory.display()))?;

    #[cfg(target_os = "windows")]
    let status = Command::new("explorer")
        .arg(&directory)
        .status()
        .with_context(|| format!("failed to launch Explorer for {}", directory.display()))?;

    #[cfg(all(unix, not(target_os = "macos")))]
    let status = Command::new("xdg-open")
        .arg(&directory)
        .status()
        .with_context(|| format!("failed to open directory {}", directory.display()))?;

    if !status.success() {
        return Err(anyhow!(
            "failed to open output directory {}",
            directory.display()
        ));
    }

    Ok(())
}

fn parse_engine_detect_output(output: &[u8]) -> Result<RawEngineResult> {
    let text = String::from_utf8_lossy(output);
    if let Ok(parsed) = serde_json::from_str::<RawEngineResult>(&text) {
        return Ok(parsed);
    }

    for line in text.lines().rev() {
        let trimmed = line.trim();
        if trimmed.starts_with('{') && trimmed.ends_with('}') {
            if let Ok(parsed) = serde_json::from_str::<RawEngineResult>(trimmed) {
                return Ok(parsed);
            }
        }
    }

    Err(anyhow!(
        "failed to parse engine detect output: {}",
        text.trim()
    ))
}

fn parse_batch_engine_result_line(line: &str) -> Result<RawBatchEngineResult> {
    serde_json::from_str::<RawBatchEngineResult>(line.trim()).with_context(|| {
        format!(
            "failed to parse engine batch detect output line: {}",
            line.trim()
        )
    })
}

fn fallback_engine_result(image_path: &Path) -> EngineResult {
    let (width, height) = image_dimensions(image_path);
    let quad = vec![
        [0.0, 0.0],
        [width as f64, 0.0],
        [width as f64, height as f64],
        [0.0, height as f64],
    ];
    let best = EngineBest {
        method: "full_frame_fallback".to_string(),
        quad: quad.clone(),
        confidence: 0.0,
        source: Some("fallback".to_string()),
        model_id: Some("full_frame_fallback".to_string()),
        debug_only: false,
    };
    let candidate = Candidate {
        method: "full_frame_fallback".to_string(),
        score: 0.0,
        quad: quad.clone(),
        metrics: serde_json::json!({
            "fallback": 1.0,
            "image_width": width,
            "image_height": height
        }),
        original_quad: Some(quad.clone()),
        manual_quad: None,
        source: Some("fallback".to_string()),
        model_id: Some("full_frame_fallback".to_string()),
        debug_only: false,
    };
    EngineResult {
        best,
        candidates: vec![candidate],
    }
}

fn raw_engine_result_to_engine_result(raw: RawEngineResult, image_path: &Path) -> EngineResult {
    if let Some(best) = raw.best {
        return EngineResult {
            best,
            candidates: raw.candidates,
        };
    }
    fallback_engine_result(image_path)
}

#[cfg_attr(not(test), allow(dead_code))]
fn run_engine_detect(app: &AppHandle, image_path: &Path) -> Result<EngineResult> {
    let engine = engine_dir(app)?;
    let mut command = Command::new(python_cmd(app)?);
    configure_engine_command(&mut command, app);
    let output = command
        .arg(engine.join("detect_frame.py"))
        .arg("detect")
        .arg("--image")
        .arg(image_path)
        .current_dir(&engine)
        .output()
        .context("failed to run detection engine")?;
    if !output.status.success() {
        return Err(anyhow!(
            "detection engine failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    let raw = parse_engine_detect_output(&output.stdout)?;
    Ok(raw_engine_result_to_engine_result(raw, image_path))
}

fn run_engine_detect_batch(
    app: &AppHandle,
    image_paths: &[PathBuf],
    scan_id: u64,
    total: usize,
) -> Result<HashMap<PathBuf, EngineResult>> {
    let engine = engine_dir(app)?;
    let manifest_dir = runtime_temp_dir();
    fs::create_dir_all(&manifest_dir).context("failed to create runtime temp directory")?;
    let manifest_path = manifest_dir.join(format!("detect-batch-{scan_id}.json"));
    let manifest = serde_json::json!({
        "images": image_paths.iter().map(|path| path.to_string_lossy().to_string()).collect::<Vec<_>>()
    });
    fs::write(&manifest_path, serde_json::to_vec(&manifest)?)
        .with_context(|| format!("failed to write {}", manifest_path.display()))?;

    let mut command = Command::new(python_cmd(app)?);
    configure_engine_command(&mut command, app);
    let mut child = command
        .arg(engine.join("detect_frame.py"))
        .arg("detect-batch")
        .arg("--manifest")
        .arg(&manifest_path)
        .current_dir(&engine)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("failed to run batch detection engine")?;

    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| anyhow!("failed to capture batch detection stdout"))?;
    let mut stderr_pipe = child
        .stderr
        .take()
        .ok_or_else(|| anyhow!("failed to capture batch detection stderr"))?;

    let mut reader = BufReader::new(stdout);
    let mut results = HashMap::new();
    let mut line = String::new();
    let mut processed = 0usize;

    loop {
        if scan_cancel_flag().load(Ordering::SeqCst) {
            let _ = child.kill();
            let _ = child.wait();
            let _ = fs::remove_file(&manifest_path);
            return Err(anyhow!("scan cancelled"));
        }

        line.clear();
        let bytes = reader
            .read_line(&mut line)
            .context("failed to read batch detection output")?;
        if bytes == 0 {
            break;
        }

        let raw = parse_batch_engine_result_line(&line)?;
        let image_path = PathBuf::from(&raw.image_path);
        let result = if raw.error.is_some() {
            fallback_engine_result(&image_path)
        } else {
            raw_engine_result_to_engine_result(
                RawEngineResult {
                    best: raw.best,
                    candidates: raw.candidates,
                },
                &image_path,
            )
        };
        processed += 1;
        let current_name = image_path
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or("page")
            .to_string();
        emit_scan_progress(
            app,
            ScanProgressEvent {
                scan_id,
                phase: "processing".to_string(),
                processed,
                total,
                current_name: Some(current_name),
                message: format!("正在识别第 {} / {} 张", processed, total),
            },
        );
        results.insert(image_path, result);
    }

    let status = child
        .wait()
        .context("failed to wait for batch detection engine")?;
    let mut stderr = String::new();
    let _ = stderr_pipe.read_to_string(&mut stderr);
    let _ = fs::remove_file(&manifest_path);
    if !status.success() {
        return Err(anyhow!("detection engine failed: {}", stderr.trim()));
    }

    Ok(results)
}

fn run_engine_preview(app: &AppHandle, image_path: &Path, quad: &[[f64; 2]]) -> Result<String> {
    let engine = engine_dir(app)?;
    let preview_dir = runtime_temp_dir();
    fs::create_dir_all(&preview_dir).context("failed to create preview directory")?;
    let preview_path = preview_dir.join(format!(
        "{}-preview.png",
        image_path
            .file_stem()
            .and_then(|value| value.to_str())
            .unwrap_or("page")
    ));
    let quad_json = serde_json::to_string(quad)?;
    let output = Command::new(python_cmd(app)?)
        .arg(engine.join("detect_frame.py"))
        .arg("preview")
        .arg("--image")
        .arg(image_path)
        .arg("--quad")
        .arg(quad_json)
        .arg("--output")
        .arg(&preview_path)
        .current_dir(&engine)
        .output()
        .context("failed to run preview engine")?;
    if !output.status.success() {
        return Err(anyhow!(
            "preview engine failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    Ok(preview_path.to_string_lossy().to_string())
}

fn run_engine_export(
    app: &AppHandle,
    project: &ProjectFile,
    options: &ExportOptions,
) -> Result<ExportResult> {
    let engine = engine_dir(app)?;
    let output_path = PathBuf::from(&options.output_path);
    let work_dir = export_work_dir(&output_path);
    fs::create_dir_all(&work_dir)
        .with_context(|| format!("failed to create {}", work_dir.display()))?;

    let pages = exportable_pages(&project.pages, options);
    if pages.is_empty() {
        return Err(anyhow!(
            "no pages are eligible for export; mark pages reviewed or enable trusted auto-ready export"
        ));
    }

    let request = ExportRequest {
        project_name: project.name.clone(),
        source_dir: project.source_dir.clone(),
        output_path: output_path.to_string_lossy().to_string(),
        work_dir: work_dir.to_string_lossy().to_string(),
        options: options.clone(),
        pages: pages
            .into_iter()
            .map(|page| ExportPage {
                id: page.id.clone(),
                name: page.name.clone(),
                path: page.path.clone(),
                status: page.status.clone(),
                confidence: page.confidence,
                quad: candidate_export_quad(page),
            })
            .collect(),
    };

    let manifest_path = work_dir.join("export-request.json");
    fs::write(&manifest_path, serde_json::to_vec_pretty(&request)?)
        .with_context(|| format!("failed to write {}", manifest_path.display()))?;

    let output = Command::new(python_cmd(app)?)
        .arg(engine.join("detect_frame.py"))
        .arg("export")
        .arg("--manifest")
        .arg(&manifest_path)
        .current_dir(&engine)
        .output()
        .context("failed to run export engine")?;
    if !output.status.success() {
        return Err(anyhow!(
            "export engine failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    let result = serde_json::from_slice::<ExportResult>(&output.stdout)
        .context("failed to parse engine export output")?;
    open_output_directory(&output_path)?;
    Ok(result)
}

fn format_time(value: SystemTime) -> String {
    let datetime: DateTime<Local> = value.into();
    datetime.format("%Y-%m-%d %H:%M:%S").to_string()
}

fn created_time(path: &Path) -> SystemTime {
    fs::metadata(path)
        .and_then(|metadata| metadata.created().or_else(|_| metadata.modified()))
        .unwrap_or(SystemTime::UNIX_EPOCH)
}

fn modified_time(path: &Path) -> SystemTime {
    fs::metadata(path)
        .and_then(|metadata| metadata.modified())
        .unwrap_or_else(|_| created_time(path))
}

fn image_dimensions(path: &Path) -> (u32, u32) {
    image::image_dimensions(path).unwrap_or((0, 0))
}

fn captured_time(path: &Path) -> Option<String> {
    let file = fs::File::open(path).ok()?;
    let mut reader = BufReader::new(file);
    let exif = Reader::new().read_from_container(&mut reader).ok()?;
    let field = exif
        .get_field(Tag::DateTimeOriginal, In::PRIMARY)
        .or_else(|| exif.get_field(Tag::DateTime, In::PRIMARY))?;
    Some(field.display_value().with_unit(&exif).to_string())
}

fn page_details(path: &Path) -> PageDetails {
    let metadata = fs::metadata(path).ok();
    let (width, height) = image_dimensions(path);
    PageDetails {
        width,
        height,
        file_size_bytes: metadata.as_ref().map(|item| item.len()).unwrap_or(0),
        captured_at: captured_time(path),
        created_at: format_time(created_time(path)),
        modified_at: format_time(modified_time(path)),
    }
}

fn build_project(app: &AppHandle, folder_path: &Path, scan_id: u64) -> Result<ProjectFile> {
    let mut image_paths = fs::read_dir(folder_path)
        .with_context(|| format!("failed to read folder {}", folder_path.display()))?
        .filter_map(|entry| entry.ok().map(|value| value.path()))
        .filter(|path| path.is_file() && supported_image(path))
        .collect::<Vec<_>>();

    image_paths.sort_by_key(|path| created_time(path));
    let total = image_paths.len();
    scan_cancel_flag().store(false, Ordering::SeqCst);
    clear_runtime_temp_artifacts()?;
    emit_scan_progress(
        app,
        ScanProgressEvent {
            scan_id,
            phase: "started".to_string(),
            processed: 0,
            total,
            current_name: None,
            message: if total == 0 {
                "文件夹中没有可处理图片".to_string()
            } else {
                format!("开始扫描，共 {} 张图片", total)
            },
        },
    );

    let mut pages = Vec::new();
    let detections = run_engine_detect_batch(app, &image_paths, scan_id, total)?;

    for (index, path) in image_paths.into_iter().enumerate() {
        if scan_cancel_flag().load(Ordering::SeqCst) {
            clear_runtime_temp_artifacts()?;
            emit_scan_progress(
                app,
                ScanProgressEvent {
                    scan_id,
                    phase: "cancelled".to_string(),
                    processed: index,
                    total,
                    current_name: None,
                    message: "已取消扫描".to_string(),
                },
            );
            return Err(anyhow!("scan cancelled"));
        }

        let current_name = path
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or("page")
            .to_string();
        let created = created_time(&path);
        let details = page_details(&path);
        let detection = detections
            .get(&path)
            .cloned()
            .unwrap_or_else(|| fallback_engine_result(&path));
        let confidence = detection.best.confidence;
        let status = if confidence < 0.05 {
            "needs_review"
        } else {
            "auto_ready"
        };
        pages.push(PageRecord {
            id: path
                .file_stem()
                .and_then(|value| value.to_str())
                .unwrap_or("page")
                .to_string(),
            name: current_name,
            path: path.to_string_lossy().to_string(),
            thumb_path: generate_thumbnail(&path),
            created_at: format_time(created),
            status: status.to_string(),
            confidence,
            best_method: Some(detection.best.method),
            selected_candidate_index: 0,
            active_quad: detection.best.quad,
            manual_quad: None,
            manual_base_candidate_index: None,
            preview_path: None,
            details,
            event_slug: None,
            difficulty_bucket: None,
            failure_tags: Vec::new(),
            bucket_reason: Vec::new(),
            review_tags: Vec::new(),
            tag_version: None,
            candidates: detection.candidates,
        });
    }

    let project_name = folder_path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("Screen PDF Project")
        .to_string();

    emit_scan_progress(
        app,
        ScanProgressEvent {
            scan_id,
            phase: "completed".to_string(),
            processed: total,
            total,
            current_name: None,
            message: format!("扫描完成，共 {} 张图片", total),
        },
    );

    Ok(ProjectFile {
        version: 1,
        data_structure_version: Some(CURRENT_DATA_STRUCTURE_VERSION),
        name: project_name,
        source_dir: folder_path.to_string_lossy().to_string(),
        project_path: None,
        selected_page_id: pages.first().map(|page| page.id.clone()),
        event_slug: None,
        event_name: None,
        tag_version: None,
        tag_summary: None,
        pages,
    })
}

fn infer_data_structure_version(project: &ProjectFile) -> u32 {
    if let Some(version) = project.data_structure_version {
        if version > 0 {
            return version;
        }
    }
    if project.pages.iter().any(|page| {
        page.manual_quad
            .as_ref()
            .is_some_and(|quad| !quad.is_empty())
    }) {
        return 1;
    }
    if project.pages.iter().any(|page| {
        page.candidates.iter().any(|candidate| {
            candidate
                .manual_quad
                .as_ref()
                .is_some_and(|quad| !quad.is_empty())
        })
    }) {
        return CURRENT_DATA_STRUCTURE_VERSION;
    }
    CURRENT_DATA_STRUCTURE_VERSION
}

fn preferred_project_file_in_dir(_folder_path: &Path) -> Option<PathBuf> {
    None
}

fn load_project_file(project_path: &Path) -> Result<ProjectFile> {
    let content = fs::read_to_string(project_path)
        .with_context(|| format!("failed to read {}", project_path.display()))?;
    let mut project = serde_json::from_str::<ProjectFile>(&content)
        .with_context(|| format!("invalid {}", project_path.display()))?;
    for page in &mut project.pages {
        let current_thumb = page.thumb_path.as_deref().map(Path::new);
        if current_thumb.is_none_or(|thumb| !thumb.exists()) {
            page.thumb_path = generate_thumbnail(Path::new(&page.path));
        }
    }
    project.data_structure_version = Some(infer_data_structure_version(&project));
    project.project_path = Some(project_path.to_string_lossy().to_string());
    Ok(project)
}

#[tauri::command]
async fn scan_folder(
    app: AppHandle,
    folder_path: String,
    scan_id: u64,
) -> Result<ProjectFile, String> {
    build_project(&app, Path::new(&folder_path), scan_id).map_err(|err| err.to_string())
}

#[tauri::command]
async fn cancel_scan() -> Result<(), String> {
    scan_cancel_flag().store(true, Ordering::SeqCst);
    clear_runtime_temp_artifacts().map_err(|err| err.to_string())
}

#[tauri::command]
async fn save_project(project_path: String, project: ProjectFile) -> Result<String, String> {
    let mut updated = project;
    updated.data_structure_version = Some(CURRENT_DATA_STRUCTURE_VERSION);
    updated.project_path = Some(project_path.clone());
    let content = serde_json::to_string_pretty(&updated).map_err(|err| err.to_string())?;
    fs::write(&project_path, content).map_err(|err| err.to_string())?;
    Ok(project_path)
}

#[tauri::command]
async fn load_project(project_path: String) -> Result<ProjectFile, String> {
    load_project_file(Path::new(&project_path)).map_err(|err| err.to_string())
}

#[tauri::command]
async fn generate_preview(
    app: AppHandle,
    image_path: String,
    quad: Vec<[f64; 2]>,
) -> Result<String, String> {
    run_engine_preview(&app, Path::new(&image_path), &quad).map_err(|err| err.to_string())
}

#[tauri::command]
async fn export_project(
    app: AppHandle,
    project: ProjectFile,
    options: ExportOptions,
) -> Result<ExportResult, String> {
    run_engine_export(&app, &project, &options).map_err(|err| err.to_string())
}

#[tauri::command]
async fn get_update_state<R: tauri::Runtime>(
    app: AppHandle<R>,
) -> Result<UpdateStatePayload, String> {
    Ok(current_update_snapshot(
        &app.package_info().version.to_string(),
    ))
}

#[tauri::command]
async fn check_for_app_update<R: tauri::Runtime>(
    app: AppHandle<R>,
) -> Result<UpdateStatePayload, String> {
    let current_version = app.package_info().version.to_string();
    {
        let mut manager = lock_update_manager()?;
        manager.snapshot = UpdateStatePayload {
            status: UpdateStatus::Checking,
            update: None,
            progress: None,
            error: None,
        };
    }

    let snapshot = match fetch_update(&app).await {
        Ok(Some(update)) => UpdateStatePayload {
            status: UpdateStatus::Available,
            update: Some(to_update_info_payload(&current_version, &update)),
            progress: None,
            error: None,
        },
        Ok(None) => UpdateStatePayload {
            status: UpdateStatus::UpToDate,
            update: None,
            progress: None,
            error: None,
        },
        Err(err) => UpdateStatePayload {
            status: UpdateStatus::Error,
            update: None,
            progress: None,
            error: Some(err),
        },
    };
    let mut manager = lock_update_manager()?;
    manager.snapshot = snapshot.clone();
    Ok(snapshot)
}

#[tauri::command]
async fn install_app_update<R: tauri::Runtime>(
    app: AppHandle<R>,
    version: String,
) -> Result<UpdateStatePayload, String> {
    let current_version = app.package_info().version.to_string();
    let update = fetch_update(&app)
        .await?
        .ok_or_else(|| "当前没有可安装的更新。".to_string())?;
    let payload = to_update_info_payload(&current_version, &update);
    if payload.version != version {
        return Err("更新版本不匹配，请重新检查更新后再试。".to_string());
    }
    {
        let mut manager = lock_update_manager()?;
        manager.snapshot = UpdateStatePayload {
            status: UpdateStatus::Downloading,
            update: Some(payload.clone()),
            progress: Some(UpdateProgressPayload::default()),
            error: None,
        };
    }

    let snapshot = match download_and_install_update(update, &current_version).await {
        Ok(()) => UpdateStatePayload {
            status: UpdateStatus::Ready,
            update: Some(payload),
            progress: Some(UpdateProgressPayload {
                percent: 100.0,
                total: 0,
                transferred: 0,
            }),
            error: None,
        },
        Err(err) => UpdateStatePayload {
            status: UpdateStatus::Error,
            update: Some(payload),
            progress: None,
            error: Some(err),
        },
    };
    let mut manager = lock_update_manager()?;
    manager.snapshot = snapshot.clone();
    Ok(snapshot)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_page(id: &str, status: &str, confidence: f64) -> PageRecord {
        PageRecord {
            id: id.to_string(),
            name: format!("{id}.jpg"),
            path: format!("/tmp/{id}.jpg"),
            thumb_path: Some(format!("/tmp/{id}-thumb.jpg")),
            created_at: "0".to_string(),
            status: status.to_string(),
            confidence,
            best_method: Some("contour_quad".to_string()),
            selected_candidate_index: 0,
            candidates: vec![Candidate {
                method: "contour_quad".to_string(),
                score: 0.95,
                quad: vec![[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],
                original_quad: Some(vec![[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]),
                manual_quad: None,
                metrics: serde_json::json!({}),
                source: Some("opencv".to_string()),
                model_id: Some("contour_quad".to_string()),
                debug_only: false,
            }],
            active_quad: vec![[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],
            manual_quad: None,
            manual_base_candidate_index: None,
            preview_path: None,
            details: PageDetails {
                width: 1000,
                height: 800,
                file_size_bytes: 1024,
                captured_at: None,
                created_at: "2026-03-21 10:00:00".to_string(),
                modified_at: "2026-03-21 10:00:00".to_string(),
            },
            event_slug: None,
            difficulty_bucket: None,
            failure_tags: Vec::new(),
            bucket_reason: Vec::new(),
            review_tags: Vec::new(),
            tag_version: None,
        }
    }

    #[test]
    fn project_page_record_round_trips_manual_base_candidate_index() {
        let page = sample_page("page-1", "reviewed", 0.9);
        let project = ProjectFile {
            version: 1,
            data_structure_version: Some(2),
            name: "demo".to_string(),
            source_dir: "/tmp/demo".to_string(),
            project_path: None,
            selected_page_id: Some("page-1".to_string()),
            event_slug: None,
            event_name: None,
            tag_version: None,
            tag_summary: None,
            pages: vec![PageRecord {
                manual_quad: Some(vec![[1.0, 2.0], [10.0, 2.0], [10.0, 12.0], [1.0, 12.0]]),
                manual_base_candidate_index: Some(0),
                ..page
            }],
        };

        let value = serde_json::to_value(&project).expect("serialize project");
        assert_eq!(value["pages"][0]["manualBaseCandidateIndex"], 0);

        let decoded: ProjectFile = serde_json::from_value(value).expect("deserialize project");
        assert_eq!(decoded.pages[0].manual_base_candidate_index, Some(0));
        assert_eq!(decoded.data_structure_version, Some(2));
    }

    #[test]
    fn project_round_trips_data_structure_version() {
        let project = ProjectFile {
            version: 1,
            data_structure_version: Some(2),
            name: "demo".to_string(),
            source_dir: "/tmp/demo".to_string(),
            project_path: None,
            selected_page_id: Some("page-1".to_string()),
            event_slug: None,
            event_name: None,
            tag_version: None,
            tag_summary: None,
            pages: vec![sample_page("page-1", "reviewed", 0.9)],
        };

        let value = serde_json::to_value(&project).expect("serialize project");
        assert_eq!(value["dataStructureVersion"], 2);

        let decoded: ProjectFile = serde_json::from_value(value).expect("deserialize project");
        assert_eq!(decoded.data_structure_version, Some(2));
    }

    #[test]
    fn infer_data_structure_version_returns_legacy_when_page_manual_quad_exists() {
        let mut project = ProjectFile {
            version: 1,
            data_structure_version: None,
            name: "demo".to_string(),
            source_dir: "/tmp/demo".to_string(),
            project_path: None,
            selected_page_id: Some("page-1".to_string()),
            event_slug: None,
            event_name: None,
            tag_version: None,
            tag_summary: None,
            pages: vec![sample_page("page-1", "reviewed", 0.9)],
        };
        project.pages[0].manual_quad = Some(vec![[1.0, 1.0], [9.0, 1.0], [9.0, 9.0], [1.0, 9.0]]);

        assert_eq!(infer_data_structure_version(&project), 1);
    }

    #[test]
    fn infer_data_structure_version_returns_current_when_candidate_manual_quad_exists() {
        let mut project = ProjectFile {
            version: 1,
            data_structure_version: None,
            name: "demo".to_string(),
            source_dir: "/tmp/demo".to_string(),
            project_path: None,
            selected_page_id: Some("page-1".to_string()),
            event_slug: None,
            event_name: None,
            tag_version: None,
            tag_summary: None,
            pages: vec![sample_page("page-1", "reviewed", 0.9)],
        };
        project.pages[0].candidates[0].manual_quad =
            Some(vec![[1.0, 1.0], [9.0, 1.0], [9.0, 9.0], [1.0, 9.0]]);

        assert_eq!(
            infer_data_structure_version(&project),
            CURRENT_DATA_STRUCTURE_VERSION
        );
    }

    #[test]
    fn infer_data_structure_version_prefers_explicit_version_when_present() {
        let mut project = ProjectFile {
            version: 1,
            data_structure_version: Some(CURRENT_DATA_STRUCTURE_VERSION),
            name: "demo".to_string(),
            source_dir: "/tmp/demo".to_string(),
            project_path: None,
            selected_page_id: Some("page-1".to_string()),
            event_slug: None,
            event_name: None,
            tag_version: None,
            tag_summary: None,
            pages: vec![sample_page("page-1", "reviewed", 0.9)],
        };
        project.pages[0].manual_quad = Some(vec![[1.0, 1.0], [9.0, 1.0], [9.0, 9.0], [1.0, 9.0]]);

        assert_eq!(
            infer_data_structure_version(&project),
            CURRENT_DATA_STRUCTURE_VERSION
        );
    }

    #[test]
    fn infer_data_structure_version_defaults_to_current_when_manual_fields_are_absent() {
        let project = ProjectFile {
            version: 1,
            data_structure_version: None,
            name: "demo".to_string(),
            source_dir: "/tmp/demo".to_string(),
            project_path: None,
            selected_page_id: Some("page-1".to_string()),
            event_slug: None,
            event_name: None,
            tag_version: None,
            tag_summary: None,
            pages: vec![sample_page("page-1", "reviewed", 0.9)],
        };

        assert_eq!(
            infer_data_structure_version(&project),
            CURRENT_DATA_STRUCTURE_VERSION
        );
    }

    #[test]
    fn infer_data_structure_version_prefers_legacy_when_missing_version_and_both_manual_shapes_exist(
    ) {
        let mut project = ProjectFile {
            version: 1,
            data_structure_version: None,
            name: "demo".to_string(),
            source_dir: "/tmp/demo".to_string(),
            project_path: None,
            selected_page_id: Some("page-1".to_string()),
            event_slug: None,
            event_name: None,
            tag_version: None,
            tag_summary: None,
            pages: vec![sample_page("page-1", "reviewed", 0.9)],
        };
        project.pages[0].manual_quad = Some(vec![[1.0, 1.0], [9.0, 1.0], [9.0, 9.0], [1.0, 9.0]]);
        project.pages[0].candidates[0].manual_quad =
            Some(vec![[2.0, 2.0], [8.0, 2.0], [8.0, 8.0], [2.0, 8.0]]);

        assert_eq!(infer_data_structure_version(&project), 1);
    }

    #[test]
    fn candidate_round_trips_manual_override() {
        let candidate = Candidate {
            method: "v28".to_string(),
            score: 0.9,
            quad: vec![[1.0, 1.0], [9.0, 1.0], [9.0, 9.0], [1.0, 9.0]],
            original_quad: Some(vec![[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]),
            manual_quad: Some(vec![[1.0, 1.0], [9.0, 1.0], [9.0, 9.0], [1.0, 9.0]]),
            metrics: serde_json::json!({}),
            source: Some("runtime".to_string()),
            model_id: Some("v28".to_string()),
            debug_only: false,
        };

        let value = serde_json::to_value(&candidate).expect("serialize candidate");
        assert_eq!(value["originalQuad"][0][0], 0.0);
        assert_eq!(value["manualQuad"][0][0], 1.0);

        let decoded: Candidate = serde_json::from_value(value).expect("deserialize candidate");
        assert_eq!(
            decoded
                .original_quad
                .as_ref()
                .and_then(|quad| quad.first())
                .map(|point| point[0]),
            Some(0.0)
        );
        assert_eq!(
            decoded
                .manual_quad
                .as_ref()
                .and_then(|quad| quad.first())
                .map(|point| point[0]),
            Some(1.0)
        );
    }

    #[test]
    fn supported_extensions_are_detected() {
        assert!(supported_image(Path::new("foo.jpeg")));
        assert!(supported_image(Path::new("foo.HEIC")));
        assert!(!supported_image(Path::new("foo.txt")));
    }

    #[test]
    fn time_format_is_stable() {
        let value = format_time(SystemTime::UNIX_EPOCH);
        assert_eq!(value, "1970-01-01 08:00:00");
    }

    #[test]
    fn exportable_pages_include_reviewed_and_trusted_auto_ready() {
        let pages = vec![
            sample_page("reviewed", "reviewed", 0.01),
            sample_page("needs-review", "needs_review", 0.01),
            sample_page("trusted", "auto_ready", 0.72),
            sample_page("untrusted", "auto_ready", 0.08),
            sample_page("error", "error", 0.9),
        ];

        let selected = exportable_pages(
            &pages,
            &ExportOptions {
                output_path: "/tmp/out.pdf".to_string(),
                export_all_pages: true,
                include_auto_ready: true,
                min_auto_ready_confidence: 0.2,
                jpeg_quality: 82,
                max_dimension: 2200,
                ocr_enabled: true,
                ocr_languages: "chi_sim+eng".to_string(),
            },
        );

        let ids = selected
            .iter()
            .map(|page| page.id.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            ids,
            vec!["reviewed", "needs-review", "trusted", "untrusted"]
        );
    }

    #[test]
    fn exportable_pages_can_disable_auto_ready_selection() {
        let pages = vec![
            sample_page("reviewed", "reviewed", 0.01),
            sample_page("trusted", "auto_ready", 0.91),
        ];

        let selected = exportable_pages(
            &pages,
            &ExportOptions {
                output_path: "/tmp/out.pdf".to_string(),
                export_all_pages: false,
                include_auto_ready: false,
                min_auto_ready_confidence: 0.2,
                jpeg_quality: 82,
                max_dimension: 2200,
                ocr_enabled: true,
                ocr_languages: "chi_sim+eng".to_string(),
            },
        );

        let ids = selected
            .iter()
            .map(|page| page.id.as_str())
            .collect::<Vec<_>>();
        assert_eq!(ids, vec!["reviewed"]);
    }

    #[test]
    fn output_directory_uses_export_parent_folder() {
        let path = Path::new("/tmp/screen-pdf/out.pdf");
        assert_eq!(output_directory(path), PathBuf::from("/tmp/screen-pdf"));
    }

    #[test]
    fn thumbnail_name_keeps_file_stem_and_suffix() {
        let path = Path::new("/tmp/demo/test.jpeg");
        let name = thumbnail_file_name(path, 1700000000);
        assert!(name.starts_with("test-1700000000"));
        assert!(name.ends_with(".jpg"));
    }

    #[test]
    fn ocr_languages_fall_back_to_available_values() {
        let resolved =
            resolve_ocr_languages("chi_sim+eng", &["eng".to_string(), "osd".to_string()]);
        assert_eq!(resolved, "eng");

        let resolved_with_match =
            resolve_ocr_languages("eng+osd", &["eng".to_string(), "osd".to_string()]);
        assert_eq!(resolved_with_match, "eng+osd");
    }

    #[test]
    fn bundled_python_candidates_cover_tauri_resource_layout() {
        let engine = Path::new("/tmp/Resources/_up_/engine");
        let candidates = bundled_python_candidates(engine);
        assert!(candidates
            .iter()
            .any(|value| value.ends_with("vendor/macos/bin/python3")));
    }

    #[test]
    fn resolve_resource_path_from_base_supports_nested_up_segments() {
        let base = std::env::temp_dir().join(format!(
            "screen-pdf-resource-layout-{}",
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let nested_engine = base.join("_up_").join("_up_").join("engine");
        fs::create_dir_all(&nested_engine).unwrap();
        let target = nested_engine.join("detect_frame.py");
        fs::write(&target, "print('ok')\n").unwrap();

        let resolved = resolve_resource_path_from_base(
            &base,
            Path::new("engine").join("detect_frame.py").as_path(),
        );

        assert_eq!(resolved.as_deref(), Some(target.as_path()));

        fs::remove_dir_all(base).unwrap();
    }

    #[test]
    fn resolve_resource_path_from_base_supports_deeper_nested_up_segments() {
        let base = std::env::temp_dir().join(format!(
            "screen-pdf-resource-layout-deep-{}",
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let nested_model = base
            .join("_up_")
            .join("_up_")
            .join("_up_")
            .join("models")
            .join("runtime");
        fs::create_dir_all(&nested_model).unwrap();
        let target = nested_model.join("deep_screen_v1_debug.pt");
        fs::write(&target, "stub").unwrap();

        let resolved = resolve_resource_path_from_base(
            &base,
            Path::new("models")
                .join("runtime")
                .join("deep_screen_v1_debug.pt")
                .as_path(),
        );

        assert_eq!(resolved.as_deref(), Some(target.as_path()));

        fs::remove_dir_all(base).unwrap();
    }

    #[test]
    fn python_probe_checks_required_modules() {
        let probe = python_probe_code();
        assert!(probe.contains("cv2"));
        assert!(probe.contains("PIL"));
        assert!(probe.contains("pypdf"));
        assert!(probe.contains("torch"));
    }

    #[test]
    fn parse_engine_detect_output_accepts_last_json_line() {
        let output = b"opencv warning\n{\"best\": {\"method\": \"contour_quad\", \"quad\": [[0,0],[1,0],[1,1],[0,1]], \"confidence\": 0.8}, \"candidates\": []}\n";
        let parsed = parse_engine_detect_output(output).unwrap();
        assert_eq!(parsed.best.unwrap().method, "contour_quad");
    }

    #[test]
    fn parse_batch_engine_result_accepts_single_json_line() {
        let line = "{\"imagePath\":\"/tmp/page-1.jpg\",\"best\":{\"method\":\"contour_quad\",\"quad\":[[0,0],[1,0],[1,1],[0,1]],\"confidence\":0.7},\"candidates\":[],\"error\":null}";
        let parsed = parse_batch_engine_result_line(line).unwrap();
        assert_eq!(parsed.image_path, "/tmp/page-1.jpg");
        assert_eq!(parsed.best.unwrap().method, "contour_quad");
    }

    #[test]
    fn fallback_engine_result_uses_full_frame_when_best_is_missing() {
        let result = fallback_engine_result(Path::new("/tmp/non-existent.jpg"));
        assert_eq!(result.best.method, "full_frame_fallback");
        assert_eq!(result.candidates.len(), 1);
    }

    #[test]
    fn export_request_serializes_manifest_keys_for_python_engine() {
        let request = ExportRequest {
            project_name: "demo".to_string(),
            source_dir: "/tmp/source".to_string(),
            output_path: "/tmp/out.pdf".to_string(),
            work_dir: "/tmp/work".to_string(),
            options: ExportOptions {
                output_path: "/tmp/out.pdf".to_string(),
                export_all_pages: true,
                include_auto_ready: true,
                min_auto_ready_confidence: 0.2,
                jpeg_quality: 82,
                max_dimension: 2200,
                ocr_enabled: true,
                ocr_languages: "chi_sim+eng".to_string(),
            },
            pages: vec![ExportPage {
                id: "page-1".to_string(),
                name: "page-1.jpeg".to_string(),
                path: "/tmp/page-1.jpeg".to_string(),
                status: "reviewed".to_string(),
                confidence: 0.9,
                quad: vec![[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]],
            }],
        };

        let value = serde_json::to_value(&request).unwrap();
        assert_eq!(
            value.get("output_path").and_then(|item| item.as_str()),
            Some("/tmp/out.pdf")
        );
        assert_eq!(
            value.get("work_dir").and_then(|item| item.as_str()),
            Some("/tmp/work")
        );
        assert!(value.get("outputPath").is_none());
        assert_eq!(
            value["options"]
                .get("ocr_languages")
                .and_then(|item| item.as_str()),
            Some("chi_sim+eng")
        );
        assert!(value["options"].get("ocrLanguages").is_none());
    }

    #[test]
    fn open_folder_does_not_auto_resolve_v1_project_file() {
        let base = std::env::temp_dir().join(format!(
            "screen-pdf-project-v1-{}",
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&base).unwrap();
        let legacy = base.join("screen-pdf-project.json");
        let tagged = base.join("screen-pdf-project_v1.json");
        fs::write(&legacy, "{}").unwrap();
        fs::write(&tagged, "{}").unwrap();

        let resolved = preferred_project_file_in_dir(&base);
        assert!(resolved.is_none());

        fs::remove_dir_all(base).unwrap();
    }

    #[test]
    fn open_folder_does_not_auto_resolve_legacy_project_file() {
        let base = std::env::temp_dir().join(format!(
            "screen-pdf-project-legacy-{}",
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&base).unwrap();
        let legacy = base.join("screen-pdf-project.json");
        fs::write(&legacy, "{}").unwrap();

        let resolved = preferred_project_file_in_dir(&base);
        assert!(resolved.is_none());

        fs::remove_dir_all(base).unwrap();
    }
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_updater::Builder::new().build())
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![
            scan_folder,
            cancel_scan,
            save_project,
            load_project,
            generate_preview,
            export_project,
            get_update_state,
            check_for_app_update,
            install_app_update
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
