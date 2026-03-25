#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use anyhow::{anyhow, Context, Result};
use chrono::{DateTime, Local};
use exif::{In, Reader, Tag};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    fs,
    io::{BufRead, BufReader, Read},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::atomic::{AtomicBool, Ordering},
    sync::OnceLock,
    time::SystemTime,
};
use tauri::{AppHandle, Emitter, Manager};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Candidate {
    method: String,
    score: f64,
    quad: Vec<[f64; 2]>,
    metrics: serde_json::Value,
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
    #[serde(rename = "previewPath")]
    preview_path: Option<String>,
    #[serde(default)]
    details: PageDetails,
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
    name: String,
    #[serde(rename = "sourceDir")]
    source_dir: String,
    #[serde(rename = "projectPath")]
    project_path: Option<String>,
    #[serde(rename = "selectedPageId")]
    selected_page_id: Option<String>,
    pages: Vec<PageRecord>,
}

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
    let direct = resource_dir.join("engine");
    if direct.exists() {
        return Ok(direct);
    }
    let tauri_bundle = resource_dir.join("_up_").join("engine");
    if tauri_bundle.exists() {
        return Ok(tauri_bundle);
    }
    Err(anyhow!(
        "failed to resolve engine directory inside resources: checked {} and {}",
        direct.display(),
        tauri_bundle.display()
    ))
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
        python_candidates(app)
            .ok()
            .and_then(|candidates| {
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
    let stem = path.file_stem().and_then(|value| value.to_str()).unwrap_or("page");
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
        return Err(anyhow!("failed to open output directory {}", directory.display()));
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

    Err(anyhow!("failed to parse engine detect output: {}", text.trim()))
}

fn parse_batch_engine_result_line(line: &str) -> Result<RawBatchEngineResult> {
    serde_json::from_str::<RawBatchEngineResult>(line.trim())
        .with_context(|| format!("failed to parse engine batch detect output line: {}", line.trim()))
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
    };
    let candidate = Candidate {
        method: "full_frame_fallback".to_string(),
        score: 0.0,
        quad,
        metrics: serde_json::json!({
            "fallback": 1.0,
            "image_width": width,
            "image_height": height
        }),
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
    let output = Command::new(python_cmd(app)?)
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

    let mut child = Command::new(python_cmd(app)?)
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

    let status = child.wait().context("failed to wait for batch detection engine")?;
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

fn run_engine_export(app: &AppHandle, project: &ProjectFile, options: &ExportOptions) -> Result<ExportResult> {
    let engine = engine_dir(app)?;
    let output_path = PathBuf::from(&options.output_path);
    let work_dir = export_work_dir(&output_path);
    fs::create_dir_all(&work_dir).with_context(|| format!("failed to create {}", work_dir.display()))?;

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
            preview_path: None,
            details,
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
        name: project_name,
        source_dir: folder_path.to_string_lossy().to_string(),
        project_path: None,
        selected_page_id: pages.first().map(|page| page.id.clone()),
        pages,
    })
}

#[tauri::command]
async fn scan_folder(app: AppHandle, folder_path: String, scan_id: u64) -> Result<ProjectFile, String> {
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
    updated.project_path = Some(project_path.clone());
    let content = serde_json::to_string_pretty(&updated).map_err(|err| err.to_string())?;
    fs::write(&project_path, content).map_err(|err| err.to_string())?;
    Ok(project_path)
}

#[tauri::command]
async fn load_project(project_path: String) -> Result<ProjectFile, String> {
    let content = fs::read_to_string(&project_path).map_err(|err| err.to_string())?;
    let mut project = serde_json::from_str::<ProjectFile>(&content).map_err(|err| err.to_string())?;
    for page in &mut project.pages {
        let current_thumb = page.thumb_path.as_deref().map(Path::new);
        if current_thumb.is_none_or(|thumb| !thumb.exists()) {
            page.thumb_path = generate_thumbnail(Path::new(&page.path));
        }
    }
    project.project_path = Some(project_path);
    Ok(project)
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
                metrics: serde_json::json!({}),
            }],
            active_quad: vec![[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],
            manual_quad: None,
            preview_path: None,
            details: PageDetails {
                width: 1000,
                height: 800,
                file_size_bytes: 1024,
                captured_at: None,
                created_at: "2026-03-21 10:00:00".to_string(),
                modified_at: "2026-03-21 10:00:00".to_string(),
            },
        }
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

        let ids = selected.iter().map(|page| page.id.as_str()).collect::<Vec<_>>();
        assert_eq!(ids, vec!["reviewed", "needs-review", "trusted", "untrusted"]);
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

        let ids = selected.iter().map(|page| page.id.as_str()).collect::<Vec<_>>();
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
        let resolved = resolve_ocr_languages("chi_sim+eng", &["eng".to_string(), "osd".to_string()]);
        assert_eq!(resolved, "eng");

        let resolved_with_match =
            resolve_ocr_languages("eng+osd", &["eng".to_string(), "osd".to_string()]);
        assert_eq!(resolved_with_match, "eng+osd");
    }

    #[test]
    fn bundled_python_candidates_cover_tauri_resource_layout() {
        let engine = Path::new("/tmp/Resources/_up_/engine");
        let candidates = bundled_python_candidates(engine);
        assert!(candidates.iter().any(|value| value.ends_with("vendor/macos/bin/python3")));
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
        assert_eq!(value.get("output_path").and_then(|item| item.as_str()), Some("/tmp/out.pdf"));
        assert_eq!(value.get("work_dir").and_then(|item| item.as_str()), Some("/tmp/work"));
        assert!(value.get("outputPath").is_none());
        assert_eq!(
            value["options"].get("ocr_languages").and_then(|item| item.as_str()),
            Some("chi_sim+eng")
        );
        assert!(value["options"].get("ocrLanguages").is_none());
    }
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![
            scan_folder,
            cancel_scan,
            save_project,
            load_project,
            generate_preview,
            export_project
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
