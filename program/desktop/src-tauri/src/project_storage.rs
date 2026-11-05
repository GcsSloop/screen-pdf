//! Portable project storage. Absolute paths exist only in the in-memory UI model.
use super::{ProjectFile, CURRENT_DATA_STRUCTURE_VERSION};
use anyhow::{anyhow, Context, Result};
use std::{
    fs,
    path::{Path, PathBuf},
    time::SystemTime,
};

pub const FILE_NAME: &str = "screen-pdf-project.json";
pub const FILE_VERSION: u32 = 2;

fn relative_path(value: &str) -> Result<PathBuf> {
    let mut result = PathBuf::new();
    for part in value.replace('\\', "/").split('/') {
        match part {
            "" | "." => (),
            ".." => {
                if !result.pop() {
                    return Err(anyhow!("project path escapes its root"));
                }
            }
            other => result.push(other),
        }
    }
    Ok(result)
}

fn absolute_text(value: &str) -> bool {
    value.starts_with('/') || value.starts_with('\\') || value.as_bytes().get(1) == Some(&b':')
}

pub fn resolve_image(root: &Path, raw: &str, old_source: &str) -> Result<PathBuf> {
    let raw_slashes = raw.replace('\\', "/");
    let source = old_source.replace('\\', "/");
    let mut choices = Vec::new();
    if !absolute_text(raw) {
        choices.push(root.join(relative_path(&raw_slashes)?));
    } else {
        let root_string = root.to_string_lossy().replace('\\', "/");
        for prefix in [root_string.as_str(), source.trim_end_matches('/')] {
            if !prefix.is_empty() {
                if let Some(suffix) = raw_slashes.strip_prefix(&format!("{prefix}/")) {
                    choices.push(root.join(relative_path(suffix)?));
                }
            }
        }
        if let Some(name) = raw_slashes.rsplit('/').next() {
            choices.push(root.join(name));
        }
        // Legacy external files remain readable; saving requires images in the project.
        choices.push(PathBuf::from(raw));
    }
    choices
        .into_iter()
        .find(|p| p.is_file())
        .ok_or_else(|| anyhow!("image missing from project: {raw}"))
}

pub fn file_to_open(requested: &Path) -> Result<PathBuf> {
    let requested = requested.canonicalize()?;
    let parent = requested
        .parent()
        .ok_or_else(|| anyhow!("project has no parent directory"))?;
    // Prefer the root project over stale copies in old export subdirectories.
    let mut selected = requested.clone();
    for ancestor in parent.ancestors().skip(1) {
        let canonical = ancestor.join(FILE_NAME);
        if canonical.is_file() {
            selected = canonical;
        }
    }
    Ok(selected)
}

pub fn hydrate(mut project: ProjectFile, root: &Path) -> Result<ProjectFile> {
    let root = root
        .canonicalize()
        .context("project directory is missing; reopen it after moving or renaming")?;
    let old_source = project.source_dir.clone();
    for page in &mut project.pages {
        page.path = resolve_image(&root, &page.path, &old_source)?
            .to_string_lossy()
            .into_owned();
        // Temporary caches are disposable and must never bind a project to a machine.
        page.thumb_path = None;
        page.preview_path = None;
    }
    project.name = root
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("ScreenPDF")
        .to_owned();
    project.source_dir = root.to_string_lossy().into_owned();
    project.project_path = Some(root.join(FILE_NAME).to_string_lossy().into_owned());
    Ok(project)
}

pub fn root_for_file(path: &Path, project: &ProjectFile) -> PathBuf {
    let root = path.parent().unwrap();
    let local = |candidate: &Path| {
        project.pages.iter().all(|page| {
            resolve_image(candidate, &page.path, &project.source_dir)
                .ok()
                .and_then(|image| image.canonicalize().ok())
                .is_some_and(|image| image.starts_with(candidate))
        })
    };
    if project.version < FILE_VERSION
        && !project.pages.is_empty()
        && root
            .file_name()
            .is_some_and(|name| name.to_string_lossy().ends_with("_export"))
        && !local(root)
    {
        if let Some(parent) = root.parent() {
            if local(parent) {
                return parent.to_path_buf();
            }
        }
    }
    root.to_path_buf()
}

pub fn root_for_runtime(project: &ProjectFile) -> Result<PathBuf> {
    let source = PathBuf::from(&project.source_dir);
    if source.is_absolute() && source.is_dir() {
        return Ok(source.canonicalize()?);
    }
    if let Some(path) = &project.project_path {
        let file = Path::new(path);
        if file.is_absolute() && file.is_file() {
            return Ok(file.parent().unwrap().canonicalize()?);
        }
    }
    Err(anyhow!(
        "project directory is missing; reopen the project in its current directory"
    ))
}

pub fn save(project: &ProjectFile) -> Result<PathBuf> {
    let root = root_for_runtime(project)?;
    let mut stored = hydrate(project.clone(), &root)?;
    for page in &mut stored.pages {
        let canonical = Path::new(&page.path).canonicalize()?;
        let relative = canonical.strip_prefix(&root).map_err(|_| {
            anyhow!(
                "image must be inside the project before saving: {}",
                page.name
            )
        })?;
        page.path = relative.to_string_lossy().replace('\\', "/");
    }
    stored.version = FILE_VERSION;
    stored.data_structure_version = Some(CURRENT_DATA_STRUCTURE_VERSION);
    stored.source_dir = ".".to_owned();
    stored.project_path = Some(FILE_NAME.to_owned());
    let mut value = serde_json::to_value(stored)?;
    value.as_object_mut().unwrap().remove("name"); // Current directory is the sole name source.
    let path = root.join(FILE_NAME);
    let stamp = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)?
        .as_nanos();
    let temporary = root.join(format!(".screen-pdf-project-{stamp}.tmp"));
    fs::write(&temporary, serde_json::to_vec_pretty(&value)?)?;
    if let Err(error) = fs::rename(&temporary, &path) {
        let _ = fs::remove_file(&temporary);
        return Err(error.into());
    }
    Ok(path)
}

pub fn export_path(project: &ProjectFile, requested: &Path) -> Result<PathBuf> {
    let root = root_for_runtime(project)?;
    let name = root
        .file_name()
        .ok_or_else(|| anyhow!("project directory has no name"))?;
    let destination = if requested.as_os_str().is_empty() {
        root.as_path()
    } else {
        requested
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(root.as_path())
    };
    Ok(destination.join(format!("{}.pdf", name.to_string_lossy())))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::load_project_file;

    fn workspace() -> PathBuf {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "screen-pdf-portable-{}-{}",
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::SeqCst)
        ));
        fs::create_dir_all(&path).unwrap();
        path.canonicalize().unwrap()
    }
    fn fixture(root: &Path) -> ProjectFile {
        fs::create_dir_all(root.join("photos")).unwrap();
        fs::write(root.join("photos/page.jpeg"), b"image").unwrap();
        serde_json::from_value(serde_json::json!({
            "version":1,"dataStructureVersion":2,"name":"outdated",
            "sourceDir":root.to_string_lossy(),"projectPath":root.join(FILE_NAME).to_string_lossy(),
            "selectedPageId":"p", "pages":[{
                "id":"p","name":"page.jpeg","path":root.join("photos/page.jpeg").to_string_lossy(),
                "thumbPath":"/tmp/thumb.jpg","previewPath":"/tmp/preview.jpg","createdAt":"0","status":"reviewed",
                "confidence":0.0,"bestMethod":"screen_corner_single","selectedCandidateIndex":0,
                "activeQuad":[[1,1],[10,1],[10,10],[1,10]],"manualQuad":null,
                "candidates":[{"method":"screen_corner_single","score":1.0,"quad":[[1,1],[9,1],[9,9],[1,9]],
                "manualQuad":[[1,1],[10,1],[10,10],[1,10]],"metrics":{},"source":"runtime_single"}]
            }]
        })).unwrap()
    }
    #[test]
    fn saves_only_portable_root_paths_and_preserves_human_quads() {
        let base = workspace();
        let root = base.join("项目 2.0");
        let original = fixture(&root);
        let path = save(&original).unwrap();
        assert_eq!(path, root.join(FILE_NAME));
        let value: serde_json::Value = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
        assert_eq!(value["version"], 2);
        assert_eq!(value["sourceDir"], ".");
        assert_eq!(value["projectPath"], FILE_NAME);
        assert!(value.get("name").is_none());
        assert_eq!(value["pages"][0]["path"], "photos/page.jpeg");
        assert!(value["pages"][0]["thumbPath"].is_null());
        assert!(value["pages"][0]["previewPath"].is_null());
        assert_eq!(
            value["pages"][0]["candidates"][0]["manualQuad"],
            serde_json::to_value(original.pages[0].candidates[0].manual_quad.clone()).unwrap()
        );
        fs::remove_dir_all(base).unwrap();
    }
    #[test]
    fn opening_legacy_is_read_only_and_resaving_upgrades_without_losing_edits() {
        for legacy_data_version in [1, 2] {
            let base = workspace();
            let root = base.join("兼容项目 2.0");
            let mut original = fixture(&root);
            original.data_structure_version = Some(legacy_data_version);
            original.pages[0].review_tags = vec!["人工确认".to_owned()];
            if legacy_data_version == 1 {
                original.pages[0].manual_quad = original.pages[0].candidates[0].manual_quad.take();
                original.pages[0].manual_base_candidate_index = Some(0);
            }
            let path = root.join(FILE_NAME);
            let legacy_bytes = serde_json::to_vec_pretty(&original).unwrap();
            fs::write(&path, &legacy_bytes).unwrap();

            let loaded = load_project_file(&path).unwrap();
            assert_eq!(fs::read(&path).unwrap(), legacy_bytes);
            assert_eq!(loaded.version, 1);
            assert_eq!(save(&loaded).unwrap(), path);
            let saved: serde_json::Value =
                serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
            assert_eq!(saved["version"], FILE_VERSION);
            assert_eq!(
                saved["dataStructureVersion"],
                CURRENT_DATA_STRUCTURE_VERSION
            );
            assert_eq!(saved["sourceDir"], ".");
            assert_eq!(saved["projectPath"], FILE_NAME);
            assert_eq!(saved["pages"][0]["path"], "photos/page.jpeg");
            assert!(saved.get("name").is_none());

            let reopened = load_project_file(&path).unwrap();
            assert_eq!(reopened.selected_page_id, original.selected_page_id);
            assert_eq!(reopened.pages[0].status, original.pages[0].status);
            assert_eq!(reopened.pages[0].review_tags, original.pages[0].review_tags);
            assert_eq!(reopened.pages[0].active_quad, original.pages[0].active_quad);
            assert_eq!(reopened.pages[0].manual_quad, original.pages[0].manual_quad);
            assert_eq!(
                reopened.pages[0].manual_base_candidate_index,
                original.pages[0].manual_base_candidate_index
            );
            assert_eq!(
                reopened.pages[0].candidates[0].manual_quad,
                original.pages[0].candidates[0].manual_quad
            );
            assert_eq!(
                reopened.pages[0].candidates[0].quad,
                original.pages[0].candidates[0].quad
            );
            fs::remove_dir_all(base).unwrap();
        }
    }
    #[test]
    fn move_and_rename_round_trip_uses_new_directory_for_export() {
        let base = workspace();
        let old = base.join("旧名");
        let original = fixture(&old);
        save(&original).unwrap();
        let new = base.join("元宇宙 2.0 新名");
        fs::rename(&old, &new).unwrap();
        let loaded = load_project_file(&new.join(FILE_NAME)).unwrap();
        assert_eq!(loaded.name, "元宇宙 2.0 新名");
        assert_eq!(
            Path::new(&loaded.pages[0].path),
            new.join("photos/page.jpeg")
        );
        assert_eq!(
            export_path(&loaded, &base.join("旧名.pdf")).unwrap(),
            base.join("元宇宙 2.0 新名.pdf")
        );
        assert_eq!(
            loaded.pages[0].candidates[0].manual_quad,
            original.pages[0].candidates[0].manual_quad
        );
        assert_eq!(save(&loaded).unwrap(), new.join(FILE_NAME));
        fs::remove_dir_all(base).unwrap();
    }
    #[test]
    fn legacy_copy_prefers_local_subpaths_even_if_original_still_exists() {
        let base = workspace();
        let old = base.join("old");
        let original = fixture(&old);
        let legacy = serde_json::to_vec_pretty(&original).unwrap();
        fs::write(old.join(FILE_NAME), &legacy).unwrap();
        let copy = base.join("renamed");
        fixture(&copy);
        fs::write(copy.join(FILE_NAME), legacy).unwrap();
        let loaded = load_project_file(&copy.join(FILE_NAME)).unwrap();
        assert_eq!(
            Path::new(&loaded.pages[0].path),
            copy.join("photos/page.jpeg")
        );
        assert_eq!(loaded.name, "renamed");
        fs::remove_dir_all(base).unwrap();
    }
    #[test]
    fn windows_legacy_paths_rebase_to_current_root() {
        let base = workspace();
        let project = fixture(&base);
        let image = resolve_image(&base, "C:\\old\\photos\\page.jpeg", "C:\\old").unwrap();
        assert_eq!(image, PathBuf::from(project.pages[0].path.clone()));
        fs::remove_dir_all(base).unwrap();
    }
    #[test]
    fn nested_export_copy_opens_root_project_instead() {
        let base = workspace();
        let root = base.join("project");
        let p = fixture(&root);
        save(&p).unwrap();
        let nested = root.join("project_export");
        fs::create_dir_all(&nested).unwrap();
        fs::write(nested.join(FILE_NAME), b"stale").unwrap();
        assert_eq!(
            file_to_open(&nested.join(FILE_NAME)).unwrap(),
            root.join(FILE_NAME)
        );
        fs::remove_dir_all(base).unwrap();
    }
    #[test]
    fn refuses_escape_and_does_not_create_nonportable_project() {
        let base = workspace();
        let root = base.join("project");
        let mut p = fixture(&root);
        assert!(resolve_image(&root, "../../escape.jpg", "").is_err());
        let external = base.join("outside.jpeg");
        fs::write(&external, b"image").unwrap();
        p.pages[0].path = external.to_string_lossy().into_owned();
        assert!(save(&p).is_err());
        assert!(!root.join(FILE_NAME).exists());
        fs::remove_dir_all(base).unwrap();
    }
    #[test]
    fn legacy_export_only_copy_migrates_back_to_image_root() {
        let base = workspace();
        let root = base.join("new name");
        let mut original = fixture(&root);
        original.source_dir = "/old/machine/project".to_string();
        original.pages[0].path = "/old/machine/project/photos/page.jpeg".to_string();
        let nested = root.join("old-name_export");
        fs::create_dir_all(&nested).unwrap();
        fs::write(
            nested.join(FILE_NAME),
            serde_json::to_vec(&original).unwrap(),
        )
        .unwrap();
        let loaded = load_project_file(&nested.join(FILE_NAME)).unwrap();
        assert_eq!(loaded.name, "new name");
        assert_eq!(save(&loaded).unwrap(), root.join(FILE_NAME));
        fs::remove_dir_all(base).unwrap();
    }
}
