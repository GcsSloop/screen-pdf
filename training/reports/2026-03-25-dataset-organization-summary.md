# Dataset Organization Summary

## Scope

- Source root: `/Users/gcssloop/WorkSpace/AIGC/展会/2026/202603 中国智慧道路照明大会`
- Dataset slug: `conference_202603_china_smart_road_lighting`
- Raw alias: `/Users/gcssloop/WorkSpace/AIGC/screen-pdf/data/raw/conference_202603_china_smart_road_lighting`

## Imported Counts

- Projects discovered: `22`
- Pages discovered: `720`
- Reviewed pages: `702`
- Pages with manual quad: `591`

## Split Policy

- Strategy: `deterministic_project_hash_v1`
- Train projects: `15`
- Val projects: `3`
- Holdout projects: `4`

Current split:

- Train: `project_001, project_004, project_006, project_008, project_010, project_011, project_012, project_013, project_015, project_016, project_017, project_018, project_019, project_020, project_021`
- Val: `project_003, project_007, project_022`
- Holdout: `project_002, project_005, project_009, project_014`

## High-Value Projects By Manual Annotation Density

- `project_002` `Al+道路照明一体化解决方案`: `84` pages, `84` manual quads
- `project_001` `AI助力道路照明高质量发展`: `92` pages, `61` manual quads
- `project_020` `智慧城市道路与景观照明融合的守正创新实践`: `59` pages, `58` manual quads
- `project_011` `洲明科技户外照明智慧产品解决方案的创新`: `40` pages, `39` manual quads
- `project_009` `如何发挥照明监控系统的实用价值`: `39` pages, `34` manual quads

## Generated Files

- Import manifest:
  - `/Users/gcssloop/WorkSpace/AIGC/screen-pdf/data/staging/imports/conference_202603_china_smart_road_lighting.json`
- Project manifests:
  - `/Users/gcssloop/WorkSpace/AIGC/screen-pdf/data/curated/projects/project_001.json` through `project_022.json`
- Page annotations:
  - `/Users/gcssloop/WorkSpace/AIGC/screen-pdf/data/curated/annotations/conference_202603_china_smart_road_lighting_pages.jsonl`
- Scene taxonomy and map:
  - `/Users/gcssloop/WorkSpace/AIGC/screen-pdf/data/curated/scenes/conference_202603_china_smart_road_lighting_scene_taxonomy.json`
  - `/Users/gcssloop/WorkSpace/AIGC/screen-pdf/data/curated/scenes/conference_202603_china_smart_road_lighting_project_scene_map.json`
- Splits:
  - `/Users/gcssloop/WorkSpace/AIGC/screen-pdf/data/splits/cross_project/conference_202603_china_smart_road_lighting_split_v1.json`
  - `/Users/gcssloop/WorkSpace/AIGC/screen-pdf/data/splits/holdout/conference_202603_china_smart_road_lighting_holdout_v1.json`
- Training registry:
  - `/Users/gcssloop/WorkSpace/AIGC/screen-pdf/training/registry/conference_202603_china_smart_road_lighting.json`

## Notes

- Raw data was not moved or renamed. The repository only mounts it through a stable English alias.
- Scene tags are an initial heuristic seed and are marked for manual review.
- This summary is the baseline for future dataset refresh and split evolution.
