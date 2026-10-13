# IMOLA / TRASA organization

The active code repository is `research/IMOLA/TRASA`, on the GitHub repository's `main` branch. The previous `research/TRASA` path is a compatibility symlink to that same folder. New model changes belong in this repository.

IMOLA's existing files are organized in its parent workspace:

```text
IMOLA/
  TRASA/                 active paper implementation, this repository
  data/                  maps, inventories, patches, vectors, prior outputs
  workflows/             existing preprocessing/inventory/request scripts
  baselines/             original LDTR_KC and SAM-Road working copies
  documents/             project description and reorganization audit
  environments/          original IMOLA environment
  legacy/
    trasa-workspace-before-cleanup/  original TRASA checkout and local edits
  archive/               earlier exploratory work and reports
```

Previous IMOLA top-level paths remain symlinks so hardcoded paths and existing virtual-environment entry points continue to resolve. Executable `.py`/`.sh` files and the CA-area `scripts/` directory were moved from mixed data folders into `workflows`; compatibility links also remain at their old locations. This is a filesystem organization change, not a claim that all old workflows or environments are operational or converted to TRASA. No data-generation or historical-map training pipeline was run during cleanup.

The original TRASA checkout was moved intact into `legacy`, including its `.git`, existing modified files and pre-existing deleted-file status. Its 264 non-Git file checksums and complete Git status were verified after relocation. Of these files, 257 matched the pre-cleanup IMOLA copy, four differed and three existed only in the TRASA copy. Divergent copies are preserved for future review, not automatically merged.

All directory/file moves used inode-preserving renames on the same filesystem. Existing datasets, model checkpoints and baseline environments were not copied, erased or regenerated. Four relocated scripts that used their own `__file__` location to derive data destinations were adjusted for the new layout, with their original contents preserved in the legacy snapshot. Older regional workflows that assume a working-directory-relative location should still be run from their corresponding data workspace. The local parent workspace's `documents/workspace-reorganization.json` records the actual old/new paths, checksums and path fixes; it stays outside the public paper repository.

The clean model code and these documents replace the mixed code/data-processing layout on GitHub `main` through an ordinary commit retaining the previous main history. The earlier implementation-only branch is unnecessary after that update. The original draft manuscript remains local; only its source archive hash and implementation mapping are published.
