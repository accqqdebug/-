# PIT-SNNTrack compact code review bundle

Fewer than 100 files including this README and the manifest.

Core lib/ sources retain model construction, training, data loading, loss, optimizer, trainer and tracker implementations. partition/ retains event preprocessing and evaluation. CONFIGURATIONS.json consolidates YAML text by original filename. RESEARCH_CODE_AND_RUNNERS.md consolidates candidate changes and experiment queue scripts. RESEARCH_AND_RESULTS.md includes the Chinese argument and aggregate results.

This is a compact code-review/upload bundle, not a standalone runnable installation. Secondary analysis/demo modules are omitted, and consolidated YAML/script text must be restored to its indicated paths if needed for execution. Use the earlier complete snapshot and original server project for reproduction.

No datasets, weights, environment, logs or duplicate repository tree are included. Original server code and full archive remain unchanged.
