# Run Flash-Next (équipier) -- notebook à pousser sur Kaggle

Objectif : un 2e run indépendant (25 jeux publics, Phase A) du clone de la solution Milestone-2 de dfranzen
(Qwen3.8 Flash-Next W4A16, RTX Pro 6000) pour mesurer la variance run-à-run. Le 1er run (Loumi, v1) : moyenne 43.96, score réel public 22.03.

**Notebook à pousser : `arc-agi-3-flashnext-dfranzen-clone.ipynb`** (ne pas le modifier -- même code que le run v1).

1. Dans `kernel-metadata.json`, remplacer `loumitrmas` par ton username Kaggle dans le champ `"id"`
   (ex. `"id": "TON_USERNAME/arc-agi-3-flashnext-dfranzen-clone"`). Ne rien changer d'autre
   (datasets dfranzen/*, model_sources, `machine_shape: NvidiaRtxPro6000`, internet désactivé).
2. Il faut avoir accepté les règles de la compétition `arc-prize-2026-arc-agi-3` et pouvoir utiliser le GPU RTX Pro 6000.
3. Pousser :
   `kaggle kernels push -p notebooks/flashnext_clone_kernel`
4. Durée attendue ~2 h. Quand le statut est COMPLETE, télécharger la sortie :
   `kaggle kernels output TON_USERNAME/arc-agi-3-flashnext-dfranzen-clone -p results/flashnext_clone_phaseA_<ton_nom>`
5. Analyse blocages + variance (tous les runs ensemble) :
   `python scripts/blocker_variance_report.py results/flashnext_clone_phaseA_v1 results/flashnext_clone_phaseA_<ton_nom> --out results/blocker_variance_flashnext`
   (le dossier `results/` n'est pas versionné : se transmettre les sorties à part.)

Ne pas soumettre ce kernel à la compétition sans accord : ce run sert uniquement à l'étude.
