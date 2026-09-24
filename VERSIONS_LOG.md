# Historique des versions — ARC-AGI-3

Trace chronologique de toutes les architectures/configs testées depuis le début du projet, avec les scores réels de soumission Kaggle quand ils existent. Compilé le 2026-09-22 depuis `DESIGN_LOG.md`, `git log` et la mémoire de session. Score réel = classement Kaggle (0-1). Score local = moyenne sur les 25 jeux publics, formule quadratique du concours (`local * ~0.35 ≈ real`, voir `kaggle_scoring_methodology`).

---

## Phase 1 — RL from-scratch (Eyes/Intuition/Brain, JEPA, GRPO)
`2026-08-14 → 2026-08-31`

→ **08-14** Proposition initiale : Eyes / frozen-brain / intuition
→ **08-16/17** Ablations GRPO épuisées sur `ls20` → pivot vers pré-entraînement JEPA
→ **08-18** Stage 2 (`jepa_pretrain.ipynb`) : fix d'archi confirmé, mais le problème de rétention est structurel à GRPO
→ **08-20** Implémenté : isolation de l'encodeur, archive Go-Explore, filtrage de groupe GRPO
→ **08-20** Ajout PPO-clip + pénalité KL (trust region)
→ **08-20** Bug NaN KL trouvé et fixé (premier run réel)
→ **08-20** Bug de masquage de curiosité (variance nulle) fixé
→ **08-20** DINOv2/v3 testé empiriquement vs from-scratch (`dino_probe.ipynb`)
→ **08-21** Journée de fixes en cascade : scale-up Stage 1, hypothèse RoPE, OOM sur collecte vectorisée, blocage "zero gradient updates", crash NaN (clamp fix), dead gradient + noisy-TV, redémarrage complet de Stage 2
→ **08-29** Fix trust-region PPO-clip confirmé comme la vraie cause du collapse, mais plafond d'exploration dur atteint à l'échelle
→ **08-31** 🎯 **PREMIÈRE SOUMISSION RÉELLE** (ID `55914294`) — **Public Score: 0.09**

---

## Phase 2 — LLM relay / ToolsAgent / VisionToolsAgent
`~2026-09-04 → 2026-09-07`

→ **09-04** Plan de fine-tuning GRPO pour le "brain" (petit LLM) — 2 runs réels, tous deux inconclusifs
→ ToolsAgent shipped → 🎯 **soumission réelle** (ref `56011798`) — **score 0.06** puis re-mesuré **0.01** (2 tentatives proches, cf `llm_relay_agent_experiments`)
→ **09-06** Score réel 0.00 sur une variante intermédiaire
→ **09-07** Grosse session LLM-relay : 6 fixes (brain-call budget, state-graph replay, cross-game memory...), UnifiedVisionAgent+calibration+GPU, détecteur de régions à palette partagée (inspiré des heuristiques de jeu du user) → 🎯 **soumission réelle** (ref `56065271`) — **Public Score: 0.06**

---

## Phase 3 — Pivot vers Duck Harness (REPL + World Model)
`2026-09-08 → aujourd'hui`

→ **09-08** Deux nouvelles architectures scoutées chez les leaders du domaine : REPL-tool harness + executable world model
→ **09-08/09** Reproduction locale de l'implémentation réelle de Duck Harness sur `ls20`
→ **09-09** Batterie level-skip + multi-jeux
→ **09-09** Root-cause du gap avec leurs claims (essai vLLM + test multimodal mesuré) → repro locale **mean 1.69** (colle à leur 1.6002 publié)
→ **09-09** 🏆 **MILESTONE : "hardware not architecture"** — leur vrai setup FP8+vLLM sur le RTX Pro 6000 Kaggle confirme **13/25 jeux, mean 1.69**
→ **09-09** Ablation par facteur critique : isole ce qui compte vraiment
→ **09-10** Stage 1 (infra Kaggle) : 4 fixes réels de plus, patch du harnais mécaniquement correct mais réponse vide persistante (10 runs Kaggle réels) — hypothèse affinée : `--reasoning-parser qwen3` est le vrai coupable, pas `--enable-auto-tool-choice`
→ **09-10** Scoping de la vraie mécanique de soumission concours (classe `MyAgent`/gateway sidecar) — incompatibilité d'architecture identifiée, design proposé mais pas encore construit
→ **09-11/12** Pivot stratégique : adoption du notebook Duck Harness réel non modifié comme "baseline v2" ancre du projet ; outils sandbox ajoutés (`SANDBOX_TOOLS.md`)
→ **09-12** `state_graph.py` priorisé (inspiré d'une approche 3ᵉ place du ARC-AGI-3 Preview Challenge)
→ **09-13** Étude de variance (4 passes) : variance intrinsèque énorme (ft09 : 0→43 sans aucun changement de code) → 🎯 **soumission "7 tools" réelle** (ref `56195748`) — **score 0.87** (un run local à 2.08 avait promis mieux → leçon de discipline de soumission)
→ **09-13** `gemma31` établi comme modèle local de substitution multimodal
→ **09-15** 🏆 **MILESTONE : "tools hurt, not help"** — sweep d'ablation confirme que les outils sandbox dégradent le score réel ; piste "plus d'outils" abandonnée
→ **09-16** 🏆 **MILESTONE : Executable World Models (EWM)** — real Kaggle 5-jeux×4-passes : **4.46** vs baseline **4.86** — meilleure config non-baseline trouvée
→ **09-17** 🎯 **soumission EWM réelle** (ref `56281544`) — **score 0.69** (régression vs baseline)
→ **09-17** Tests locaux : thinking-budget (−32% latence sans perte), bugs tool-call Ollama (vision), comparaison vision vs texte sur `ls20`
→ **09-19** 🏆 **MILESTONE : "throughput starvation"** — le vrai goulot n'est pas la stratégie de l'agent : 28 jeux concurrents se partagent 1 GPU (~8 tok/s chacun), tous meurent sur le cap de temps, 900s perdues par jeu en timeouts
→ **09-19/20** 🎯 **Baseline pur, resoumission propre** (ref `56369160`) — **score 1.00 — MEILLEUR SCORE RÉEL DU PROJET À CE JOUR**, bat "7 tools" (0.87) et EWM (0.69) (une 1ʳᵉ tentative, ref `56303316`, avait échoué en ERROR infra)
→ **09-20** Levier "batching addendum" testé localement → régresse (4→1.5 actions/tour), abandonné avant de gaspiller un run Kaggle
→ **09-20** Investigation variance `sb26` : confirmée comme vraie variance de modèle (pas un artefact de levier) ; 2 nouveaux leviers identifiés : toggle multimodal, tuning de la concurrence
→ **09-21** `G_bigcap` (levier budget-temps pur) testé → local mean **1.50** vs baseline 2.85 — ne marche pas, confirme la nécessité de la piste fine-tuning
→ **09-21** Fine-tuning Step 1 : dataset de 4065 tours de raisonnement miné depuis 75 vrais transcripts Qwen3.6-27B (58 progress / 212 no-effect) — bloqué sur mismatch driver GPU
→ **09-21** Fine-tuning Step B : 218 paires DPO générées depuis des runs EWM locaux (ministral-3:14b)
→ **09-22** GPU réparé. Pipeline STaR SFT (`star_sft_qwen38.ipynb`) entièrement débuggé (5 bugs fixés), dataset de 2753 tours depuis 20 jeux gagnants, modèle `qwen38-27b-fp8` téléchargé localement — entraînement à lancer manuellement par le user
→ **09-22** `H_nomodal` (v10, multimodal désactivé) testé → local mean **2.59** vs baseline 2.85 — confirme que le multimodal vaut son coût, pas encore resoumis en réel
→ **09-22** `I_concur14` (v11, concurrence 28→14) — **EN COURS** sur Kaggle au moment de cette note

---

## Récap des scores réels Kaggle (tous les scores officiels obtenus)

| Date | Config | ref/ID | Score réel |
|---|---|---|---|
| 2026-08-31 | RL from-scratch (Eyes/Brain/GRPO/JEPA) | 55914294 | **0.09** |
| ~2026-09-05 | ToolsAgent (premier) | 56011798 | **0.01** |
| 2026-09-07 | UnifiedVisionAgent + calibration + GPU | 56065271 | **0.06** |
| 2026-09-13 | Duck Harness + 7 tools sandbox | 56195748 | **0.87** |
| 2026-09-17 | Duck Harness + Executable World Model | 56281544 | **0.69** |
| 2026-09-19 | Duck Harness baseline pur (1ʳᵉ tentative) | 56303316 | ERROR (infra) |
| 2026-09-20 | Duck Harness baseline pur (resoumission propre) | 56369160 | **1.00 🏆 (meilleur à ce jour)** |

**Conclusion qui tient depuis le 09-20** : toute modification ajoutée au notebook Duck Harness (outils, EWM, prompt tweaks) a jusqu'ici *dégradé* le score réel par rapport au baseline pur. La piste actuelle (fine-tuning / STaR SFT) vise à casser ce plafond autrement qu'en ajoutant des outils ou du prompt engineering.

---
*Pour le détail narratif complet de chaque étape, voir `DESIGN_LOG.md` (jusqu'au 2026-09-10) et les fichiers de mémoire de session (2026-09-11 → aujourd'hui).*
