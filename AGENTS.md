# Agent Instructions

## MASSH

For any prompt related to MASSH (mapping, data assimilation, SSH, ocean state estimation, or anything referencing the MASSH codebase), read the context file at [MASSH/AGENTS.md](MASSH/AGENTS.md) before proceeding.

## Project organization and memory

Apply these rules to every project, respecting its existing document names and
more specific instructions.

- À chaque nouveau prompt utilisateur, avant toute analyse ou action sur ce
  projet, lire intégralement et dans cet ordre [AGENTS.md](AGENTS.md),
  [CONTEXT.md](CONTEXT.md), [README.md](README.md) et
  [JOURNAL.md](JOURNAL.md). Cette lecture s’applique à chaque prompt,
  y compris dans une conversation déjà commencée.
- Après cette lecture, vérifier les fichiers concernés, la branche Git active
  et les modifications non committées avant de poursuivre.
- Utiliser ces noms avec leur casse exacte. `CONTEXT.md` réunit le contexte du
  projet et le vocabulaire technique ; `JOURNAL.md` conserve l’historique.
- Keep the context focused on the current objective, scope, organization,
  confirmed decisions, constraints, and next steps. Identify open questions
  and assumptions explicitly.
- After each significant step, append a dated journal entry describing the
  action and its purpose, the outcome, decisions and their rationale where
  useful, checks and their results, limitations or blockers, and known next
  steps. Significant steps include setup, a decision affecting subsequent work,
  a functional change, an experiment, a fix, or discovery of a blocker. Group
  minor operations belonging to the same step.
- Update the context when the objective, scope, organization, decisions,
  constraints, or next steps change. Keep the chronological history in the
  journal and distinguish verified results from assumptions.
- Before finishing a significant step, check that the journal reflects the
  completed work and the context remains accurate. Keep these documents
  versionable with the project; follow the user's instructions for commits.

## Running experiments

Write all figure text in English, including titles, axis labels, legends,
colorbar labels, annotations, and captions.

Run computations in a detached background process that survives the session,
using `nohup` with redirected input/output or a scheduler. Keep logs and the
process/job identifier, and verify completion. Follow the project's
`runs/README.md` when available.

## VarDyn — protection du code source

- Modifier le code source de VarDyn, notamment `mapping/src/` et
  `mapping/models/`, uniquement sur demande explicite de l'utilisateur portant
  sur une modification de ce code.
- Si une modification semble nécessaire sans avoir été explicitement demandée,
  présenter les fichiers concernés, le motif et les effets attendus, puis
  demander la validation de l'utilisateur et attendre sa réponse avant de
  modifier le code. Cette règle couvre aussi les copies du code et les patchs
  qui changent son comportement à l'exécution.
- Une demande de lancement d'expérience, de conversion de configuration ou de
  diagnostic ne constitue pas une autorisation de modifier le code source.
  Adapter en priorité la configuration au code existant.

## VarDyn_IT

These instructions apply to `/home/flo/Workspace/VarDyn_IT`:

- Preserve the roles of `CONTEXT.MD` (project context) and `CONTEXT.md` (shared
  VarDyn technical vocabulary).
- Keep the shared CSWM/BMIT preparation and wiring of interior internal-tide
  generation in `mapping/src/mod.py`.
- Report experiment configurations, figures, error metrics and conclusions in
  the [project progress slides](https://docs.google.com/presentation/d/17ZrZV9ekCASZ6njfnmCSepVjUdj5IX3ccx-sqqcILjw/edit?usp=sharing).
  Distinguish verified results, hypotheses and computations still in progress.
