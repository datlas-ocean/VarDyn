# Contexte du projet VarDyn Tropics

Dernière mise à jour : 2026-11-09.

## Objectif scientifique

Implémenter VarDyn pour reconstruire la hauteur de surface de la mer (SSH) et les courants de surface dans les régions tropicales, à partir d'observations altimétriques de SSH et d'un modèle dynamique.

Le modèle retenu est QGSW, avec son cœur shallow-water (équations en eau peu profonde). L'assimilation des observations altimétriques dans ce modèle doit permettre de contraindre conjointement la SSH et la circulation de surface.

## Périmètre initial

- Mettre en place et configurer le cœur shallow-water de QGSW pour un domaine tropical.
- Préparer les observations altimétriques de SSH pour leur assimilation.
- Implémenter la chaîne VarDyn couplant le modèle et l'assimilation.
- Produire des reconstructions de SSH et de courants de surface.
- Évaluer ces reconstructions à l'aide d'observations indépendantes, selon un protocole à définir.

## Évolutions visées

À terme, faire évoluer le modèle pour représenter explicitement la couche de mélange afin de mieux décrire le forçage par le vent et sa contribution aux courants de surface.

Ajouter également l'advection de traceurs de température de surface (SST) et de salinité de surface (SSS). Les modalités de couplage de ces traceurs avec la dynamique, les flux de surface et le mélange restent à définir. Leur assimilation éventuelle ne fait pas partie du périmètre initial établi.

## État actuel

**Suivi de revue du 2026-11-09 :** à la demande de l’utilisateur, seul le point 2 (traceurs sans données extérieures aux frontières ouvertes) est corrigé. L’interface conserve l’absence de donnée par traceur dans les tableaux de frontières ; le cœur extrapole alors la concentration intérieure à chaque étape et désactive la relaxation correspondante. Une concentration extérieure explicitement nulle reste imposée. Validation ciblée terminée : **28 tests CPU réussis**, dont 16 cas nouveaux (frontières x/y, données absentes ou partiellement présentes, zéro explicite, éponge, JIT et gradients), plus les régressions SW existantes. Rapports dans `runs/fix_missing_tracer_bc_20261109/`. Le point 1, concernant le passage automatique au format BMaux historique lors du chargement d’un fond, reste en suspens et son code est inchangé.

Le projet est un worktree Git dans `/home/flo/Workspace/VarDyn_Tropics`. La branche active est désormais **`VarDyn_Tropics`**, à la demande de fusion du 2026-11-09. La fusion de **`Dev-SW`** est validée sans conflit : 81 tests CPU de la suite complète et 2 tests supplémentaires du pilote Optax réussis. Les corrections locales SW précédentes ont été enregistrées sur `Dev-SW` au commit `66c8aded` pour les préserver avant fusion. Les correctifs propres à Tropics, notamment L4 vide, sont conservés. Le dépôt principal `/home/flo/VarDyn` reste indépendant dans ses fichiers de travail.

**Disposition actuelle :** les outils locaux sont à la racine dans `scripts/`, `tests/` et `learning_local/`. Le dossier `mapping/` contient exactement les **43 fichiers de main local `350fa951`**, sur disque et dans l’index. Cette égalité porte sur les noms des fichiers ; les modifications fonctionnelles SW de la branche sont conservées. Les trois tests de référence de main restent dans `mapping/tests/`, tandis que les 15 fichiers de tests complémentaires sont dans le dossier racine `tests/`. Les quatre pilotes utilisent la racine du dépôt et chargent le cœur depuis `mapping/`. Déplacement validé dans `runs/root_local_layout_20261109/` : inventaires disque/index/main égaux, contenus mapping inchangés, 51 fichiers compilés, quatre --help réussis depuis /tmp, 40 tests CPU du cœur puis 43 tests d’apprentissage réussis. Le cœur reste sans import des modules locaux.

**Organisation demandée après fusion :** `mapping/src/` contient uniquement les **13 fichiers suivis par `main`** (référence locale `350fa951`). Les 17 fichiers supplémentaires — covariance.py et les 16 modules SW d’apprentissage — sont déplacés dans **`learning_local/`**, ignoré par Git et conservé sur disque. Les classes de covariance réellement partagées sont intégrées à `mapping/src/tools.py` ; basis.py et inv.py les importent depuis tools.py. Aucun module src ne doit importer le dossier local ni les modules déplacés. Les modifications de comportement SW déjà intégrées restent dans les fichiers historiques.

**Apprentissage local optionnel :** les quatre pilotes locaux dans `scripts/` utilisent le paquet local `learning_local`. Leurs chemins de provenance et les tests d’apprentissage sont adaptés ; les huit fichiers de tests optionnels sont ignorés à la collecte lorsque le paquet local est absent. Les dossiers racine `scripts/`, `tests/` et `learning_local/` sont ignorés par Git et conservés sur disque. Les trois tests de référence présents sur main sont conservés et suivis dans `mapping/tests/`. Une nouvelle copie Git contient ces trois tests, mais pas les sources locales d’apprentissage, les pilotes ni les tests complémentaires. Les configurations scientifiques et données historiques restent à fournir. Guide : `docs/sw_learning.md`. L’inventaire `docs/sw_learning_source_inventory.json` décrit la fusion historique ; le déplacement actuel est détaillé dans `runs/src_main_layout_20261109/moves.json`.

**Validation historique de la fusion :** 81 tests CPU puis 2 tests Optax réussis, compilation de 36 fichiers et 29 imports src locaux autonomes, avant la réorganisation. Logs dans `runs/merge_dev_sw_20261109`. La nouvelle organisation est validée dans `runs/src_main_layout_20261109` : audit AST sans import des 17 modules déplacés ; tous les modules du cœur importés et **40 tests CPU réussis avec les imports locaux interdits**, 8 fichiers optionnels ignorés ; **43 tests d’apprentissage local réussis** séparément. Compilation py_compile de 34 fichiers, quatre CLI --help et git diff --check réussis. Index Git src identique à main ; 17 fichiers locaux présents, non suivis et ignorés. Aucun entraînement scientifique ni test GPU.

Comparaison directe de Dev-SW au main local `350fa951` : 11 fichiers, 2 770 ajouts et 345 suppressions ; extensions du modèle SW, bases de forçage, covariances multivariées, fenêtres de forçage et minimiseur par blocs. Rapport : [comparaison Dev-SW/main](runs/dev_sw_comparison_20261106/report.md). La branche Dev-SW est locale et absente du distant configuré, y compris après fetch ; aucune version distante plus récente n'a été identifiée.

**Corrections SW du 2026-11-09, enregistrées dans `66c8aded` avant fusion :** les remarques Plannotator sont implémentées. Les bases SW journalière, générique et calibrée et leurs opérateurs spatiaux sont réunis dans `mapping/src/basis.py`. Les helpers spécifiques de compatibilité BMaux sont dans sa classe. Les opérateurs indépendants/diagonaux partagés, initialement placés dans `mapping/src/covariance.py`, se trouvent désormais dans `mapping/src/tools.py` après la réorganisation. Les imports locaux n'ont plus besoin des modules non suivis du checkout historique. Aucun fichier de `/home/flo/VarDyn` n'a été modifié.

**Décisions et comportement :** autocorrélations temporelles conservées (AR(1), ACF et fenêtres), bases de forçage par fenêtres conservées, corrélations croisées BMaux et signatures multivariées retirées. Le minimiseur block-coordinate, ses configurations, checkpoints, évaluateurs locaux et métadonnées de fenêtres sont supprimés ; les minimiseurs SciPy et Optax existants restent utilisables. Les contrôles enregistrés par l'ancien mode bloc ne sont plus pris en charge par ce mode supprimé.

Les frontières caractéristiques sont disponibles sur x et y (masques humides, caractéristiques entrantes, flux, intégrateurs sans/avec traceurs) via `open_boundary_x`/`open_boundary_y`, fermées par défaut. La divergence impose un flux nul sur les faces extérieures fermées, y compris lorsqu'un autre axe est ouvert. Les cartes auxiliaires peuvent fournir g_prime en configurant son nom dans `name_var_params_aux` ; elles alimentent la fermeture physique, les conversions SSH/interface, la CFL, l'affichage et les sorties. La transformation log commune conserve les planchers/plafonds H et Hwind et les cas particuliers H/c.

**Validation :** 31 tests du checkout réussis, puis deux tests supplémentaires ciblés réussis (intégration frontières y/xy avec traceur constant et variance/adjoint de la base calibrée), sans chargement de sources historiques. Couverture : bases/AR(1), géométrie et masse, RMS, opérateurs et gradients SW Python/scan/différences finies, frontières et budget de flux, carte g_prime/SSH/c, contrôles-log, reprises et refus du minimiseur supprimé. Syntaxe, py_compile de mod.py/sw.py et git diff --check réussis. Logs, PID, invocations et résultats : `runs/review_changes_20261109/`. Tests CPU ; aucune nouvelle OSE annuelle ni validation GPU. Le correctif L4 de main, hors annotations traitées lors de cette validation historique, est maintenant conservé par la fusion dans VarDyn_Tropics.

**Trois défauts de revue corrigés le 2026-11-09 :** le calcul de profondeur du vent reçoit aussi g_prime_control dans les intégrateurs avec/sans traceurs, y compris lorsque le plancher H*g_prime couple les deux contrôles. La gravité de référence QGSW résolue est portée par l'état (et conservée par copie) pour valider les calibrations contre les champs réellement utilisés, y compris cartes ou diagnostic H/c ; les calibrations incompatibles restent refusées. Les concentrations extérieures des traceurs sont utilisées sur les faces ouvertes entrantes, les valeurs intérieures sur les faces sortantes ; sans concentration extérieure, l'extrapolation précédente est conservée. La donnée c_b traverse les étapes d'intégration et fonctionne sans éponge. Quatre régressions ajoutées ; suite complète **37 tests CPU réussis**, imports locaux autonomes, syntaxe/py_compile et git diff --check réussis. Tests des quatre faces, quatre schémas, plusieurs traceurs, intégration/JIT/gradient de bord, référence diagnostiquée/cartographiée, copie d'état et cohérence profondeur/gradient du vent. Logs/PID : `runs/review_fixes_20261109/`. Pas de nouvelle OSE annuelle ni de commit/push.

**Suite :** rattacher les configurations/données des expériences originales puis préparer un pilote tropical avant de reproduire les slides. Le cœur versionné est indépendant des sources locales d’apprentissage ; les pilotes et modules d’apprentissage locaux sont disponibles dans ce worktree. Les tests synthétiques CPU ne reproduisent pas les OSE annuelles. La fusion est enregistrée localement, sans push.

Les spectres altimétriques locaux restent disponibles ; les expériences d'assimilation des slides sont rapportées par leur source, sans reproduction dans ce worktree.

Réplication spectrale Atlantique terminée (2026-11-09), à la demande de l'utilisateur, dans `runs/tropical_atlantic_spectrum_20261109/` : **6°S–6°N, 44°W–10°E**, mêmes dates, huit missions et trois variables SLA/LWE/SLA−LWE que dans le Pacifique. Le domaine est la plus longue bande contiguë satisfaisant >=90 % de couverture par bande de latitude et >=95 % en moyenne, après extraction large 60°W–16°E ; ces critères évitent les colonnes entières non observées mais ne constituent pas un masque terrestre. Les bandes côtières peuvent ne représenter que leur portion marine. Même grille de 2 jours × 2° × 4° et même méthode Welch ; **97,4283 %** de cellules observées, 752 trous interpolés localement. Pas spectral zonal **0,018519 degré⁻¹**, plus grossier que dans le Pacifique du fait des 54° de largeur. Variances après traitement : SLA 7,850020 cm², LWE 0,813764 cm², SLA−LWE 9,321354 cm². Parseval, linéarité, signe est/ouest et réouverture NetCDF vérifiés ; figures en anglais inspectées. Rapport et limites dans `report.md`, données `spectra_atlantic.nc`. Les modifications locales VarDyn sur `Dev-SW` sont préservées ; aucune identification de mode ni validation des hautes fréquences revendiquée.

## Documents de référence et lecture à chaque prompt

À chaque nouveau prompt, lire intégralement, dans cet ordre :
[AGENTS.md](AGENTS.md), [CONTEXT.md](CONTEXT.md), [README.md](README.md)
et [JOURNAL.md](JOURNAL.md), puis vérifier l’état Git et les fichiers concernés.
`AGENTS.md` contient directement les consignes et constitue leur point d’entrée
standard. Ce contexte conserve la fusion du contexte Tropics et du vocabulaire
technique partagé ; le contexte de projet précède le vocabulaire.

## Documentation de la branche

`README.md` décrit la branche VarDyn_Tropics : modèles, conventions SSH/interface,
contrôles physiques, bases, observations, trois chemins de minimisation,
reprises, sorties, orchestration et protocole d’apprentissage. Les exemples
scientifiques viennent des comptes rendus locaux de `/home/flo/VarDyn`
(vent seul, MAP1/MAP2 GLORYS, OSE nadir 2024 et cibles sous la couche mélangée).
Le README distingue le cœur versionné, les 17 modules d’apprentissage locaux
ignorés par Git et les ressources historiques absentes d’une nouvelle copie.
La présence d’un chemin numérique n’est pas présentée comme une validation GPU
ou une reproduction annuelle. Les liens locaux et la syntaxe des exemples sont
vérifiés ; aucune source ni expérience n’a été modifiée pour cette rédaction.

## Référence scientifique et premières expériences

L'utilisateur a corrigé le nom de branche : **`Dev-SW`**, et non `VarDyn-SW`. Le nom de la présentation reste `VarDyn-SW`.

Source fournie par l'utilisateur : [VarDyn-SW — contexte scientifique et premières expériences](https://docs.google.com/presentation/d/118r7YlfgA3VTehZW2Bu-kNpaTCNZqA5-Kn2LlNoFVc0/edit), 43 diapositives consultées le 2026-11-06, révision `raKVVbff2fismA`. Les éléments ci-dessous sont **rapportés dans les slides**, sans reproduction ni vérification des sorties originales dans ce worktree. Cette référence concerne VarDyn Tropics et est distincte des slides de VarDyn_IT mentionnées dans `AGENTS.md`.

### Modèle et base réduite (slides 2–12)

- Modèle shallow-water à une couche, état vitesse horizontale et déplacement d'interface η ; SSH = (g′/g)η. Advection, Coriolis, pression, traînée, diffusion, vent ERA5 τ/(ρ₀ Hwind), forçage de quantité de mouvement F et source de hauteur Fh ; intégration sur grille C.
- F = ∇φ + ez × ∇ψ : composantes divergente et rotationnelle. Atomes spatiaux locaux fixes, fenêtres cos², huit bandes entre 80 et 971 km. Les atomes de hauteur compensés ont une masse nulle par bassin ; une composante de hauteur moyenne par bassin est aussi présente.
- Initialisation des amplitudes par défauts journaliers de GLORYS relativement aux tendances SW avec ERA5 ; projection conjointe des huit bandes par régression ridge pondérée par l'aire. Calibration journalière sur SSH et, selon la stratégie, courants GLORYS ; estimation locale des moyennes, écarts-types et autocorrélations des coefficients.
- Compression temporelle par fenêtres cos² compactes, normalisées en partition de l'unité, espacées d'une demi-largeur w/2 ; w ajuste l'autocorrélation journalière. Tdec désigne le temps de décroissance à 1/e de la corrélation résultante. Les amplitudes sont renormalisées pour conserver l'écart-type calibré.

### Protocole tropical (slides 14–15)

- Apprentissage GLORYS : juillet 2020–juin 2021. Quatre stratégies : SSH + U/V de surface (référence), SSH seule, SSH + U/V moyennés sur 30 m sous la profondeur de couche de mélange `mlotst`, SSH + U/V interpolés à `mlotst + 50 m`.
- OSE nadir réelle : janvier–décembre 2024, SSH seule, mission AL exclue de l'assimilation pour validation ; pas de temps 900 s, vent ERA5, SLA MIOST aux frontières. Les conditions aux limites U/V suivent la cible d'apprentissage.
- Apprentissage et OSE optimisent quatre champs physiques : H, g′, Hwind et traînée. L'OSE repart d'un forçage nul ; prior physique centré sur MAP1, écarts-types multipliés par 0,5. Les fichiers, le domaine exact de chaque étape et les réglages complets restent à identifier.
- Convergence : MAP1 SSH seule et OSE « mean30 » non convergés ; autres OSE arrêtés selon ftol, gtol désactivé. Le résultat mean30 fini a été exporté après échec de recherche linéaire.

### Résultats et limites rapportés (slides 16–23 et 39–43)

Sur 876 285 échantillons AL communs, protocole **SLA brute sans soustraction LWE**, MIOST atteint 2,4887 cm de RMSE :

| Cible d'apprentissage | RMSE VarDyn | Gain relatif vs MIOST |
| --- | --- | --- |
| SSH + U/V de surface | 2,4256 cm | 2,53 % |
| SSH seule | 2,4469 cm | 1,68 % |
| SSH + moyenne 30 m sous MLD | 2,4137 cm | 3,01 % (OSE non convergée) |
| SSH + U/V à MLD + 50 m | 2,4078 cm | 3,25 % |

- Le meilleur score ponctuel SSH brute (MLD + 50 m) ne s'étend pas aux courants : RMSE U = 13,22 contre 11,19 cm/s pour MIOST (+18,16 %), V = 11,37 contre 11,12 cm/s (+2,25 %). Domaine drifters/spectres : 60°W–14°E, 15°S–15°N, 15 janvier–15 décembre 2024 ; le score AL brut couvre le domaine tropical complet, dont les bornes restent à relever dans les configurations.
- Les cartes et spectres SLA emploient la référence `sla_unfiltered - lwe`, à distinguer du classement AL brut. Pour MLD + 50 m : amélioration dans 973/1 581 cellules de 1°, changement médian RMSE −0,038 cm ; résolution effective améliorée dans 885/1 603 cellules, changement médian −1,30 km. Les scores spectraux SLA restent inférieurs à MIOST à 200, 300 et 500 km.
- Mean30 : RMSE U = 11,17 et V = 10,74 cm/s, gains rapportés de 0,17 % et 3,36 % ; résolution médiane dégradée de +0,98 km. Ces résultats restent assortis de la non-convergence de l'OSE.
- Les spectres complexes u + iv décrivent deux sens de rotation aux fréquences positives/négatives, pas les scores séparés de U et V. Aucun intervalle d'incertitude du classement n'a été relevé dans le texte des slides consulté.

Suite documentaire : retrouver les configurations et sorties originales, établir leur lien avec la version actuelle du code, puis définir les prochaines expériences en tenant compte du compromis SSH/courants, de la convergence et du protocole LWE.

## Points à préciser

Extension terminée à la demande de l'utilisateur : spectres de `lwe` et `sla_unfiltered - lwe`, mêmes observations et traitement, sorties PNG/PDF et NetCDF dans `runs/tropical_pacific_spectrum_20261106/lwe_comparison/`. Selon les métadonnées des huit missions, `sla_unfiltered - lwe` rétablit le signal avant la correction LWE déjà appliquée au produit. Comparaison sur la même échelle de couleurs absolue que le premier spectre ; alignement exact des échantillons, support commun, linéarité des champs, Parseval et réouverture NetCDF vérifiés. Variances spectrales après traitement : SLA 10,473794 cm², LWE 0,804951 cm², SLA − LWE 12,427873 cm² ; co-spectre intégré SLA–LWE −0,574564 cm². Les limites d'échantillonnage et d'interprétation du calcul initial sont conservées.

Une analyse spectrale autonome est terminée dans `runs/tropical_pacific_spectrum_20261106/` : spectre zonal–fréquence de SLA L3 CMEMS, 6°S–6°N, 140°E–80°W, domaine validé par l'utilisateur. Extraction sur la plage commune nominale 2023-01-16–2025-01-07 ; fenêtres spectrales couvrant effectivement 2023-01-16–2025-01-04 inclus. Huit missions combinées, avec des lacunes propres à chacune (notamment HY-2B après le 2024-12-20). Grille de 2 jours × 2° longitude × 4° latitude, 97,6085 % de cellules observées ; interpolation locale des trous. Welch 2D de 180 jours, recouvrement 50 %, Hann et détrendage temporel linéaire, puis moyenne de puissance entre trois bandes de latitude. Figures PNG/PDF, données `spectrum.nc`, méthode et limites dans `report.md`. Parseval et signe est/ouest vérifiés ; lobe ouest à environ 30 jours et 12,7° de longueur d'onde. Courbes théoriques indicatives à c = 2,8 et 1,7 m/s, sans ajustement ni identification formelle de mode. Les fréquences supérieures à 0,1 cycle/jour restent particulièrement sensibles aux alias ; Nyquist nominal ne signifie pas résolution effective. Cette analyse ne modifie pas le code VarDyn et ne constitue pas une expérience d'assimilation. Suites possibles : contrôler les alias et la sensibilité aux missions, puis définir la prochaine expérience tropicale d'assimilation en tenant compte des OSE rapportées dans les slides.

Revue du 2026-11-06, avant rebase : les deux commits propres à l'ancienne branche
depuis son ancêtre commun avec `main` ont été examinés. Les 12 tests ajoutés
passaient sur CPU ; deux cas limites de la correction gaussienne ont été relevés :
facteur de variance nul rendant la covariance singulière et coordonnées de
longitude dupliquées après conversion d'une carte globale incluant les deux
extrémités. Aucune correction dédiée n'a été apportée. Les différences historiques
avec `main` sont consignées dans `JOURNAL.md` ; elles ne décrivent plus l'état
actuel après rebase. Aucun nouveau test scientifique n'a été lancé après rebase.

- Domaine géographique tropical, période d'étude et résolutions spatiale et temporelle.
- Version de QGSW, configuration verticale, paramètres physiques et conditions aux limites.
- Formulation de VarDyn, variables de contrôle, fonction coût, contraintes et méthode d'optimisation.
- Missions altimétriques, produits SSH, corrections, contrôle qualité et erreurs d'observation.
- Sources du forçage atmosphérique et traitement initial du vent.
- Données indépendantes, expériences de référence et métriques de validation de la SSH et des courants.
- Formulation de la couche de mélange et traitement des traceurs SST et SSS lors des évolutions futures.

## Organisation documentaire

- Règle de présentation : tous les textes des figures sont en anglais (titres, axes, légendes, barres de couleurs, annotations et captions), conformément à `AGENTS.md`. Les scripts et figures de l'analyse spectrale ont été traduits en ce sens.

- `AGENTS.md` porte les consignes et la procédure de lecture à chaque prompt.
- `README.md` décrit les fonctionnalités et leur utilisation.
- `CONTEXT.md` réunit les objectifs, le périmètre, les choix structurants et le vocabulaire technique partagé. Le mettre à jour lorsque ces éléments évoluent.
- `JOURNAL.md` consigne chronologiquement les avancées, décisions, expériences, résultats et difficultés. Ajouter une entrée datée à chaque avancée significative et conserver l'historique.

## Terminologie

| Terme | Signification dans le projet |
| --- | --- |
| VarDyn | Méthode de reconstruction dynamique à implémenter ; formulation détaillée à documenter. |
| QGSW | Modèle retenu, utilisé ici avec son cœur shallow-water. |
| SSH | Hauteur de surface de la mer ; variable observée par altimétrie et cible de reconstruction. |
| Courants de surface | Composantes horizontales de la vitesse à la surface, à reconstruire avec la SSH. |
| Couche de mélange | Couche supérieure de l'océan dont la représentation explicite est une évolution visée. |
| SST | Température de surface de la mer ; futur traceur à advecter. |
| SSS | Salinité de surface de la mer ; futur traceur à advecter. |

## Vocabulaire technique partagé

VarDyn reconstructs ocean dynamical variables by combining a reduced control
space, dynamical propagation, observations, and variational inversion.

### Branch-specific context

When the active Git branch is `Dev-SW`, **read
[CONTEXT_Dev-SW.md](/home/flo/VarDyn/CONTEXT_Dev-SW.md) in addition to this file
before working on the branch**. It records the GLORYS shallow-water learning,
MAP1/MAP2 results, active SSH nadir experiments, covariance coupling, and
follow-up instructions. Recheck its dated run status against the live outputs.

### Language

**North Atlantic Midlatitude Domain**:
The Atlantic region from 80°W to 0° and 30°N to 60°N used for the
GLORYS learning experiment.
_Avoid_: Tropical Atlantic domain

**North Atlantic Absolute SSH**:
The GLORYS sea-surface height used without subtraction of a spatial mean in
the North Atlantic calibration, boundary fields, and synthetic nadirs.
_Avoid_: North Atlantic SSH anomaly

**Admissible Reference Depth**:
The uniform initial SW layer depth chosen so the year's absolute GLORYS SSH
starts with positive total layer thickness throughout the North Atlantic domain.
_Avoid_: Chelton-derived regional depth, learned final depth

**OSE**:
An ocean reconstruction experiment assimilating measurements from real observing systems.
_Avoid_: OSSE, synthetic-observation experiment

**Withheld Real Nadir**:
Along-track sea-level measurements from a real altimeter mission excluded from the OSE observation cost and used to assess reconstructed SSH.
_Avoid_: Assimilated nadir, synthetic nadir

**First-Baroclinic-Mode Current**:
The part of a three-dimensional current associated with the first vertical baroclinic mode, whose shape depends on water-column stratification and bottom depth.
_Avoid_: Current at a single depth, sub-mixed-layer current

**Sub-Mixed-Layer Target Current**:
A GLORYS current target averaged over a thin layer immediately below the density-defined mixed layer, rather than sampled at the surface or interpreted as a full-depth baroclinic mode.
_Avoid_: First-baroclinic-mode current, surface current

**Mixed-Layer-Offset Current**:
A GLORYS current sampled at a specified depth below the density-defined mixed-layer base, distinct from a mean over a layer beneath that base.
_Avoid_: First-baroclinic-mode current, fixed-depth current

**OSSE**:
An ocean reconstruction experiment assimilating observations sampled from a reference simulation.
_Avoid_: OSE, real-observation experiment

**MAP1-Centered OSE Physical Prior**:
A prior for real-nadir reconstruction whose physical-parameter fields are centered on the GLORYS-trained MAP1 estimate, with reduced uncertainty so real observations can adjust them.
_Avoid_: MAP1 physics fixed for OSE, original-reference physical prior

**GLORYS OSE Boundary SLA**:
A GLORYS height boundary field expressed relative to GLORYS mean dynamic topography, matching the real-nadir SLA convention.
_Avoid_: Absolute GLORYS height in an SLA analysis

**Direct GLORYS Boundary Current**:
A GLORYS current boundary field supplied as a full current without subtracting a mean current.
_Avoid_: Current anomaly

**Analysis Window**:
The time interval jointly optimized by one 4DVar inversion.
_Avoid_: Batch, run period

**Checkpoint**:
A time in the Analysis Window where VarDyn applies the Reduced Basis, evaluates
observation misfits, or stores state needed by the adjoint.
_Avoid_: Snapshot, save time

**Control Vector**:
The coefficients optimized by 4DVar; the Reduced Basis maps them to model
variables and parameters.
_Avoid_: Parameter vector, weights

**Reduced Basis**:
The mapping between the Control Vector and corrections to model variables or
parameters over space and time.
_Avoid_: Encoder, feature basis

**Model-Error Forcing**:
A time-dependent additive correction to the model tendencies, inferred by the
variational analysis. Its components follow the model variables they force.
_Avoid_: State correction, analyzed state

**Forcing Bias**:
The time mean of a Model-Error Forcing over its calibration period, stored
separately from the zero-mean covariance represented by BMaux.
_Avoid_: Mean BMaux coefficient, background state

**Analyzed-Forcing Calibration**:
The estimation of Model-Error Forcing statistics from corrections inferred by
variational analysis. These statistics describe regularized analyzed corrections,
not an unbiased model-error covariance or equation defects evaluated on reference states.
_Avoid_: Direct GLORYS residual calibration, observed physical forcing

**Prescribed Forcing Bias**:
The learned temporal mean of Model-Error Forcing applied as a fixed background
correction, separately from the covariance of centered forcing fluctuations.
_Avoid_: Zero-mean constant control, bias standard deviation

**BMaux Temporal Scale**:
The local half-width of a compact temporal basis function, calibrated from the
autocovariance of zero-mean Model-Error Forcing at one spatial scale.
_Avoid_: E-folding time, regional decorrelation time

**BMaux Calibration Domain**:
The wet model domain used to estimate Model-Error Forcing statistics, including
sponge zones and excluding land.
_Avoid_: Diagnostic interior, propagation-safe domain

**Spatial Filter Support**:
The fraction of a scale-dependent filter's weighted footprint covered by valid
calibration data. A band is locally supported when this fraction reaches 50%.
_Avoid_: Temporal sample coverage, wet-point fraction

**BMaux Scale Grid**:
The eight spatial wavelengths actually represented by BMaux between 80 and
1000 km; denser filter bands are calibration diagnostics rather than controls.
_Avoid_: Diagnostic wavelength grid, arbitrary spectral bins

**Interface-Displacement Forcing**:
The vertical Model-Error Forcing expressed in the SW model's interface
coordinate, distinct from physical sea-surface-height forcing.
_Avoid_: SSH tendency, sea-level forcing

**Basin-Mean Height Forcing**:
The area-weighted spatial mean of Interface-Displacement Forcing within each
connected basin, which may vary in time. Its spatially uniform contribution is
distinct from the remaining zero-integral height forcing and from a temporal Forcing Bias.
_Avoid_: Temporal mean forcing, physical SSH forcing

**Local Multiscale SW Forcing Calibration**:
The estimation of spatially varying amplitudes and temporal scales for
divergent, rotational, and zero-integral height Model-Error Forcing at multiple
coexisting spatial scales. Basin-Mean Height Forcing is characterized separately
for each connected basin.
_Avoid_: Homogeneous forcing calibration, single local spatial scale

**SW Forcing Wavelength**:
The spatial scale of a forcing component identified by the peak of variance
per logarithmic wavenumber interval. It is distinct from the compact support
radius of the potential or compensated-height function that represents it.
_Avoid_: Potential radius, compact half-width

**BMaux Auxiliary Set**:
The three variable-specific auxiliary files that jointly define the covariance
of zonal, meridional, and interface-displacement Model-Error Forcing.
_Avoid_: Combined auxiliary file, shared forcing prior

**Calibrated BMaux Temporal Scale**:
A BMaux Temporal Scale written in physical days and consumed without an
empirical multiplier, within the identifiable range of 1 to 60 days.
_Avoid_: Scaled Tdec, heuristic decorrelation factor

**Stationary Forcing Covariance**:
The covariance of temporally centered Model-Error Forcing over the calibration
period, without removing a trend or seasonal harmonic.
_Avoid_: Seasonal covariance, detrended forcing covariance

**SW Coefficient Decorrelation Time**:
The lag at which the modeled temporal correlation of SW forcing coefficients
reaches `1/e`. The original daily AR1 model uses an exponential. The reduced
local MAP2 fits the covariance induced by its actual temporal-window bank,
matching temporal centering in the empirical ACF; its reported `tdec` is the
1/e crossing before sample-mean subtraction. Window width and node spacing
are separate quantities.
_Avoid_: Temporal node spacing, cosine-envelope half-width

**SW Coefficient Standard Deviation**:
The stationary standard deviation of centered SW forcing coefficients at
temporal control nodes, for a specified spatial-function normalization.
_Avoid_: Innovation standard deviation, pointwise forcing amplitude

**Coupled Spectral Calibration**:
The joint inference of BMaux component variances and temporal scales from
overlapping diagnostic bands using the basis's spectral response.
_Avoid_: One-band-one-component fit, independent band calibration

**Log-Spectral Variance Fit**:
The strictly positive inference of BMaux component variances by matching the
logarithms of measured and reconstructed band variances, with curvature
regularization across wavelength. It gives comparable influence to each
resolved spectral decade while admitting smooth power-law spectra.
_Avoid_: Absolute-variance NNLS, zero-filled component spectrum

**Cross-Forcing Diagnostic**:
The scale-dependent correlation among zonal, meridional, and
interface-displacement Model-Error Forcing, evaluated on a common h grid. It is
reported as a diagnostic because the current variable-specific BMaux bases do
not encode cross-variable covariance.
_Avoid_: Coupled BMaux prior, multivariate auxiliary covariance

**Completed Auxiliary Field**:
A BMaux field completed independently at each wavelength from directly
supported wet points by spatial regularization within the connected ocean.
Direct and completed values remain distinguishable through support, fit-quality,
and completion masks; land fill exists only to make interpolation well-defined.
_Avoid_: Unflagged extrapolation, land estimate

**Temporal-Scale Uncertainty**:
The 10th-to-90th-percentile interval of a BMaux Temporal Scale obtained by
resampling the stationary calibration record in 30-day temporal blocks.
Boundary-hitting or weakly constrained estimates are flagged and completed
spatially before operational use.
_Avoid_: Formal posterior interval, independent-day bootstrap

**Source-Basis Resolution Diagnostic**:
The comparison of observed forcing spectra with the synthesis response and
conditioning of the Gaussian source basis. It distinguishes spectral energy
present in analyzed forcings from scales that its coefficient lattice resolves
independently; requested 80 km components remain represented and quality-flagged.
_Avoid_: Hard nominal-scale cutoff, energy-only resolution claim

**Cost-Gradient Evaluation**:
One complete background-cost, forward-model, observation-misfit, and adjoint
calculation for a Control Vector.
_Avoid_: Objective call, model pass

**4DVar Minimizer**:
The algorithm that proposes Control Vectors and accepts steps using
Cost-Gradient Evaluations.
_Avoid_: Solver, optimizer backend

**Historical SciPy Minimizer**:
The host-resident SciPy L-BFGS-B 4DVar Minimizer retained for compatibility and
reference comparisons.
_Avoid_: CPU mode, legacy solver

**Device-Resident Optax Minimizer**:
The L-BFGS 4DVar Minimizer whose Control Vector and history remain on the JAX
device while Python decides the scalar Armijo line search.
_Avoid_: Full-GPU solver, JAXopt mode

**Converged Analysis**:
An Analysis Window whose accepted cost changes satisfy the configured relative
cost tolerance for the required consecutive iterations.
_Avoid_: Small-gradient analysis, exhausted iteration budget

**Annual Shared Forcing Statistics**:
The spatially local forcing-coefficient mean, stationary standard deviation,
and temporal decay scale inferred jointly from the year's Analysis Windows.
They are common across windows, rather than averages of independently fitted
window parameters.
_Avoid_: Monthly parameter average, spatially uniform annual prior

**Window Moment Ownership**:
The assignment of each analysis date to one Analysis Window for annual
statistical aggregation. Observations can inform multiple overlapping windows
while the selected posterior moments contribute only once.
_Avoid_: Observation exclusion, duplicated annual moments

**Continuous Annual Empirical Calibration**:
The October 7, 2026 experiment reconstructs one uninterrupted annual trajectory
with daily forcing controls and a permissive physical prior (tdec 2 days).
The revised MAP2 learns mean, variance and ACF locally for every spatial atom,
fits the ACF of compact temporal windows, and reduces temporal centers as their
width increases. It starts from zero physical forcing controls, with the same
initial ocean state as MAP1, and performs no posterior covariance solves.
The earlier pooled-statistics, warm-start MAP2 and its sensitivity chain are
superseded; their outputs remain archived. See
`docs/sw_reduced_local_map2_20261007.md` for the revised strategy and
`docs/sw_annual_empirical_learning.md` for the historical daily strategy.
_Avoid_: Windowed EM, posterior-corrected variance, assumed prior independence

**Equal Physical Scale Amplitude**:
A common expected wet-area RMS of the total forcing contribution at each
wavelength, separately for divergent momentum, rotational momentum and height.
The coefficient prior accounts for derivatives, masks and overlapping atoms.
The October 7 experiment uses 0.1 (m/s)/day for each momentum component and
1 m interface/day for height, with amplitude sensitivity factors 0.5 and 2.
_Avoid_: Equal potential coefficient std, equal amplitude per isolated atom

**GLORYS-Informed Initial Physical Amplitude**:
An experiment-specific initialization measured from daily GLORYS equation
defects with the current fixed Chelton/g'=0.025 model, jointly projected onto
the daily spatial potential basis. The October 7 configuration
`config_sw_daily_glorys_prior.py` uses physical RMS values 0.0304 and 0.0198
(m/s)/day for divergent and rotational forcing, 13.5 m interface/day for
compensated height and 1.23 m interface/day for basin means, with tdec=2 days.
Each component keeps the same RMS across wavelengths. These are approximate
initial amplitudes, not learned posterior uncertainties or SSH standard
deviations. See `docs/glorys_daily_prior_20261007.md` for support and limitations.
