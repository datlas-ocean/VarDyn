# Journal du projet VarDyn Tropics

Ce journal suit les avancées du projet par ordre chronologique. Pour chaque expérience, préciser autant que possible la configuration, les données, la commande ou le script utilisé, ainsi que l'emplacement des résultats afin de permettre sa reproduction.

## 2026-11-06 — Initialisation et cadrage

### Avancées

- Définition de l'objectif : reconstruire la SSH et les courants de surface dans les tropiques avec VarDyn, en assimilant les observations altimétriques de SSH.
- Choix de QGSW et de son cœur shallow-water comme base dynamique.
- Identification des évolutions futures : représentation explicite de la couche de mélange pour mieux décrire le forçage par le vent, puis advection des traceurs SST et SSS.
- Création de `context.md` et de ce journal.

### État des travaux

Le cadrage scientifique est posé. L'implémentation, la préparation des données et les expériences restent à réaliser ; aucun résultat n'est disponible à ce stade.

### Prochaines étapes proposées

1. Définir le domaine tropical et la période d'une première expérience.
2. Identifier la version de QGSW à utiliser et vérifier l'exécution d'un cas shallow-water de référence.
3. Préciser la formulation de VarDyn et son interface avec QGSW.
4. Choisir les produits altimétriques et les données de validation indépendantes.
5. Définir une première expérience d'assimilation et ses critères d'évaluation.

---

## 2026-11-06 — Création de la branche de développement

- Création de `VarDyn_Tropics` dans le dépôt `/home/flo/VarDyn`, depuis la branche locale `Dev-Tropics`.
- La branche de départ inclut quatre commits locaux supplémentaires par rapport à `origin/Dev-Tropics` au moment de la création.
- Vérification : les deux branches pointent initialement sur le même commit. Le dépôt source reste sur `Dev-Tropics` ; ses fichiers de travail sont préservés.

## 2026-11-06 — Installation du code dans le projet

- Création du worktree `/home/flo/Workspace/VarDyn_Tropics` sur la branche `VarDyn_Tropics`, au commit `1ff3053c`.
- Conservation de `context.md` et `journal.md` ; les fichiers suivis de la branche sont maintenant disponibles dans ce projet.
- Le dépôt principal `/home/flo/VarDyn` reste sur `Dev-Tropics` et peut utiliser une autre branche indépendamment de ce worktree.
- Les fichiers non suivis du dépôt principal ne sont pas copiés. Vérification de la branche active, du chemin du worktree et de la présence du code.

## 2026-11-06 — Revue de la branche par rapport à main

- **Périmètre :** `git diff main...HEAD`, ancêtre commun `6d32c84f`, commits
  `c89a32be` et `1ff3053c`, cinq fichiers. `main` pointe sur `350fa951` et compte
  63 commits propres ; aucune revue des changements exclusifs à `main`.
- **Vérification :** revue des conventions déléguée conformément à la compétence
  code-review ; aucune violation documentée relevée. Pas de spécification externe
  identifiée : vérification fonctionnelle fondée sur les intentions des commits.
- **Tests :** les deux modules ajoutés exécutés avec `unittest discover`,
  `PYTHONPATH=mapping`, `JAX_PLATFORMS=cpu` et Python de
  `/home/anaconda/envs/massh` : 12 tests réussis. Avertissements environnementaux
  NetCDF/CUDA sans échec des tests. `git diff --check main...HEAD` réussi.
- **Constats reproduits :** facteur auxiliaire nul ⇒ Q nul ⇒ coût de fond NaN
  avec `INV_4DVAR.prec=False` (défaut) ; carte incluant -180 et +180 convertie en
  0–360 ⇒ coordonnées dupliquées ⇒ `InvalidIndexError` à l'interpolation xarray.
- **Suite :** présenter ces deux problèmes et leurs localisations. Aucune
  modification du code source ni expérience scientifique effectuée.

## 2026-11-06 — Clarification de la comparaison avec main

- À la demande de l'utilisateur, vérification des états actuels via
  `git diff main HEAD` : 45 fichiers différents, 7 498 ajouts et 5 024 suppressions
  dans le sens main vers la branche courante.
- Les deux changements décrits précédemment apparaissent aussi sur main
  (`9589b724`, `a0dab7ca`). `git cherry main HEAD` confirme l'équivalence du
  correctif QG1L ; les deux modules de tests ajoutés sont identiques entre branches.
- La première revue utilisait l'ancêtre commun (`main...HEAD`) et ne couvre pas
  la totalité des différences actuelles : dynamique QGSW, bases, inversion,
  orchestration, état, diagnostics, SLURM et documentation notamment.
- Aucun code source modifié. Une revue complète des états actuels reste à faire.

## 2026-11-06 — Détail des différences bases, inversion et état

- **Méthode :** lecture des différences directes `main HEAD` pour `basis.py`,
  `inv.py`, `state.py` et des options associées dans `config_default.py`.
- **Bases :** la branche ne contient pas BASIS_BM ni le spinup GAUSS2D de main ;
  les opérateurs spatiaux gaussiens passent par des tableaux denses avant CSR,
  les facteurs temporels GAUSS3D/BMaux/HBC sont préassemblés et indexés par temps
  exact. BMaux utilise une sélection de bandes et de supports moins contrôlée,
  et n'applique pas le plafond à 1 du facteur auxiliaire présent sur main.
  Les fonds persistants et le partage des opérateurs entre bases restent présents.
- **Inversion :** SciPy et Optax découplé conservés, Optax entièrement compilé
  absent ; absence du repli sur zéro si le fichier d'initialisation manque et des
  conversions explicites du gradient vers le dtype du contrôle. Le calcul
  coût-gradient de Variational est identique dans cette comparaison.
- **État :** lecture/écriture NetCDF conservée ; support Zarr, compression,
  batching et collecte des sorties multi-modèles absents. Chargement de grille
  direct avec open_dataset plutôt que le helper de redémarrage de main.
  La conservation des tableaux JAX dans les copies reste présente.
- **Limite :** analyse statique descriptive ; aucun benchmark de mémoire, de
  vitesse ou de précision ajouté. Aucun code source modifié.

## 2026-11-06 — Rebase de VarDyn_Tropics sur main

- **Demande :** faire dériver la branche de main. Création préalable de
  `backup/VarDyn_Tropics-before-main-20261106` sur l'ancien HEAD `1ff3053c`.
- **Action :** `git rebase main`, terminé sans conflit. Git a ignoré `c89a32be`
  comme déjà appliqué et supprimé le rejeu de `1ff3053c` car son contenu est
  déjà présent en amont.
- **Résultat vérifié :** `HEAD` et `main` valent tous deux
  `350fa9516b2aab1cce28692700d356acc71cdd28` ; divergence `0 / 0` et comparaison
  des fichiers suivis vide. La branche active reste `VarDyn_Tropics`.
- **Préservation :** `AGENTS.md`, `context.md` et `journal.md`, non suivis,
  conservés ; contexte actualisé. Aucun push ni commit documentaire effectué.
- **Validation :** contrôles Git ; pas de relance des tests, le code étant
  exactement celui de main et aucune résolution manuelle n'ayant été nécessaire.

## 2026-11-06 — Préparation du spectre zonal–fréquence altimétrique tropical

- **Demande :** calculer un spectre comparable à l'image fournie, à partir des L3 de `/data1/data/obs/level3/Alti_l3_cmems_DT`, entre 6°S et 6°N.
- **Domaine validé :** proposition 140°E–80°W et période commune des missions acceptée par l'utilisateur (« ok »). Inventaire des noms de fichiers : plage commune 2023-01-16–2025-01-07 inclus ; huit missions (`al`, `c2`, `h2b`, `j3`, `s3a`, `s3b`, `s6a`, `SWOTnadir`). La continuité réelle et les lacunes restent à mesurer.
- **Préparation :** extraction régionale de `sla_unfiltered` et `sla_filtered` (conversion m → cm), coordonnées et temps exacts, avec manifestes par mission. Exclusion de `old_al` (archive dupliquée) et des fichiers préfixés `erroneous_`.
- **Exécution :** script autonome `runs/tropical_pacific_spectrum_20261106/extract.py`, lancé par `nohup`, journal et PID dans le même dossier. Aucun code source VarDyn modifié.
- **Décision méthodologique :** ne pas appliquer une FFT aux points irréguliers sans traitement de la couverture ; choix final de grille et de bande spectrale après diagnostic. Les fréquences élevées de l'image ne constituent pas une résolution garantie des L3 disponibles.
- **Vérifications :** branche `VarDyn_Tropics`, fichiers suivis initialement propres ; lecture des contextes locaux et de `/home/flo/MASSH/AGENTS.md`. En-têtes NetCDF : temps en jours depuis 1950, SLA en mètres, corrections DAC/marée/LWE déjà appliquées selon les métadonnées.

## 2026-11-06 — Extraction achevée et choix de la grille spectrale

- **Résultat :** extraction régionale des huit missions terminée ; aucune erreur dans `extract_fixed.log` et `extraction.json`. Deux syntaxes CF équivalentes des unités de temps ont été prises en charge ; l'extraction initiale rejetait à tort certains fichiers et a été entièrement remplacée. Les points hors intervalle temporel sont exclus d'après leurs temps effectifs.
- **Persistance :** les commandes `nohup` seules étaient arrêtées avec l'environnement d'exécution. Lancement réussi par unités utilisateur systemd exécutant `nohup`, sorties redirigées dans le dossier de l'analyse. Unité d'extraction corrigée `vardyn-spectrum-extract-fixed-20261106`, PID de lancement 3688002 ; diagnostic `vardyn-spectrum-coverage-20261106`.
- **Diagnostic :** `coverage_candidates.json` mesure 97,6085 % de cellules occupées pour 2 jours × 2° longitude × 4° latitude ; 84,0137 % en quotidien sur la même grille. Choix de cellules de 2 jours et de trois bandes de latitude, moyenne de la puissance spectrale entre bandes. Les rares trous seront interpolés localement, avec arrêt si leur distance au point observé le plus proche excède trois cellules.
- **Méthode :** moyenne par passage/jour/cellule pour limiter la surpondération des échantillons voisins à 1 Hz ; Welch 2D, fenêtres de 180 jours, recouvrement de 50 %, Hann en temps et longitude, retrait d'une tendance linéaire temporelle par longitude dans chaque fenêtre. Fréquences positives et nombres d'onde signés, k positif pour une propagation vers l'est.
- **Calcul lancé :** `analyze.py` par l'unité `vardyn-spectrum-analyze-20261106`, journal `analyze.log`. Sorties prévues : figures PNG/PDF et données NetCDF, contrôles de Parseval et du signe de propagation, sensibilités à une moyenne temporelle de six jours et au choix SLA filtrée/non filtrée. Courbes théoriques à c = 2,8 et 1,7 m/s, valeurs supposées et non ajustées.
- **Limites :** Nyquist nominal de 0,25 cycle/jour et 0,25 cycle/degré ; il ne démontre pas la résolution effective des observations ni l'absence de repliement spectral. Les fréquences supérieures à 0,1 cycle/jour seront signalées comme particulièrement sensibles à l'échantillonnage.

## 2026-11-06 UTC — Spectre calculé, figures et données validées

- **Achèvement :** extraction, diagnostic, analyse et vérification terminés. Calcul principal par l'unité `vardyn-spectrum-analyze-20261106`, PID de lancement 3690722 ; contrôle final `vardyn-spectrum-verify-20261106`. Journaux conservés, identifiants dans `jobs.json`. `analyze.log` se termine par `ANALYSIS COMPLETE` et `verification.json` confirme tous les contrôles.
- **Livrables :** `spectrum.png`/`.pdf`, vue symétrique et basses fréquences, Hovmöller/couverture, comparaison temporelle 2/6 jours ; `spectrum.nc` avec densités et champs intermédiaires ; `report.md` documentant intégralement méthode, références, limites, reproduction et hypothèses.
- **Période effective :** grille complète 2023-01-16–2025-01-06 inclus ; sept fenêtres de 180 jours décalées de 90 jours couvrant 2023-01-16–2025-01-04 inclus. L'inventaire nominal ne signifie pas continuité de chaque mission : HY-2B n'a plus de points régionaux après le 2024-12-20. Cette limite est explicitée dans le rapport.
- **Couverture :** 1 813 cellules manquantes sur 75 810, soit 2,3915 % ; distance maximale à une observation √2 cellule, bien sous le seuil d'arrêt de trois cellules. Aucun doublon temporel par mission. Extraction finale sans erreur.
- **Vérification :** Parseval 10,4737940027515 cm², égal à la variance pondérée après détrendage, confirmé après réouverture du NetCDF ; densités finies et non négatives, grille complétée finie. Ondes synthétiques vers l'est et l'ouest retrouvées au bon signe et au bon pic. Figures inspectées visuellement. Syntaxe Python de l'analyse vérifiée. Fichiers suivis du dépôt inchangés.
- **Résultats descriptifs :** lobe ouest maximal, dans la bande de recherche 20–60 jours et k entre −0,15 et −0,03 degré⁻¹, à 30 jours et k = −0,07857 degré⁻¹ (12,73°). Ne constitue pas une identification de mode. Puissance filtrée/non filtrée = 1,000112 dans la bande 20–90 jours et longueurs d'onde 20–140°. Spectres intégrés de moyennes 2/6 jours proches à basse fréquence.
- **Limites :** interpolation, moyennes spatiales et temporelles, repliement orbital et constellation variable ; aucune dé convolution ni validation des hautes fréquences. Courbes à c = 2,8/1,7 m/s supposées. IC ponctuel indicatif [0,529 P ; 2,554 P], environ 13,36 degrés de liberté temporels sous hypothèses stationnaires/gaussiennes ; aucune significativité d'onde revendiquée.
- **Mémoire :** contexte mis à jour avec le résultat et les limites ; scripts et données régionaux conservés (~274 Mo). Aucun commit ni push. Suites possibles : analyse des alias et sensibilité aux missions avant interprétation dynamique plus poussée.

## 2026-11-06 UTC — Extension demandée aux variables LWE et SLA − LWE

- **Demande :** mêmes spectres pour `lwe` et `sla_unfiltered - lwe` (nom de variable corrigé par rapport à la coquille du message).
- **Vérification du signe :** les métadonnées `lwe` des huit missions indiquent que la SLA du produit est déjà corrigée et que la SLA avant cette correction est `sla - lwe`. Application littérale de la soustraction demandée ; ce sens a été expliqué à l'utilisateur.
- **Méthode conservée :** mêmes manifestes, observations, domaine, grille, fenêtres, normalisation et courbes théoriques que le premier spectre. Extraction LWE en cm, alignement exact des temps avec le cache SLA d'origine ; arrêt si une valeur LWE manque sur ce support. Échelle de couleurs commune log10 = [0,5 ; 3,5].
- **Travaux :** script autonome `lwe_analysis.py`, sortie `lwe_comparison/`. Ajout d'un callback facultatif à `make_grid` dans le script d'analyse expérimental pour réutiliser strictement le calcul existant ; aucun changement dans les sources VarDyn. Syntaxe des deux scripts vérifiée.
- **Exécution :** unité utilisateur systemd `vardyn-spectrum-lwe-20261106`, exécutant `nohup` ; journal `lwe_analysis.log`. Vérifications prévues : support identique, linéarité de la soustraction au niveau des champs, Parseval pour les deux variables, réouverture du NetCDF.

## 2026-11-06 UTC — Spectres LWE et SLA − LWE terminés

- **Achèvement :** unité `vardyn-spectrum-lwe-20261106`, PID initial 3693099 ; journal terminé par `LWE ANALYSIS COMPLETE`. Extraire le LWE depuis les mêmes manifestes a permis de conserver le support exact de l'analyse SLA : aucune valeur manquante supplémentaire pour les huit missions.
- **Résultats :** variance spectrale après traitement 0,8049507557 cm² pour LWE, 12,4278726627 cm² pour SLA − LWE, contre 10,4737940028 cm² pour SLA. Co-spectre intégré SLA–LWE = −0,5745639521 cm² ; la puissance de la différence inclut ce terme croisé, elle n'est pas la différence des puissances.
- **Livrables :** dans `lwe_comparison/`, figures individuelles PNG/PDF, comparaison trois panneaux `comparison.png`/`.pdf` avec échelle log10 commune [0,5 ; 3,5], `spectra_lwe_comparison.nc`, `summary.json`, rapport, caches LWE alignés et inventaire d'extraction. Identifiants d'exécution ajoutés à `jobs.json` et `lwe_analysis.pid`.
- **Validation :** mêmes comptes de passages et masques de trous que le résultat initial ; linéarité `champ(SLA − LWE) = champ(SLA) − champ(LWE)` vérifiée à 10⁻¹¹ cm ; Parseval pour les deux variables à précision machine ; densités finies et non négatives, sauvegarde/réouverture NetCDF sans différence. Comparaison inspectée visuellement. Syntaxe des scripts vérifiée ; aucun code source VarDyn modifié.
- **Mémoire et limites :** contexte et rapport actualisés ; le signal avant correction LWE est bien obtenu avec le signe demandé. Les courbes restent illustratives et les hautes fréquences sensibles aux alias ; aucune identification d'onde ni significativité nouvelle revendiquée. Aucun commit ni push.

## 2026-11-06 UTC — Règle de langue des figures et traduction

- **Demande :** figures en anglais et inscription de cette règle dans le fichier d'instructions. Ajout dans le fichier existant `AGENTS.md` : tous les textes des figures (titres, axes, légendes, barres de couleurs, annotations et captions) doivent être en anglais. Compétence `writing-for-agents` appliquée pour une règle concise et explicite.
- **Travaux :** textes des figures traduits dans les scripts expérimentaux `analyze.py` et `lwe_analysis.py`, décimales anglaises incluses. Script `replot_english.py` pour régénérer les figures depuis les spectres NetCDF existants ; seule la courbe diagnostique à six jours, non sauvegardée initialement, est reproduite depuis les caches régionaux.
- **Exécution :** unité persistante `vardyn-spectrum-english-20261106`, PID initial 3693987, journal `replot_english.log`. Contrôle SHA256 des deux NetCDF prévu avant/après, afin de vérifier que seuls les livrables graphiques changent. Aucun code source VarDyn modifié.
- **Résultat vérifié :** sept PNG et quatre PDF régénérés en anglais ; journal terminé par `ENGLISH FIGURES COMPLETE`. Comparaison trois panneaux inspectée visuellement et texte du PDF extrait pour contrôle des libellés. Aucune chaîne française de figure restante dans les scripts. Les SHA256 des deux NetCDF sont identiques avant/après (`english_figures_verification.json`). Contexte actualisé.

## 2026-11-06 — Intégration de la référence scientifique VarDyn-SW

- **Source :** présentation fournie par l'utilisateur, https://docs.google.com/presentation/d/118r7YlfgA3VTehZW2Bu-kNpaTCNZqA5-Kn2LlNoFVc0/edit ; titre `VarDyn-SW`, 43 diapositives, révision `raKVVbff2fismA`. Lecture du texte natif de toutes les slides ; aucune édition de la présentation.
- **Cadrage :** documentation du modèle SW à une couche, des forçages divergent/rotationnel et de hauteur, des huit bandes spatiales et de la calibration GLORYS des statistiques et fenêtres temporelles.
- **Expériences rapportées :** apprentissage juillet 2020–juin 2021 ; OSE nadir 2024, AL exclue, quatre cibles de courants/SSH, pas de temps 900 s et optimisation de H, g′, Hwind et traînée. Contexte corrigé : des expériences tropicales existent déjà selon les slides ; leur provenance locale reste à établir.
- **Résultats :** meilleure RMSE AL brute rapportée à MLD + 50 m (2,4078 contre 2,4887 cm, gain 3,25 %), avec dégradation des courants U/V de 18,16 %/2,25 %. Résultats mean30 assortis de la non-convergence ; MAP1 SSH seule également non convergé. Distinction conservée entre classement SLA brute et cartes/spectres sur `sla_unfiltered - lwe`.
- **Validation et limites :** faits transcrits depuis les textes des slides, sans reproduction des expériences ni inspection détaillée des figures intégrées. Configurations, sorties originales, domaine AL complet et correspondance avec le code actuel non vérifiés. Branche `VarDyn_Tropics`, changements initiaux uniquement non suivis ; aucun code source modifié, aucun calcul lancé, aucun commit/push.
- **Suite :** utiliser cette présentation comme référence scientifique du projet ; retrouver les configurations et sorties avant de déduire les réglages d'une prochaine OSE. `context.md` actualisé, vocabulaire partagé `CONTEXT.md` préservé.

## 2026-11-06 — Recherche de la branche scientifique VarDyn-SW

- **Demande :** récupérer `VarDyn-SW`, identifiée par l'utilisateur comme branche des expériences des slides, puis analyser ses différences avec main.
- **Vérifications :** `git fetch origin` réussi avec accès aux métadonnées Git partagées ; `git ls-remote --heads origin` confirme seulement `Dev-Tropics`, `Dev_Sam`, `main` et `perf/gpu-4dvar` sur `datlas-ocean/VarDyn`. Aucune branche locale `VarDyn-SW` ; aucune référence correspondante dans les branches connues de `/home/flo/MASSH`, dans les reflogs VarDyn ou dans le dépôt `Workspace/MASSH_slurm`.
- **Blocage :** emplacement de `VarDyn-SW` inconnu ; question adressée à l'utilisateur pour obtenir le dépôt ou la machine source. Ne pas assimiler cette branche à `Dev-SW`, `Dev-Tropics` ou à un nom d'expérience.
- **Préservation :** worktree conservé sur `VarDyn_Tropics` ; aucun code source modifié. Les documents et sorties locales sont conservés. `main` local était à `350fa951`, distinct de `origin/main` (`0310d9a4`) ; préciser les deux références dans la future comparaison.
- **Suite :** récupérer la branche exacte une fois localisée, comparer les états directs à main et distinguer les changements propres à la branche depuis l'ancêtre commun. Contexte actualisé.

## 2026-11-06 — Dev-SW récupérée et comparée à main

- **Clarification :** l'utilisateur corrige le nom de branche en `Dev-SW`. Branche locale disponible à `29e20578ed7e20f24888f3e7c7e90d8828ee424b`, installée par `git switch Dev-SW` dans ce worktree ; documents et sorties non suivis préservés. `VarDyn_Tropics` conservée à `350fa951`. Dev-SW absente du distant configuré ; pas de version plus récente identifiée.
- **Périmètre :** comparaison directe avec main local `350fa951` (11 fichiers, +2 770/−345) et distinction du diff depuis l'ancêtre commun `e974805e` (+3 000/−279). 11 commits côté Dev-SW, 4 côté main ; deux correctifs Dev-SW équivalents déjà présents dans main. Rapport dans `runs/dev_sw_comparison_20261106/report.md`.
- **Analyse :** extensions des contrôles et cartes physiques SW, diffusion conservative, frontières caractéristiques x, bases SW journalières/calibrées, BMaux multivariée, covariance opérateur, fenêtres de forçage et optimisation par blocs, signatures de restart. Revue parallèle standards/cohérence scientifique selon la compétence code-review ; aucune violation documentée ni incompatibilité scientifique démontrée.
- **Blocage reproduit :** imports inconditionnels de modules absents de HEAD (covariance, bmaux_layout, sw_forcing_basis), plus deux modules des bases SW. Les cinq existent dans `/home/flo/VarDyn/mapping/src/` comme fichiers non suivis ; aucun copié. Les pilotes/configurations spécialisés non suivis empêchent aussi la reproduction depuis le seul commit.
- **Validation :** neuf fichiers Python du diff direct compilés en mémoire sans erreur ; 5 tests QG réussis. Tests gauss3d et restart bloqués à la collecte par les imports absents. Trois tests L4 existants de main, extraits dans runs et exécutés contre Dev-SW, échouent sur un gather vide : correctif main `b50bf09f` non repris. Journaux et JSON conservés.
- **Exécution :** tests CPU détachés sous nohup via unités utilisateur `vardyn-dev-sw-review-20261106` et `vardyn-dev-sw-review-l4-20261106`, identifiants d’invocation dans `jobs.json` (PID numérique indisponible) ; achèvement vérifié dans les logs et statuts des unités. Lancement nohup direct initial non persistant, remplacé par systemd.
- **Décision et suite :** contexte actualisé ; établir la provenance/version des modules et configurations manquants, puis traiter le correctif L4 dans une demande de modification dédiée. Aucun code source édité, aucune fusion/rebase, aucun calcul scientifique, commit ou push.

## 2026-11-09 — Ouverture de la revue Plannotator Dev-SW/main

- **Demande :** revue interactive du diff avec la compétence explicitement invoquée `plannotator-review`.
- **Périmètre :** worktree courant sur `Dev-SW`, comparaison à `main`, mode `branch`, références locales ; commande `plannotator review --base main --diff-type branch --no-git-remote-check --json /home/flo/Workspace/VarDyn_Tropics`.
- **Exécution :** session ouverte sur http://localhost:19432 après lancement hors sandbox, nécessaire à l'écoute du serveur local. Aucune autre session interrompue ; le premier message de port occupé provenait du lancement contraint dans le sandbox.
- **État :** attente des annotations ou de la décision de revue. Aucun code source modifié ; contexte scientifique inchangé.

## 2026-11-09 — Annotations Plannotator reçues et vérifiées

- **Décision retournée :** `annotated`, diff de branche contre main, répertoire VarDyn_Tropics. L'utilisateur a confirmé que l'interface fonctionnait et a réalisé sa revue. Session CLI terminée normalement.
- **Périmètre demandé :** open_y et frontières caractéristiques y ; limiter les nouveaux modules aux fonctions réellement partagées ; déplacer les helpers/constantes BMaux dans la classe ; retirer corrélations multivariées et minimiseur block-coordinate avec ses helpers ; expliquer l'absence de g_prime dans les cartes auxiliaires et les contrôles-log distincts.
- **Vérification ciblée :** support caractéristique limité à x confirmé (masques, configuration et calculs). Les constantes de quadrature sont utilisées seulement dans la base multivariée ; elles deviendraient inutiles si cette classe est retirée. Helpers de compatibilité legacy spécifiques à BMaux. Les appels locaux, métadonnées de fenêtres et paramètres de configuration devront être traités ensemble avec le retrait du minimiseur.
- **Questions physiques :** le chargeur de cartes ne lit effectivement pas g_prime, malgré son support dans le modèle. H et h_wind ont déjà un contrôle logarithmique dans main, avec planchers et plafonds ; les nouveaux coefficients ont une simple exponentielle multiplicative. Mutualiser le calcul est possible, uniformiser les formules modifierait les contraintes/prior et certains cas spéciaux H/c.
- **Ambiguïtés à discuter :** comprendre la suppression comme celle des corrélations croisées multivariées, sans supprimer par défaut la structure temporelle de chaque composante ; retirer le minimiseur ne prouve pas à lui seul que toutes les bases de forçage par fenêtres doivent disparaître.
- **État :** aucune modification du code, conformément à la consigne explicite du retour de revue demandant discussion et validation des constats avant édition. Résultats fondés sur lecture ciblée et comparaison à main ; pas de nouvelle expérience ni test exécuté. Prochaine étape : validation du périmètre puis mise en œuvre autorisée.

## 2026-11-09 — Base DailySWForcing intégrée dans basis.py

- **Autorisation et décision :** l'utilisateur conserve les autocorrélations temporelles et demande explicitement de placer DailySWForcing dans basis.py, en supprimant sw_daily_forcing.py.
- **Travaux :** migration depuis la copie historique de DailySWForcing, DailyAR1Covariance et du calcul des rayons spectraux vers `mapping/src/basis.py` ; dispatch direct. Initialisation des amplitudes RMS intégrée à la classe afin d'éviter une dépendance transitive via sw_learning_empirical vers le module retiré. Les helpers de géométrie/construction des atomes restent partagés avec la base calibrée. AR(1), interpolation, atomes, unités et signatures conservés.
- **Suppression :** sw_daily_forcing.py était déjà absent et non suivi dans ce worktree ; aucun fichier à supprimer ici, aucune référence restante dans les sources locales. Le fichier externe `/home/flo/VarDyn/mapping/src/sw_daily_forcing.py` est préservé avec son checkout indépendant.
- **Validation :** trois tests ajoutés dans `mapping/tests/test_daily_sw_basis.py` : covariance exponentielle/adjoint/inverse, dispatch/interpolation/adjoint de projection, amplitudes RMS physiques réelles. Tous passent. Comparaison à l'ancienne classe sur petite grille : égalité exacte des statistiques, géométrie, signatures/descripteurs, projections et racine de covariance, avec et sans initialisation RMS. Import du module retiré bloqué pendant les tests de la classe migrée. Syntaxe compilée en mémoire et git diff --check réussis.
- **Exécution :** CPU détaché sous nohup via unité `vardyn-daily-basis-move-20261109`, invocation `bc6650f22512442f803086ca3ebf066c`, PID et résultats dans `runs/daily_sw_basis_move_20261109/`. Marqueur VERIFICATION COMPLETE et JSON vérifiés.
- **Limite :** les dépendances partagées absentes ont été chargées pour validation depuis `/home/flo/VarDyn/mapping/src` en étendant uniquement le chemin du paquet dans le pilote de test ; le module basis testé est bien celui du worktree. Cela ne rend pas la branche autonome. Aucune autre annotation implémentée, aucune expérience scientifique relancée, aucun commit/push. Contexte actualisé.

## 2026-11-09 — Construction des opérateurs SW intégrée dans basis.py

- **Demande explicite :** ajouter la construction du forçage par opérateurs rotationnel/divergent dans basis.py.
- **Travaux :** déplacement des helpers de géométrie/coins/masques de grille C, signature de grille et construction CSR des atomes depuis la référence historique sw_forcing_calibrated vers des fonctions privées dans basis.py. Fenêtres cos² locales, gradient de φ, gradient tourné de ψ, hauteur compensée par bassin et moyenne de bassin conservés. Suppression de l'import des helpers externes ; le dispatch distinct de la base calibrée n'est pas modifié. Aucun fichier du checkout historique édité.
- **Vérification :** cinq tests ciblés réussis, dont AR(1), interpolation/adjoint, RMS physiques, divergence nulle des atomes rotationnels et conservation de masse des atomes compensés dans deux bassins séparés. Modules src.sw_daily_forcing et src.sw_forcing_calibrated interdits à l'import pendant les tests locaux. Équivalence exacte des matrices CSR (data/indices/indptr), records, géométries et signatures à l'ancienne construction pour deux normalisations, avec/sans côte et métriques non uniformes ; équivalence complète DailySWForcing conservée. Syntaxe et git diff --check réussis.
- **Exécution :** unité détachée CPU nohup `vardyn-sw-atom-move-20261109`, invocation `7e3468604fdd4489831f046897549762` ; PID, logs et vérification JSON dans `runs/sw_atom_construction_move_20261109/`, achèvement confirmé.
- **Limites :** autres dépendances partagées absentes chargées du répertoire historique dans le pilote de test uniquement ; aucune intégration scientifique ni nouvelle optimisation. Contexte actualisé ; aucun commit/push.

## 2026-11-09 — Autres remarques Plannotator implémentées

- **Autorisation :** demande explicite de modifier le code pour appliquer les autres remarques de revue. Autocorrélations temporelles conservées comme confirmé ; bases de forçage par fenêtres conservées indépendamment du minimiseur supprimé.
- **Frontières :** ajout open_y dans Masks et open_boundary_y dans configuration, wrapper et cœur SW ; caractéristiques entrantes sud/nord, flux sur faces humides, application dans intégrateurs avec/sans traceurs. Flux extérieur nul sur les faces fermées lorsqu'un autre axe est ouvert. Ancienne option flag_obc incompatible avec les frontières caractéristiques sur l'un ou l'autre axe.
- **Nettoyage :** retrait BMaux multivariée, import bmaux_layout et covariances croisées, constantes de quadrature inutilisées et configurations associées. Deux helpers legacy déplacés dans _Basis_bmaux. Suppression du minimiseur block-coordinate, de son dispatch/réglages, ses checkpoints, métadonnées de fenêtres, signatures multivariées et évaluateurs locaux (_scan_state_at/make_window_evaluator). Chemin scan global simplifié ; AR(1) et fenêtres temporelles maintenus.
- **Organisation :** bases SW générique et calibrée intégrées à basis.py avec leurs fonctions propres ; aucun nouveau module spécifique à une seule base. Création de covariance.py uniquement pour les interfaces/opérateurs indépendants réellement partagés par basis.py et inv.py. Les dépendances manquantes de la branche sont ainsi résolues pour les imports et tests locaux ; aucune source du checkout historique modifiée.
- **Paramètres physiques :** g_prime lu dans les cartes auxiliaires lorsqu'un nom de variable est fourni, utilisé par fermeture H/g_prime/c, conversion SSH, CFL, diagnostics et sauvegarde. Transformation log mutualisée pour H, Hwind, g_prime et coefficients dissipatifs, sans changement des planchers/plafonds ni des options spéciales de H/c.
- **Tests :** suite locale de 31 tests réussie (26,33 s), puis 2 tests supplémentaires ciblés réussis (7,65 s), sans overlay historique. Contrôles AR(1)/bases, masse/divergence, covariance, adjoints, gradients SW Python/scan/différences finies, reprise, rejet du mode retiré, masques et signes y, budgets de flux x/y/xy, carte g_prime, contrôles-log ; intégrations y/xy avec traceur constant et variance/adjoint de la base calibrée. Tests de forçage existants historiques réutilisés/adaptés aux modules locaux, sans conserver les assertions relatives aux signatures multivariées supprimées.
- **Exécution :** unités CPU détachées nohup vardyn-review-changes-20261109 et vardyn-review-extra-20261109, invocations f667c97e757c431d9ea38a3666bbd50a / 26724d4631624c20be0d351cbef172cd ; PID, logs, JSON dans runs/review_changes_20261109. Achèvement et codes de sortie vérifiés. py_compile mod.py/sw.py, compilation des sources et git diff --check réussis ; seuls les fichiers bytecode créés par ces contrôles ont été retirés.
- **Limites et suite :** pas d'OSE annuelle ni validation GPU ; le correctif L4 de main demeure hors périmètre des annotations appliquées. Vérifier le diff puis identifier les configurations de reproduction scientifique. Contexte actualisé, aucun commit/push.

## 2026-11-09 — Réplication des spectres dans l'Atlantique tropical

- **Demande :** même spectre dans l'Atlantique tropical ; interprétation conservant les trois variables du dernier résultat, `sla_unfiltered`, `lwe` et `sla_unfiltered - lwe`, 6°S–6°N, période et huit missions identiques au Pacifique, figures en anglais.
- **État initial :** branche `Dev-SW`, modifications locales de code et tests déjà présentes, lues dans le contexte/journal et préservées. Aucun fichier VarDyn ni résultat Pacifique modifié.
- **Domaine :** extraction large 60°W–16°E (bornes de cellules de 2°), puis sélection automatique de la plus longue bande de longitude contiguë avec >=90 % de cellules observées dans chaque bande de latitude et moyenne >=95 %. Ce critère est indépendant des amplitudes SLA/LWE ; les colonnes côtières insuffisamment observées sont écartées avant interpolation. Le domaine annoncé large n'est donc pas assimilé à une grille océanique intégralement observée.
- **Exécution :** scripts autonomes dans `runs/tropical_atlantic_spectrum_20261109/`. Extraction achevée (`ATLANTIC EXTRACTION COMPLETE`), unité `vardyn-atlantic-extract-20261109`, PID initial 3813341. Analyse lancée par `vardyn-atlantic-analyze-20261109`, journaux locaux. Même agrégation par passage/jour/cellule, trois bandes de latitude de 4°, Welch/Hann 180 jours avec 50 % de recouvrement et détrendage temporel linéaire ; réutilisation de la fonction spectrale expérimentale Pacifique sans la modifier.
- **Validation prévue :** support commun des trois variables, interpolation de petits trous uniquement (<5 % et distance <=3 cellules), linéarité des champs et signe est/ouest, Parseval, réouverture NetCDF, inspection des figures. Syntaxe des nouveaux scripts vérifiée.

## 2026-11-09 — Spectres atlantiques terminés et vérifiés

- **Domaine effectif :** 44°W–10°E, 6°S–6°N ; 27 colonnes de 2°. Sélection sur la couverture indépendante des amplitudes SLA/LWE, seuils annoncés conservés. Le critère est celui de bandes observées et non un masque de terre : certaines cellules côtières ne représentent que leur portion marine.
- **Couverture :** 97,4283 % des 29 241 cellules occupées ; 752 trous, distance maximale 1/1/√2 cellule selon la bande de latitude, interpolés par la même méthode locale. Extraction des huit missions sans échantillon SLA/LWE invalide ni doublon temporel.
- **Résultats :** variances spectrales après traitement SLA = 7,8500199829 cm², LWE = 0,8137642402 cm², SLA−LWE = 9,3213544127 cm² ; co-spectre intégré SLA–LWE −0,3287850948 cm². Domaine de 54° : Δk = 0,0185185 degré⁻¹, à distinguer de la résolution du spectre Pacifique sur 140°. Δf inchangé = 0,0055556 cycle/jour.
- **Validation :** Parseval sur les trois champs (erreur relative <=3,4×10⁻¹⁶), linéarité des champs complétés à 10⁻¹¹ cm, propagation synthétique est/ouest avec ±2/54 degré⁻¹ et 30 jours, densités finies/non négatives et réouverture NetCDF sans différence. Figures comparaison et couverture inspectées visuellement, textes anglais ; syntaxe des nouveaux scripts vérifiée.
- **Exécution :** unité d'analyse `vardyn-atlantic-analyze-20261109`, PID initial 3813921, invocation c74f9797685144cdbd382431f6f6d8b6 ; journal terminé par `ATLANTIC ANALYSIS COMPLETE`. Extraction PID 3813341, invocation 7fbc8b4237b6448a8cf8699bea3ab3d6. Métadonnées dans `jobs.json`.
- **Livrables :** comparaison normale et basses fréquences, trois figures individuelles, PNG/PDF, `spectra_atlantic.nc`, statistiques/inventaires, diagnostic de couverture et rapport détaillé dans `runs/tropical_atlantic_spectrum_20261109/`. Contexte actualisé. Les sources VarDyn, modifications locales existantes et résultats Pacifique sont préservés. Aucun commit/push.
- **Limites :** moyennes méridiennes partielles près des côtes, repliement orbital et hautes fréquences non validées, vitesses théoriques supposées ; la différence de largeur des bassins empêche d'interpréter directement les variances comme une comparaison physique contrôlée.

## 2026-11-09 — Correction des trois défauts relevés par la revue déléguée

- **Demande :** corriger les trois problèmes rapportés par le sous-agent, sans modifier les autres travaux locaux ni les résultats spectraux.
- **Vent :** g_prime_control transmis à _wind_depth_to_model dans jstep_core et jstep_core_trac ; lorsque phase_speed_squared_floor couple H/g_prime, le dénominateur du vent et son gradient utilisent la profondeur effective de la dynamique.
- **Calibration :** le modèle expose sa référence g_prime résolue dans State.qgsw_reference_g_prime (coordonnées de l'état) ; State.copy la conserve. La base calibrée compare reference_g_prime à cette référence avant repli sur le scalaire configuré, au lieu de comparer directement un MOD.g_prime éventuellement None. Une référence différente reste rejetée ; l'absence de modèle résolu donne une erreur explicite lorsque nécessaire.
- **Traceurs :** flux extérieurs en amont suivant le signe de la vitesse normale sur les quatre faces : donnée c_b sur l'entrée, cellule intérieure sur la sortie. Transmission des concentrations extérieures aux tendances de chaque étape d'intégration. Conservation des flux fermés et de l'extrapolation historique lorsque c_b est absent. Correction applicable aux différents schémas de flux et à plusieurs traceurs.
- **Validation :** quatre tests de régression ajoutés dans mapping/tests/test_sw_review_fixes.py : cohérence des deux chemins du vent et dérivée avec contrôle conjoint, bases calibrées avec g_prime diagnostiqué/cartographié et rejet des références différentes, entrées/sorties sur quatre faces et quatre schémas avec deux traceurs, intégration sans éponge et gradient JIT par rapport à c_b. Suite complète : 37 tests réussis en 38,94 s ; imports depuis ce worktree uniquement, aucune source historique chargée. Syntaxe, py_compile et git diff --check réussis.
- **Exécution :** CPU détaché nohup via unité vardyn-review-fixes-20261109, invocation f0b61aadfb524c22957bbfa1e3686d60 ; PID, log et JSON dans runs/review_fixes_20261109. Marqueur VERIFICATION COMPLETE et code zéro vérifiés. Seuls les fichiers bytecode générés par py_compile retirés.
- **Mémoire et limites :** contexte actualisé, expériences/sorties spectrales préservées ; pas de validation GPU ni OSE annuelle, aucun commit/push.

## Modèle d'entrée pour les prochaines avancées

### AAAA-MM-JJ — Titre

- **Objectif :** question scientifique ou technique traitée.
- **Travaux réalisés :** changements, préparation de données ou expériences exécutées.
- **Configuration et références :** version du code, paramètres, données, scripts ou commandes et chemins des sorties.
- **Résultats et validation :** observations, métriques et limites constatées.
- **Décisions :** choix retenus et raisons ; mettre à jour `context.md` si nécessaire.
- **Difficultés et points ouverts :** problèmes rencontrés et questions restantes.
- **Prochaines étapes :** actions à poursuivre.

## 2026-11-09 — Fusion Dev-SW dans VarDyn_Tropics et récupération de l’apprentissage SW (en cours)

- **Demande :** fusionner le code Dev-SW dans VarDyn_Tropics et garantir la présence des sources nécessaires à l’apprentissage des bases réduites SW.
- **Préservation :** état initial sur Dev-SW `29e20578` avec les corrections de revue et tests non committés. Ces 13 fichiers ont été enregistrés au commit `66c8aded` avant changement de branche ; documents, runs et checkout historique préservés. Fusion `--no-ff --no-commit Dev-SW` sur VarDyn_Tropics `350fa951`, sans conflit. Les quatre commits propres à Tropics sont conservés, dont la neutralité des L4 vides en scan.
- **Sources manquantes :** récupération de 16 modules SW depuis `/home/flo/VarDyn/mapping/src` ; imports de DailyAR1Covariance adaptés vers basis.py. Pas de réintroduction des trois anciens modules de bases regroupés ni de bmaux_layout/corrélations multivariées. Quatre pilotes CLI récupérés avec --config obligatoire et empreintes utilisant les modules actuels ; sept fichiers de tests historiques d’apprentissage repris, sauf le test d’un coordinateur non intégré.
- **Traçabilité :** empreintes originales/locales dans runs/merge_dev_sw_20261109/source_inventory.json. Guide versionnable docs/sw_learning.md. Contextes lus, dont CONTEXT_Dev-SW.md historique ; MASSH/AGENTS.md absent des deux checkouts inspectés.
- **Validation en cours :** compilation en mémoire, inventaire des imports relatifs, imports de tous les modules src depuis ce worktree et aide des CLI, puis suite pytest CPU détachée via nohup/systemd (unité vardyn-merge-sw-20261109, invocation adc0b24bcd464186962d6544b4068692, PID 3831104). Aucun entraînement annuel ni changement du checkout historique. Résultat final à consigner après achèvement.


## 2026-11-09 — Fusion SW validée et sources d’apprentissage complètes

- **Résultat :** fusion sans conflit de Dev-SW (`66c8aded`, incluant les corrections locales précédentes) sur VarDyn_Tropics, avec conservation des correctifs Tropics L4/QG1L/Gauss3d et des changements de messages de préparation. Les 16 modules SW manquants, quatre pilotes et huit fichiers de tests d’apprentissage sont intégrés. Les modules de bases restent réunis dans basis.py ; aucun retour des corrélations BMaux croisées ni du minimiseur par blocs.
- **Défaut découvert/corrigé à l’intégration :** sw_joint_optax passait groups=None au préconditionnement groupé en mode prior. Échec reproduit sur un problème gaussien ; correction par échelle identité et libellé prior cohérent. Les deux variantes prior/groupée retrouvent l’optimum analytique, reprennent les checkpoints et rejettent une signature incompatible. Le test de sélection d’un coordinateur historique absent n’a pas été repris ; le test de préparation CLI windowed utilise une configuration factice explicite.
- **Vérifications :** aucun import relatif manquant ; 29 modules src importés exclusivement depuis le worktree ; --help des quatre CLI réussi. Suite complète : **81 tests CPU réussis en 130,57 s**. Tests supplémentaires du pilote corrigé : **2 réussis en 2,03 s**. py_compile de 36 fichiers réussi avec cache isolé dans /tmp ; git diff --check réussi. Les avertissements concernent les précisions JAX, clip déprécié et la profondeur de vent des petits cas historiques, sans échec.
- **Exécution détachée :** suite unité vardyn-merge-sw-20261109, invocation adc0b24bcd464186962d6544b4068692, PID 3831104, VERIFICATION COMPLETE 0 ; test pilote initial invocation 329e775b9d2e4c9c88258b13211ca178 (1 échec reproduit, 1 succès), puis corrigé unité vardyn-merge-sw-extra-fixed-20261109, invocation e8636e263be5416f896dcfb77487652b, EXTRA VERIFICATION COMPLETE 0. Journaux, PID, XML et JSON dans runs/merge_dev_sw_20261109 ; empreintes originales/locales dans docs/sw_learning_source_inventory.json.
- **Documentation et limites :** guide docs/sw_learning.md ; context.md actualisé en distinguant ce contexte projet du vocabulaire CONTEXT.md. --config obligatoire dans les pilotes pour éviter des chemins de configurations absentes. Aucune configuration scientifique, donnée, OSE annuelle ni validation GPU revendiquée. Prochaine étape : rattacher configuration/données et préparer un pilote tropical. Les runs préexistants, AGENTS.md non suivi et fichiers du checkout historique sont préservés ; aucun push.

## 2026-11-09 — Alignement des fichiers src suivis sur main (validation en cours)

- **Demande :** ne suivre dans src que les fichiers présents sur main, déplacer tous les autres dans un autre dossier et vérifier qu’aucun import des scripts déplacés ne subsiste dans src. Périmètre appliqué au dossier mapping/src, objet des deux listes précédentes.
- **Référence :** main local `350fa951`, 13 fichiers. État initial propre pour les fichiers suivis, branche VarDyn_Tropics au merge `005ac566` ; AGENTS.md et runs préexistants non suivis préservés.
- **Déplacement :** 17 fichiers retirés de src et déplacés sans perte dans mapping/learning_local (ignoré via .gitignore). covariance.py est conservé comme copie locale ; les définitions indépendantes réellement nécessaires à basis.py/inv.py sont intégrées dans tools.py, déjà suivi par main. Les imports du cœur pointent exclusivement vers tools.py.
- **Compatibilité :** imports et empreintes des quatre pilotes CLI adaptés au paquet optionnel learning_local ; huit fichiers de tests d’apprentissage adaptés et ignorés si les sources locales sont absentes. Le test de covariance du cœur utilise src.tools. Guide et contexte actualisés ; provenance originale du merge conservée comme historique.
- **Contrôles :** index Git src déjà identique à l’inventaire main, les 17 fichiers du dossier local sont ignorés. Audit statique et tests CPU détachés en cours via nohup/systemd, unité vardyn-src-main-layout-20261109, invocation bae6c1b701d3488581399ecd5b65397e. Driver/PID/logs et inventaire des déplacements dans runs/src_main_layout_20261109. Les tests du cœur interdisent explicitement les imports des modules déplacés, puis l’apprentissage est testé séparément. Aucun calcul scientifique annuel ni push.

## 2026-11-09 — Organisation src identique à main et indépendance du cœur vérifiées

- **Résultat :** les fichiers suivis de mapping/src sont exactement les 13 fichiers de main local `350fa951`. Les 17 modules supplémentaires sont présents dans mapping/learning_local, non suivis et ignorés par Git. La demande porte sur l’inventaire des fichiers src ; les autres fichiers du dépôt ne sont pas retirés de Git.
- **Dépendances :** covariance.py était importé par basis.py et inv.py ; ses trois opérateurs partagés sont maintenant définis dans tools.py et importés depuis ce module établi. Les 16 modules d’apprentissage déplacés utilisent les bases et opérateurs du cœur dans le sens local → src. Aucun import src → modules déplacés, confirmé par AST et recherche textuelle, puis à l’exécution avec un bloqueur d’imports.
- **Tests :** 40 tests CPU du cœur réussis en 39,06 s avec learning_local, src.sw_* et src.covariance interdits ; huit fichiers optionnels ignorés pendant cette vérification. Processus séparé avec le paquet local disponible : 43 tests d’apprentissage réussis en 89,09 s (moyennes/variances/ACF, projections, adjoints/gradients SW, physique, MAP, checkpoints/reprises). py_compile de 34 fichiers réussi avec cache isolé /tmp ; --help des quatre pilotes réussi ; git diff --check réussi. Aucun test scientifique annuel ni GPU.
- **Traçabilité :** unité détachée vardyn-src-main-layout-20261109, invocation bae6c1b701d3488581399ecd5b65397e ; PID dans runs/src_main_layout_20261109/verify.pid. Marqueurs CORE VERIFICATION COMPLETE 0, LEARNING VERIFICATION COMPLETE 0 et LAYOUT VERIFICATION COMPLETE [0, 0] vérifiés ; rapports JSON/XML et déplacements conservés dans le même dossier.
- **Mémoire et suite :** guide docs/sw_learning.md et contexte actualisés ; l’inventaire de la fusion antérieure est explicitement historique. Les sources d’apprentissage restent utilisables dans ce worktree, mais ne sont plus incluses dans une copie Git neuve. Aucun fichier du checkout historique ni résultat d’expérience préexistant modifié ; aucun push.

## 2026-11-09 — README spécifique à VarDyn_Tropics

- **Demande :** décrire l’ensemble des fonctionnalités de la branche dans un README, en prenant exemple sur les travaux de /home/flo/VarDyn.
- **Sources consultées :** README précédent, contexte et journal du projet, dispatchs/configurations et interfaces de mod.py, basis.py, inv.py, state.py, observations, diagnostics et orchestration ; exemples WHIRLS/CSWM et SLURM. Dans le checkout historique : README tropical_sw et tropical_calibration, documentation des modèles, protocoles sw_annual_empirical_learning, sw_reduced_local_map2_20261007, sw_ose_miost_sla_glorys_currents_2024 et sw_submixed_consistent_bc_ose_20261015, ADR d’interface monocouche et fermeture H/g_prime/c. Les statuts de calcul datés ne sont pas repris comme statuts vivants.
- **Contenu :** README.md réécrit en français pour VarDyn_Tropics : architecture et statut de versionnement, modèles disponibles et limites de l’interface QGSW monocouche, conventions SSH/interface, vent, traceurs, frontières x/y, contrôles physiques et cartes, familles de bases et forçages, observations/diagnostics, trois minimiseurs (dont optax-full-gpu présent dans le dispatch), reprises et orchestration, chaîne locale MAP1/statistiques/MAP2 et EM, exemples scientifiques documentés, environnement et commandes, sorties et validation. Les protocoles historiques sont distingués des capacités présentes et des reproductions scientifiques non effectuées ici.
- **Précisions :** les 13 fichiers src restent suivis comme sur main ; les 17 modules locaux ignorés restent nécessaires aux seuls pilotes d’apprentissage. Le cœur est indépendant. Les dépendances additionnelles pyinterp/astropy/cartopy et Zarr sont explicitées sans annoncer une nouvelle installation testée. La couche mélangée dynamique et les rétroactions thermodynamiques des traceurs restent des évolutions.
- **Vérification :** neuf liens locaux valides, sept blocs de code vérifiés (AST Python, bash -n pour les commandes shell, clôture Markdown), inventaire src inchangé et git diff --check réussi. Les 40 tests du cœur et 43 tests locaux sont rapportés comme validation antérieure ; aucune suite relancée pour cette modification documentaire, aucun calcul scientifique/GPU. Aucun code source ni fichier du checkout historique modifié.
- **Mémoire :** context.md actualisé, ancienne mention du module covariance.py corrigée vers tools.py ; modifications documentaires laissées non committées. Aucun push. Suite scientifique inchangée : rattacher configuration/données et préparer un pilote tropical.

## 2026-11-09 — Scripts et tests conservés localement hors Git

- **Demande :** ignorer également scripts et tests dans Git ; périmètre mapping/scripts et mapping/tests, cohérent avec l’organisation de l’apprentissage SW.
- **Travaux :** ajout des deux exclusions ancrées dans .gitignore et retrait de l’index par git rm --cached, sans effacer les fichiers. Quatre pilotes et 17 fichiers de tests sont conservés localement. Les changements documentaires du tour précédent sont préservés ; README, guide SW et contexte actualisés pour ne plus présenter ces dossiers comme versionnés.
- **Validation :** git ls-files vide pour les deux dossiers ; les 21 fichiers existent, leurs SHA-256 sont identiques avant/après et ils sont tous ignorés par git check-ignore. Inventaire local dans runs/ignore_scripts_tests_20261109/files_before.json. Aucun code modifié ni test relancé pour ce changement de suivi ; aucune source historique ni résultat scientifique modifié. Aucun push.

## 2026-11-09 — Documents fusionnés, noms canoniques et lecture à chaque prompt

- **Demande :** fusionner context.md et CONTEXT.md, renommer journal.md en JOURNAL.MD et imposer la lecture d’AGENT.MD, CONTEXT.MD, README.MD et JOURNAL.MD à chaque nouveau prompt.
- **Réalisation :** les deux contextes sont réunis dans CONTEXT.MD : contexte Tropics, décisions, ressources scientifiques et intégralité du vocabulaire partagé conservés. Les anciens fichiers de contexte sont retirés. Journal renommé JOURNAL.MD sans altération de son contenu antérieur ; README renommé README.MD, liens et métadonnée readme de pyproject.toml adaptés.
- **Consignes :** AGENT.MD reprend les règles préexistantes de protection du code, expériences, mémoire et consignes conditionnelles ; la procédure impose de lire intégralement les quatre documents, dans l’ordre demandé, à chaque prompt, puis de vérifier les fichiers et l’état Git. Un court AGENTS.md sert de point d’entrée de découverte et renvoie vers cette procédure, sans dupliquer les règles. Compétence writing-for-agents appliquée.
- **Validation :** quatre noms canoniques présents, anciens doublons absents ; vocabulaire partagé identique après normalisation des en-têtes ; SHA-256 du journal identique avant/après renommage avant ajout de cette entrée ; liens du README, du contexte et du point d’entrée valides ; pyproject.toml analysé et cible README.MD vérifiée. Aucune modification du code scientifique ni relance de calcul. Les exclusions et retraits d’index de scripts/tests du tour précédent sont préservés ; aucun push.

## 2026-11-09 — Extensions Markdown minuscules et consignes dans AGENTS.md

- **Demande :** renommer les extensions .MD en .md et remplacer AGENT.MD par AGENTS.md.
- **Réalisation :** les quatre documents canoniques sont AGENTS.md, CONTEXT.md, README.md et JOURNAL.md. AGENTS.md contient désormais toutes les consignes précédemment dans AGENT.MD ; le court fichier de renvoi est remplacé et AGENT.MD est retiré. Le contexte fusionné et l’historique du journal sont conservés. La règle de lecture intégrale à chaque nouveau prompt utilise les nouveaux noms.
- **Références :** liens du contexte, du README, du guide SW et métadonnée readme de pyproject.toml actualisés ; mention du point d’entrée intermédiaire supprimée. Les noms historiques des anciennes entrées du journal et les consignes propres au checkout distinct VarDyn_IT sont conservés comme tels.
- **Vérification :** présence des quatre documents, absence des anciens noms à extension .MD à la racine, validité des liens locaux et de la cible README.md dans pyproject.toml ; git diff --check. Aucun code scientifique modifié ni calcul relancé. Changements non committés, retraits d’index et exclusions scripts/tests précédents préservés.

## 2026-11-09 — Outils locaux à la racine et inventaire mapping aligné sur main

- **Demande :** placer tests, scripts et learning_local à la racine du projet ; mapping doit comporter les mêmes fichiers que main.
- **État initial :** modifications documentaires et retraits d’index précédents présents, lus et préservés. Référence main locale `350fa951`, 43 fichiers sous mapping, dont trois tests. L’égalité de l’inventaire ne demande pas de revenir au contenu du code main.
- **Organisation :** scripts/ (quatre pilotes), learning_local/ (17 modules) et tests/ (14 fichiers complémentaires) déplacés à la racine, toujours ignorés par Git. Les trois tests de référence déjà présents sur main restent dans mapping/tests et leur suivi est rétabli, conformément à la nouvelle exigence d’inventaire complet. Aucun contenu de fichier sous mapping n’est modifié par ce déplacement. Inventaire et SHA-256 initiaux dans runs/root_local_layout_20261109/before.json.
- **Compatibilité :** racine calculée par parents[1] dans les pilotes ; ajout explicite de la racine et de mapping au chemin d’import, chemins de provenance et appels entre pilotes adaptés. Le test de préparation CLI retrouve automatiquement scripts depuis la racine. README, contexte et guide mis à jour ; les trois dossiers locaux restent exclus et le cœur reste indépendant de learning_local.
- **Vérification en cours :** égalité des inventaires mapping sur disque, dans l’index et sur main (43 fichiers) ; hash des contenus préservés. Test des --help depuis /tmp et suite CPU détachée via nohup/systemd, unité vardyn-root-layout-20261109, invocation dbf51b91695b4731bd77289edac4f744 ; logs, PID et résultats dans runs/root_local_layout_20261109. Aucun entraînement annuel ni push.

## 2026-11-09 — Déplacement à la racine validé

- **Résultat :** scripts/, learning_local/ et les tests complémentaires sont à la racine, ignorés par Git. mapping contient exactement les 43 fichiers de main locale (350fa951), y compris ses trois tests de référence suivis ; 14 fichiers de tests complémentaires sont conservés dans tests/. L’égalité est vérifiée sur disque et dans l’index ; les SHA-256 des 43 contenus sont inchangés par cette opération. Les modifications SW antérieures sont donc préservées.
- **Contrôles :** audit AST sans imports du cœur vers les outils locaux, imports src et learning_local depuis leurs nouveaux chemins, --help des quatre CLI depuis /tmp, compilation de 51 fichiers avec cache isolé /tmp. **40 tests CPU du cœur réussis en 35,78 s** (huit fichiers optionnels ignorés avec bloqueur d’imports), puis **43 tests d’apprentissage réussis en 82,75 s**. Git diff --check réussi.
- **Exécution :** unité vardyn-root-layout-20261109, invocation dbf51b91695b4731bd77289edac4f744, PID dans runs/root_local_layout_20261109/verify.pid ; CORE VERIFICATION COMPLETE 0, LEARNING VERIFICATION COMPLETE 0 et ROOT LAYOUT VERIFICATION COMPLETE [0, 0] vérifiés. Rapports JSON/XML et empreintes dans ce dossier.
- **Documentation :** README, contexte et guide adaptés aux nouveaux chemins ; le vrai dossier auxiliaire est aux_files. Pour toute la suite locale : PYTHONPATH incluant la racine et mapping, puis python -m pytest tests mapping/tests -q. Aucun entraînement annuel, aucune source du checkout historique modifiée, aucun push. Changements préparés, non committés.

## 2026-11-09 — Revue de VarDyn_Tropics contre origin/main

- **Périmètre :** comparaison demandée par `git diff 0310d9a4af268b792981c25062b0ce0068c356de`, incluant les changements préparés ; branche VarDyn_Tropics et modifications préexistantes vérifiées et préservées. Revue parallèle des bases/inversion et de la dynamique SW selon la compétence code-review ; intentions établies depuis le contexte, le README et l’historique, sans spécification externe supplémentaire.
- **Constats :** la sélection automatique du format BMaux historique dès qu’un chemin de fond est présent change la disposition des coefficients d’un calcul récent ; les helpers effectifs donnent sept bandes sans fond contre huit avec fond pour lmin=80, lmax=970, npsp=3.5, facpsp=1.5. Le nouveau flux de traceur entrant consomme aussi les zéros fabriqués par le wrapper lorsque ses données de frontière sont absentes : une SST uniforme à 20 descend à 19.989189 après 60 s avec une entrée à 0.1 m/s ; l’absence explicite de donnée dans le cœur conserve le traceur constant.
- **Vérification :** reproductions ciblées détachées avec nohup/scheduler, scripts et logs `/tmp/review_bmaux_legacy.py`, `/tmp/review_bmaux_legacy.log`, `/tmp/vardyn_spec_wrapper.py`, `/tmp/vardyn_spec_wrapper.log` ; marqueurs COMPLETE vérifiés. Invocation du cas traceur : `42777ca64cb24ae0b955bb338acec22a`. `git diff --check` contre la base réussi. Pas de nouvelle suite complète, d’expérience annuelle ou de validation GPU.
- **Suite :** signaler ces deux défauts avec leurs lignes ; aucune correction du code scientifique effectuée. Le contexte scientifique et les prochaines étapes restent inchangés ; aucun commit/push.
- **Confirmation BMaux complète :** avec le fichier auxiliaire livré `aux_reduced_basis_BM.nc`, une grille 4×4 et les temps [0, 1], la construction réelle passe de 1 836 contrôles (sept bandes) à 3 240 (huit bandes) lorsque seul INV.path_background est renseigné. Script, log et PID détaché dans `/tmp/review_bmaux_full.py`, `/tmp/review_bmaux_full.log` et `/tmp/review_bmaux_full.pid` ; achèvement COMPLETE et code zéro vérifiés.

## 2026-11-09 — Correction du point 2 : traceurs sans données extérieures (validation en cours)

- **Demande et périmètre :** corriger uniquement le point 2 de la revue ; conserver le point 1 BMaux en suspens. Branche VarDyn_Tropics, changements préparés antérieurs et journal de revue préservés. Aucun changement de basis.py.
- **Régression :** ajout du test local tests/test_sw_missing_tracer_bc.py sur le chemin réel Model_qgsw.step ; avant correction, SST uniforme à 20 devient 19.97838 après deux pas avec une entrée à 0.1 m/s. Échec exact enregistré dans red.xml (un échec, code 1), unité vardyn-tracer-bc-red-tests-20261109, invocation 952d6f21e3cd467480e4e5d6e8dc8fac ; PID dans red.pid. Le pilote utilise l’environnement massh et ajoute le chemin de pytest partagé, comme les validations précédentes.
- **Correction :** _apply_bc_tracer conserve les absences par des marqueurs NaN dans les tableaux de forme fixe. SW.step_with_tracer transforme ces marqueurs en masque avant l’arithmétique et la différentiation ; chaque étape utilise l’intérieur pour les valeurs manquantes, sans relaxation d’éponge. Les données présentes, y compris zéro, restent utilisées. Fonctionne aussi lorsque seuls certains traceurs disposent de données.
- **Contrôles en cours :** 16 cas nouveaux x/y, accès Python/tableaux préempilés et JIT/gradient, avec/sans éponge, toutes données absentes ou salinité explicitement nulle. Les tests SW de revue/forçage existants sont inclus. Un premier passage valide deux cas puis révèle un défaut du montage JIT du test (State.to_device manquant), corrigé dans le test ; aucune extension du périmètre scientifique. Unité vardyn-tracer-bc-verified-20261109, invocation 4a757d754bf840b4864c10d377f89c9e, logs/PID/XML dans runs/fix_missing_tracer_bc_20261109. Compilation py_compile initiale des deux sources et du test réussie avec cache isolé dans /tmp.

## 2026-11-09 — Correction du point 2 validée

- **Résultat :** 28 tests CPU réussis en 100.52 s : 16 cas de régression nouveaux et 12 tests SW existants. Les concentrations constantes sont conservées lorsque les données extérieures sont absentes ; une salinité explicitement nulle est appliquée indépendamment de l’absence de SST. Couverture x/y, éponge active/inactive, propagation normale et forçages préempilés compilés, gradients finis, advection sur quatre faces et quatre schémas, adjoints Python/scan et différences finies.
- **Traçabilité :** unité vardyn-tracer-bc-verified-20261109, invocation 4a757d754bf840b4864c10d377f89c9e ; PID 3898820. `verified.exit=0` et `VERIFIED VERIFICATION COMPLETE 0` vérifiés ; log/XML et pilote dans runs/fix_missing_tracer_bc_20261109. Les avertissements sont ceux de précision/dépréciation JAX et profondeur du vent déjà présents dans les petits cas.
- **Contrôles et mémoire :** compilation finale py_compile de mod.py, sw.py et du test avec cache isolé /tmp ; git diff --check réussi. Contexte et inventaire documentaire des tests locaux actualisés (15 fichiers complémentaires). Le nouveau test reste local et ignoré, conformément à l’organisation demandée du dépôt. Changements préexistants conservés.
- **Périmètre :** point 1 BMaux laissé en suspens et basis.py inchangé. Aucune expérience annuelle, validation GPU, commit ou push.

## 2026-11-09 — Préparation du commit demandé

- **Périmètre :** changements d’organisation et de documentation déjà préparés (documents canoniques, outils locaux à la racine et hors Git), puis correctif des traceurs sans données extérieures. Les suppressions de pilotes/tests dans mapping correspondent à leurs déplacements locaux déjà validés ; aucune suppression des copies locales ignorées.
- **Vérification :** diff et index relus sur VarDyn_Tropics ; résultat précédent des 28 tests CPU confirmé par verified.exit=0. Aucun changement supplémentaire de code ni nouvelle exécution scientifique. Les sources ignorées, tests locaux et runs restent hors du commit ; le point 1 BMaux reste en suspens.
- **Enregistrement :** commit local demandé par l’utilisateur ; aucun push demandé. Les validations détaillées restent consignées dans les entrées précédentes.

## 2026-11-09 — Cadrage de l’expérience annuelle tropicale 2024

- **Demande :** lancer une expérience d’un an en 2024 avec des bases SW apprises sur GLORYS ; compétence grill-me invoquée, redirigeant vers grilling.
- **État vérifié :** branche VarDyn_Tropics, HEAD 5e414dfc ; aucun changement suivi initial, runs/ non suivi. Documents du projet et consignes MASSH lus. Pilotes et modules locaux présents ; sources scientifiques inchangées.
- **Recherche :** exploration en lecture seule déléguée selon la compétence. Les documents historiques sw_ose_glorys_joint_2024.md et sw_ose_miost_sla_glorys_currents_2024.md dans /home/flo/VarDyn donnent un domaine 70°W–15°E, 20°S–20°N, grille 0,25° (341 × 161), du 2024-01-01 au 2025-01-01, avec pas SW 225 s. Ces réglages historiques ne sont pas encore choisis pour la nouvelle expérience ; disponibilité et compatibilité des calibrations à vérifier.
- **Suite :** préciser type d’expérience et domaine, puis calibration, observations et protocole numérique. La compétence demande explicitement de confirmer la compréhension partagée avant exécution ; aucun calcul lancé, aucun commit/push.

## 2026-11-09 — OSE 2024 et domaine confirmés ; calibrations retrouvées

- **Décisions utilisateur :** oui aux deux propositions : OSE annuelle 2024 avec AL réservée à la validation, Atlantique 70°W–15°E, 20°S–20°N à 0,25°.
- **Recherche déléguée achevée :** statistiques et paramètres MAP1 de surface présents dans /home/flo/VarDyn/outputs/sw_joint_drag20d_control_20261011/nadir_dt225_retry/map1_dt225_source ; configuration du 2020-07-01 au 2021-06-30, achèvement avec succès par variation du coût, sans convergence du gradient. Variantes mean30 et offset50 dans outputs/sw_submixed_chain_cfl01_20261014/{mean30,offset50}/map1. Présence vérifiée des statistiques, paramètres, configurations et marqueurs d’achèvement ; compatibilité avec le code actuel non testée.
- **Entrées retrouvées :** SLA MIOST et courants GLORYS filtrés à 10 jours, MDT MIOST et vent ERA5 2024 présents dans les sorties historiques. runs/README.md absent. Ces ressources restent en lecture seule.
- **Suite du questionnement :** réutilisation ou réapprentissage, cible de courant et transfert/réoptimisation de la physique. Aucun calcul ni changement de code source ; état Git conservé, documentation actualisée.

## 2026-11-09 — Cible surface confirmée, g′ fixé et dissipation explicitée

- **Décisions :** cible SSH + courants de surface confirmée ; g′ doit être fixé. Réutilisation des artefacts non encore acceptée : l’utilisateur demande leur contenu. Valeur uniforme ou carte apprise figée à distinguer.
- **Inspection déléguée :** local_statistics_after_map.npz contient géométrie/records de 71 848 atomes, moyenne, écart-type, tdec, largeur/espacement temporels et ACF (31 × 71 848), plus diagnostics. physical_parameters.npz contient les champs 161 × 341 H, g_prime_field, h_wind, bottom_drag_coef, c et les contrôles associés. Configuration de référence g′=0,025 m/s² ; carte apprise entre environ 0,00440 et 0,03994 m/s².
- **Dissipation vérifiée :** MAP1 surface fixe visc_coef=diff_coef=1000 m²/s ; ces coefficients ne figurent pas dans ses quatre contrôles. Certaines OSE historiques emploient 150 m²/s. Le cœur actuel applique visc_coef aux vitesses et diff_coef à l’épaisseur par diffusion conservative. Proposition : garder 1000 m²/s pour conserver la dynamique de calibration, à confirmer.
- **Limite technique :** retirer simplement g_prime des contrôles ne charge pas automatiquement sa carte apprise ; le pilote historique impose quatre contrôles. La préparation devra préserver explicitement la carte si elle est choisie, via une configuration compatible avec le cœur actuel. Aucun pilote ni source modifié, aucun calcul lancé.

## 2026-11-09 — Calibration, g′ et dissipation confirmés ; protocole OSE vérifié

- **Décisions :** réutiliser les statistiques/paramètres MAP1 surface, figer la carte g′ apprise, ajuster H/Hwind/traînée avec écarts-types du prior ×0,5 ; viscosité et diffusion fixées à 1000 m²/s chacune.
- **Ressources :** contrôle NVIDIA dans le sandbox impossible ; contrôle en lecture seule hors sandbox réussi, deux RTX 4090 de 24 Gio accessibles, GPU 0 environ 1,5 Gio utilisés, GPU 1 environ 2,3 Gio. Aucun calcul scientifique lancé.
- **Faits délégués :** configuration MIOST historique, observations C2N/H2B/J3N/S3A/S3B/S6A/SWON, AL exclue ; SLA brute par défaut et variante SLA−LWE disponibles. Erreurs spatiales par mission. SLA MIOST et courants directs GLORYS filtrés à 10 jours aux limites, état initial issu des frontières, MDT MIOST ajoutée une seule fois, vent ERA5 6 h. Trajectoire annuelle continue ; forçages physiques initialement nuls mais prior moyen appris conservé (contrôles blanchis −mean/std).
- **Numérique historique :** RK2 et pas effectif 225 s, CFL 0,024 ; optax-decoupled, maxiter 3000, ftol 5e-5 pendant 10 itérations, gtol désactivé, historique L-BFGS 10. Propositions à confirmer pour la nouvelle expérience ; pas de validation numérique avec la carte g′ figée à ce stade.
- **Suite :** terminer les choix scientifiques observations/frontières/trajectoire puis préparer la configuration et un préflight. L’orchestration et la configuration peuvent être adaptées hors cœur pour les trois contrôles physiques ; conventions de hauteur/références à vérifier explicitement. Sources inchangées ; mémoire actualisée.

## 2026-11-09 — Protocole confirmé, CFL 0,1 et lancement du pilote

- **Autorisation finale :** oui aux observations SLA brute, frontières MIOST/GLORYS filtrées 10 jours, ERA5 6 h et trajectoire annuelle continue ; correction utilisateur « clf=0.1 » comprise comme CFL=0,1. Pas effectif à vérifier ; Optax découplé, 3000 itérations au maximum, ftol 5e-5 sur 10 itérations, gtol désactivé.
- **Préparation :** configuration autonome et orchestration dans runs/ose_surface_2024_cfl01_20261109, utilisant uniquement le cœur de ce worktree et learning_local. Aucun fichier scientifique existant modifié. Le slot g_prime est conservé dans le modèle mais son contrôle MAP1 est constant dans decode et absent du vecteur optimisé ; références de hauteur/Fh historiques préservées. Les trois contrôles ajustables sont centrés sur les champs MAP1 avec prior ×0,5.
- **Vérifications prévues :** géométrie des 71 848 atomes, grille et champs physiques initiaux comparés aux artefacts, forçages physiques initialement nuls, observations présentes et AL exclue, coût/gradient finis et baisse du coût pendant deux itérations pilotes, trajectoire marine finie, épaisseur positive, g′ inchangé. Syntaxe des deux fichiers Python et bash -n réussis.
- **Exécution :** unité utilisateur vardyn-ose-surface-2024-cfl01-20261109, invocation 016ea770984248108a1b2c241feb7b6c, GPU 0 ; lanceur détaché avec enfants nohup et entrées/sorties redirigées. PID/logs dans le dossier du run. Pilote de trois jours sur la grille complète, puis annuel automatique uniquement si le pilote réussit. Statut final non encore vérifié ; aucun commit/push.

## 2026-11-09 — Pilote numérique réussi, correction de son export NetCDF

- **Résultats numériques :** grille 341 × 161, 71 848 atomes identiques au MAP1, 327 018 contrôles de forçage et trois blocs physiques de 835 contrôles, 66 922 observations. CFL=0,1 résout bien dt=900 s. Champs initiaux MAP1 identiques dans la tolérance float32, forçage physique nul. Deux itérations : coût 79 086,640625 → 55 745,703125 ; gradient fini (norme infinie finale 475,47). Arrêt volontaire au budget pilote, sans revendiquer convergence.
- **Incident d’export :** après validation des états marins, de l’épaisseur positive et du g′ constant, création NetCDF refusée par xarray car latitude/longitude sont des cartes 2D. L’annuel n’a pas démarré. Seul le nouvel export expérimental est corrigé pour nommer les dimensions y/x ; les sources du modèle et de l’apprentissage restent inchangées.
- **Récupération :** résultats numériques et ancien pilote conservés (run_before_export_fix.py). finish_pilot_export.py relit les états/contrôles, refait les contrôles finis/épaisseur/g′ et exporte sans répéter l’optimisation. Nouvelle unité vardyn-ose-surface-2024-cfl01-continue-20261109, invocation 2960a8cfa3ae4def87820e60998ed2c6 ; continue.sh lance ensuite l’annuel si cette clôture réussit. Logs/PID/exit distincts de la tentative initiale.

## 2026-11-09 — Pilote clôturé et OSE annuelle démarrée

- **Validation finale du pilote :** export NetCDF corrigé et réouvert avec SLA identique, états SSH/U/V tous finis en mer, épaisseur minimale 239,6364 m, g′ physique et contrôle figés identiques au MAP1. pilot/complete.json indique pilot_passed=true et optimization_converged=false (deux itérations seulement), avec empreintes de l’ancien pilote, du correctif d’export et de la récupération. Aucun calcul d’optimisation répété.
- **Annuel :** démarrage effectif après cette validation, PID 3930189, GPU 0. Unité vardyn-ose-surface-2024-cfl01-continue-20261109 vérifiée active/running ; configuration résolue et provenance annuelles écrites, dt=900 s confirmé, préparation des bases en cours à ce contrôle. Log annual/run.log, identifiants dans jobs.json. Le processus survit à la session ; aucun résultat annuel ni convergence revendiqué.
- **Mémoire et intégrité :** contexte et README du run actualisés ; code versionné du modèle et learning_local inchangés. Syntaxe Python/bourne des pilotes vérifiée. Aucun commit/push. Suite : suivre annual/run.log et les checkpoints, vérifier le statut de convergence et la trajectoire à l’achèvement, puis évaluer AL sur le support commun selon la convention SLA confirmée.

## 2026-11-09 — Rapport d’avancement : OSE annuelle achevée

- **Demande :** rapport d’avancement ; contrôle des journaux, fichiers finaux et état de l’unité détachée, sans relance du calcul.
- **Achèvement vérifié :** continue.log annonce Annual OSE completed, continue.exit=0 ; annual/complete.json porte optimization_converged=true ; unité inactive/dead, Result=success et ExecMainStatus=0. Le PID historique n’est plus un calcul actif.
- **Chiffres :** 6 862 517 observations, 9 727 554 contrôles, 499 itérations et 826 évaluations. Coût total 50 783 544 → 3 595 770,5 (environ −92,9 %), coût observations final 3 401 414,1875 et coût prior 194 356,3125. Optimisation 5038,6 s (84 min), exécution annuelle complète 5307,8 s (88,5 min).
- **Critère d’arrêt :** variation relative du coût sous ftol=5e-5 pendant 10 itérations consécutives ; gradient non convergé (norme infinie finale 45 796,55, seuil gtol désactivé). La convergence déclarée est donc celle du coût uniquement ; elle ne démontre pas une amélioration sur les données indépendantes.
- **Validation enregistrée par le calcul :** aucun état non fini en mer pour SSH/U/V sur l’année ; épaisseur totale minimale 112,84095 m ; carte g′ fixe vérifiée. Sorties présentes : daily_reconstruction.nc (~310 Mio), reconstruction.npz (~223 Mio), physical_parameters.npz (~933 Kio). La sauvegarde NetCDF inclut son contrôle de réouverture dans le pilote.
- **Mémoire et suite :** contexte et statut jobs.json actualisés ; aucun modèle/pilote relancé ni code scientifique modifié. AL reste exclue de l’assimilation et son score n’est pas encore calculé ; prochaine étape scientifique : validation AL et comparaison à MIOST sur un support commun, puis diagnostics SSH/courants. Aucun commit/push.

## 2026-11-09 — Diagnostics AL/MIOST selon le Datachallenge : deux conventions LWE

- **Demande :** compétence vardyn-explained-variance invoquée pour comparer MIOST, puis précision utilisateur : diagnostics comme 2026_DC_SSH_mapping_SWOT_OSE, deux cibles AL sla_unfiltered et variable décorrigée LWE. Interprétation confirmée par métadonnées : sla_unfiltered−lwe rétablit le signal avant correction. Reconstruction annuelle préservée.
- **Protocole consulté :** AGENTS/README/JOURNAL et notebook SSH MIOST du checkout Datachallenge ; CONTEXT.md absent. Fonctions natives mod_interp/mod_stat/mod_filter/mod_spectral utilisées en lecture seule : interpolation inverse-distance, cases de 1°, séries quotidiennes, régimes, filtre 65–200 km, segments spectraux 1000 km et fenêtres de 10°. Le domaine/période sont ceux de la reconstruction tropicale 2024, tronqués uniquement au support temporel commun, sans extrapolation.
- **Références :** cache régional MIOST non filtré issu des mêmes cartes quotidiennes que la configuration OSE ; contrôle direct prévu sur douze dates mensuelles. AL alg-L3 2024, points natifs, un poids par point. Même support fini pour les deux méthodes et les deux cibles. AL reste exclue du coût VarDyn ; MIOST contribue aux frontières. Les barres à quatre centrages de la compétence sont ajoutées sur un intérieur commun hors éponges.
- **Cohérence des cibles :** les fonctions statistiques Datachallenge soustraient leur entrée lwe alors que les spectres utilisent sla_unfiltered directement. Le pilote fournit donc explicitement chaque cible dans ce champ et une entrée lwe nulle, sans changer les modules ; les variables natives et corrections sont conservées dans les collocations. Cette normalisation évite double soustraction et références divergentes entre spectres et statistiques. Aucune marée interne ajoutée à cette comparaison SSH.
- **Exécution :** scripts autonomes diagnose.py/render.py/launch.sh dans runs/ose_surface_2024_cfl01_20261109/al_miost_datachallenge_20261109 ; syntaxe Python et bash -n réussis. Unité CPU détachée vardyn-al-miost-dc-20261109, invocation 4e31cf0c1c214325a79f6c0b9daf6377, enfant nohup avec logs/PID/exit. Aucun score final encore vérifié ; aucune source VarDyn ou Datachallenge modifiée, aucun commit/push.

## 2026-11-09 — Diagnostics Datachallenge AL/MIOST achevés et validés

- **Achèvement :** run.exit=0, status completed, validation.json passed ; unité vardyn-al-miost-dc-20261109 inactive avec Result=success et ExecMainStatus=0. Durée 300,1 s. Les deux références, deux méthodes, quatre statistiques, quatre spectres et les huit scores de variance expliquée par cible sont produits.
- **Support :** 366 fichiers AL, 889 869 points régionaux dans l’année ; 2363 au-delà du temps commun, 11221 sans prédictions communes finies ; 876 285 points natifs retenus du 2024-01-01 06:52:28 au 2024-12-30 22:36:53. Aucun manque SLA/LWE ni doublon temps/position. Les deux méthodes et deux cibles emploient exactement ce support. Intérieur hors éponges : 708 516 points, mêmes poids natifs ; pixels 0,25°/0,5° pondérés par leurs effectifs.
- **Contrôles indépendants :** 24 sondes de grille par reconstruction sans écart ; douze sondes mensuelles du cache MIOST contre les cartes sources sans écart. Reproduction de 366 517 collocations d’un précédent diagnostic MIOST archivé (hors ses frontières) : écart maximal 9,99e-17 m ; variables AL/LWE natives identiques. Contrôle séparé détaché vardyn-al-miost-dc-probe-20261109, invocation 844daa084cd24fbd9ce6c83189bac051, terminé avec succès ; JSON et empreinte conservés.
- **RMSE :** sla_unfiltered : VarDyn 2,4243388 cm, MIOST 2,4888957 cm, gain 2,5938 % ; biais −0,01978/−0,03528 cm. sla_unfiltered−lwe : 2,9671102 contre 3,0022672 cm, gain 1,1710 % ; biais −0,23763/−0,25313 cm.
- **Spectres :** différence médiane appariée de résolution VarDyn−MIOST +2,6217 km sur 2450 cellules natives, +3,4925 km sur 2448 cellules pour SLA−LWE ; plus fin pour VarDyn dans 942 et 895 cellules respectivement. À 200/300/500 km, les scores VarDyn sont légèrement inférieurs aux scores MIOST pour les deux cibles. Ne pas déduire le changement apparié de la différence des médianes régionales séparées, qui peut présenter un signe inverse. Le gain de RMSE ne constitue donc pas un gain spectral.
- **Variance expliquée :** quatre centrages de la compétence exécutés avec l’observation SSH explicite, sans ajout de marée interne. Variance globale centrée native : 18,7671/18,4082 cm² (VarDyn/MIOST), SLA−LWE : 19,1866/18,9789 ; centrage 0,25° : 15,0637/14,7592 et 15,2795/15,1370. Vérifications quadratiques et des variances locales pondérées réussies. Les valeurs non centrées sont des réductions de moyenne quadratique incluant les niveaux moyens.
- **Livrables :** REPORT.md, summary.json, al_scores.csv, regime_scores.csv, collocations communes et validation/provenance ; sous raw/ et minus_lwe/, cinq jeux de figures PNG/PDF (cartes, séries, résolution, spectres, régimes), stat/psd/interpolations NetCDF, puis barres/PDF/CSV aux quatre centrages dans explained_variance/. Les textes génériques IT du helper de compétence sont remplacés dans le rendu par des textes SSH exacts, sans changer ses scores. Figures comparatives, spatiales, spectrales et barres inspectées ; échelles, labels et unités anglais lisibles.
- **Intégrité et limites :** empreintes des trajectoires/entrées revérifiées inchangées ; sources VarDyn, learning_local et Datachallenge inchangées. Les figures DC principales conservent les éponges pour suivre le protocole ; leur intérieur est distingué dans le diagnostic de variance expliquée. AL exclue du coût, mais dépendance indirecte DUACS/MIOST et frontières MIOST conservées ; aucun test de significativité. Contexte actualisé, aucun commit/push.

## 2026-11-09 — Restitution avec les fonctions de tracé natives du Datachallenge

- **Précision utilisateur :** utiliser les mêmes diagnostics que dans le Datachallenge. Les statistiques/spectres déjà calculés proviennent de ses fonctions natives ; la nouvelle restitution appelle également directement mod_plot.py, comme les notebooks SSH MIOST et miost_nadir_vs_vardyn_nadir.
- **Fonctions retenues :** compare_stat_score_map_png (huit panneaux toutes échelles/65–200 km), compare_psd_score_png (résolution MIOST et différences absolues/relatives), plot_stat_score_map_png, plot_effective_resolution_png, plot_stat_score_timeseries, plot_stat_by_regimes, plot_psd_scores (PSD/bruit, rapport PSD, score spectral, cohérence), plot_diff_stat_by_regimes. Deux conventions AL conservées ; aucun nouveau calcul d’assimilation, interpolation ou spectres.
- **Rendu :** emprise tropicale et titres explicites ; limites de couleur natives conservées. Une étiquette native de signe sur la variance d’erreur est clarifiée sans changer les valeurs. Séries temporelles et tableaux HTML/CSV ; fenêtre spectrale statique centrée 25°W/0°N et neuf fenêtres de 10° sélectionnables dans le HTML. Les cartes globales de scores/résolution restent disponibles.
- **Exécution :** nouveau dossier al_miost_datachallenge_20261109/native_datachallenge, script render_native.py et lanceur nohup via unité vardyn-native-dc-plots-20261109, invocation 36bd9f92b45b4bd9848dcbd59c89b632. Vérification prévue des valeurs réellement tracées, des empreintes des NetCDF et du module natif ; aucune source Datachallenge/VarDyn modifiée. Rendu en cours, ancien diagnostic conservé.
- **Achèvement vérifié :** export final détaché vardyn-native-dc-exports-20261109, invocation 12ff98bc0a5048a4bbdd9947cbbbb1d1, terminé en 90,55 s, run.exit=0, unité inactive Result=success/ExecMainStatus=0. Rapport index.html et deux séries complètes disponibles. Les passes de présentation précédentes ont permis de corriger les anciennes propriétés Cartopy, l’aspect imposé et les limites logarithmiques HoloViews dans les exports Matplotlib ; aucun recalcul scientifique.
- **Contrôles finaux :** validation.json passed, 22 appels natifs documentés, 12 contrôles de valeurs cartographiques identiques et limites natives conservées ; empreintes actuelles des huit NetCDF, de mod_plot.py et du renderer revérifiées. Inspection visuelle des comparaisons pour les deux références, des cartes de résolution, séries et quatre panneaux spectraux ; axes spectraux logarithmiques et données visibles. CONTEXT.md et REPORT.md pointent vers cette restitution principale. Sources VarDyn/Datachallenge inchangées ; aucun commit/push.

## 2026-11-10 — Relance annuelle avec observations sla_unfiltered−lwe

- **Demande :** même expérience annuelle, variable observée sla_unfiltered−lwe. Domaine, dates, bases GLORYS, initialisation, physique, frontières et optimisation conservés ; AL exclue du coût.
- **Préparation :** nouveau dossier runs/ose_surface_2024_minus_lwe_cfl01_20261110, configuration autonome et copie du pilote expérimental validé ; chargeur natif configuré par name_var SSH=[sla_unfiltered,lwe], combine_var SSH=[1,−1]. Aucune source VarDyn/learning_local modifiée. Caches séparés ; pas de reprise des contrôles ajustés précédents.
- **Contrôles :** comparaison récursive des configurations : seules l’expression observée et les identifiants/chemins de sortie diffèrent. Empreintes du cœur et des modules locaux identiques à la provenance annuelle de référence. Syntaxe Python et bash -n réussis ; deux GPU libres à la vérification, GPU 0 retenu. comparison_to_reference.json conserve les contrôles.
- **Lancement :** unité persistante vardyn-ose-surface-2024-minus-lwe-20261110.service, invocation cbe39318bf824e7f842b744aab16d8ca, enfants nohup ; logs/PID/exit et jobs.json dans le nouveau dossier. Pilote complet trois jours/deux itérations, puis annuel conditionnel depuis l’initialisation d’origine. Achèvement non encore vérifié ; aucun résultat ni convergence revendiqué.
- **Suite :** contrôler les valeurs observées préparées, la validation du pilote et le démarrage annuel, puis le statut final et les diagnostics. Expérience initiale et diagnostics préservés ; branche VarDyn_Tropics, modifications documentaires antérieures conservées ; aucun commit/push.

- **Validation et démarrage effectif :** pilote terminé avec pilot_passed=true ; 66 922 observations et dt=900 s, coût 91 500,33594 → 65 313,89453 en deux itérations ; états marins finis, épaisseur minimale 238,59512 m, contrôle et champ g′ inchangés. Annuel automatiquement démarré, PID 4125850, unité active/running et lanceur PID 4123400 vérifiés. Aucun résultat annuel encore disponible.
- **Audit indépendant des observations :** valeurs SLA−LWE des caches comparées aux fichiers natifs, un lot pilote par mission (11 523 points au total) ; sept missions réussies, mêmes temps/positions/erreurs que le cache SLA brute. Les deux premières tentatives de l’audit limitaient la recherche au domaine nominal et manquaient les points de la marge native du chargeur ; audit corrigé en incluant cette marge, sans changement du calcul ni de ses caches. Validation finale observation_validation.json passed, log observation_validation3.log ; unité vardyn-ose-minus-lwe-observation-check3-20261110, invocation 6bf9d8d5f51a4729894836def442799d. Statuts dans jobs.json et contexte actualisés.

## 2026-11-10 — Rapport d’avancement SLA−LWE : arrêt non convergé

- **Contrôle :** fichiers finaux/logs et unité détachée relus ; unité failed, Result=exit-code, ExecMainStatus=1, launch.exit=1. Aucune nouvelle exécution ni modification scientifique.
- **Résultat :** 6 862 517 observations, 352 itérations acceptées, 629 évaluations ; coût 52 523 984 → 4 854 996,5 (réduction 90.757 %), dont observations 4 529 068,96875 et prior 325 927,53125. Gradient infini 315 860,78125 ; durée optimisation 3882,47194 s. Arrêt line search failed, ftol non atteint et gtol désactivé.
- **Reprises prévues :** première tentative 352 itérations/418 évaluations puis dix reprises avec historique L-BFGS réinitialisé, chacune sans nouvelle itération acceptée et 21 évaluations. Échec persistant, pas une convergence ni un simple épuisement de maxiter.
- **Livrables/limites :** checkpoint ~33 Mio, historique, résultat et log conservés. L’exception intervient avant validation/export de la trajectoire ; complete.json, daily_reconstruction.nc et paramètres physiques finaux absents. Aucun score AL/MIOST nouveau ; résultats de la référence SLA brute préservés.
- **Mémoire et suite :** PROGRESS.md créé, jobs.json/contexte actualisés ; diagnostiquer la recherche de pas au checkpoint avant choix d’une reprise. Branche VarDyn_Tropics, changements documentaires antérieurs conservés, aucun code édité, commit/push ou relancement.

## 2026-11-10 — Copie des skills VarDyn dans le projet

- **Demande et résultat :** copie intégrale de `vardyn-perf-ose-tropics` et `vardyn-progress-report` depuis `/home/flo/VarDyn/.agents/skills/` vers `.agents/skills/`, avec le script `build_map_progress_report.py`.
- **Validation :** inventaires identiques et contenus binaires identiques pour les trois fichiers ; empreintes SHA-256 calculées. Sources originales préservées ; aucune adaptation des chemins ni exécution des skills.
- **Mémoire :** disponibilité locale ajoutée au contexte ; branche `VarDyn_Tropics`, modifications documentaires préexistantes conservées. Aucun code scientifique modifié, aucun commit/push.

## 2026-11-10 — Préparation des diagnostics du skill tropical OSE

- **Cible confirmée par l’utilisateur :** OSE achevée du 9 novembre, `runs/ose_surface_2024_cfl01_20261109/annual` ; variante SLA−LWE non convergée exclue.
- **Inspection :** trois scripts historiques présents dans `/home/flo/VarDyn/mapping/scripts/`, modules DataChallenge et entrées AL/drifters/GLORYS/MIOST disponibles. Dates `dates` au lieu de `dates_ns` et SLA physique déjà validée dans `daily_reconstruction.nc` ; ne pas appliquer aveuglément leur conversion SSH scalaire. Figures historiques partiellement françaises.
- **Périmètre préparé :** comparaison 60°W–14°E, 15°S–15°N, 15 janvier–15 décembre 2024 ; animation 60°W–12°E pour les 366 jours. Entrées et sorties scientifiques précédentes préservées. Adaptation de copies des scripts pour la lecture des sorties et les textes anglais soumise à validation selon AGENTS.md ; aucun script source modifié à ce stade.

## 2026-11-10 — Diagnostics tropicaux et animation lancés

- **Autorisation explicite :** adaptation des copies locales des trois scripts pour les sorties du 9 novembre et les textes anglais. Copies et diffs dans `runs/ose_surface_2024_cfl01_20261109/performance_skill_20261110/` ; originaux et modèle préservés, empreintes conservées.
- **Adaptation :** SLA physique et vitesses de `annual/daily_reconstruction.nc`, sans nouvelle conversion g′ ; centrage des faces inchangé. Méthodes DataChallenge historiques, cibles AL SLA−LWE, dates/domaines du skill conservés. Compilation massh py_compile et bash -n réussis.
- **Exécution persistante :** unité diagnostics `vardyn-perf-diagnostics-20261110`, invocation 9668ab2090eb48f6a800c167b2cd485c ; aperçu `vardyn-perf-preview-20261110`, invocation cc64d40492bc4de999b74003084903ab, code zéro. Aperçu anglais inspecté, puis animation complète lancée par `vardyn-perf-video-20261110`. Nohup enfants, PID/logs/exit et jobs.json conservés.
- **État :** diagnostics et animation en cours, aucun nouveau score final vérifié. Sorties antérieures préservées ; suites : vérifier achèvement, appariement et scores, inspecter les cartes/spectres, ffprobe des 366 images et contrôle final des empreintes.

## 2026-11-10 — Export demandé du checkpoint annuel SLA−LWE non convergé

- **Demande :** écrire les sorties de cette expérience. Rejeu du dernier contrôle accepté à l’itération 352, sans optimisation ni modification du cœur ou des modules d’apprentissage. Échec de recherche de pas et non-convergence conservés explicitement.
- **Préparation :** pilote expérimental export_checkpoint.py reprenant la préparation et l’export du run initial, appel d’optimisation remplacé par lecture/vérification du checkpoint (signature, itérations, coût et finitude). Caches observations/opérateurs relus ; provenance annuelle originale vérifiée. Export des paramètres physiques, états journaliers NPZ et NetCDF SLA/U/V, contrôle des états marins, épaisseur positive, g′ fixe et réouverture NetCDF prévu.
- **Exécution :** nohup via unité vardyn-ose-minus-lwe-export-20261110.service, invocation 5586c28219b14c0095290aee51dcf5e3, GPU 0. Logs annual/export.log, PID annual/export.pid, retour export.exit et marqueur distinct annual/export_complete.json. Aucun complete.json de convergence créé. Syntaxes Python/bash validées ; export en cours.

- **Achèvement :** export.exit=0, unité inactive Result=success/ExecMainStatus=0 ; rejeu et export en 123,76 s. annual/export_complete.json distingue export_completed=true, optimization_converged=false et numerical_optimization_repeated=false.
- **Sorties vérifiées :** daily_reconstruction.nc (324 012 948 octets), reconstruction.npz (232 902 413 octets) et physical_parameters.npz (954 844 octets). NetCDF réouvert : 367 dates quotidiennes du 2024-01-01 au 2025-01-01, SLA physique en mètres, U/V sur faces C. Aucun état non fini en mer ; épaisseur totale minimale 97,14745 m, g′ fixe vérifié. Métadonnées de non-convergence présentes. Empreintes du checkpoint, historique, map_result et provenance initiale inchangées ; sources scientifiques inchangées. Contexte, README, PROGRESS et jobs.json actualisés ; aucun diagnostic supplémentaire ni commit/push.

## 2026-11-10 — Diagnostics du skill OSE et animation achevés et validés

- **Achèvement :** diagnostics.exit=0, preview.exit=0, video.exit=0, finish.exit=0 ; unités persistantes terminées avec succès. Animation : 366 images, 12 fps, durée 30,5 s vérifiées par ffprobe. Logs/PID/invocations dans performance_skill_20261110/jobs.json.
- **Résultats communs :** 580 548 AL sur 596 243 candidats et 38 805 drifters sur 249 112 candidats ; domaine/période du skill, AL SLA−LWE. RMSE VarDyn/MIOST : SLA 3,069164/3,111595 cm (gain 1,36364 %) ; u 11,677756/11,190908 cm/s (dégradation 4,35039 %) ; v 10,945962/11,118013 cm/s (gain 1,54750 %). Les courants ne couvrent que 15,58 % des candidats sur le support commun fini.
- **Spatiaux/spectraux :** 1581 cases SLA, 144 cases u et v, 1603 cases de résolution (minimum 30 observations/segments). VarDyn meilleur dans 949/1581 cases SLA, 69/144 pour u, 77/144 pour v. Résolution médiane appariée +2,932091 km, 657 cases améliorées et 946 dégradées ; ne pas soustraire les médianes séparées 328,531/329,837 km. Scores SLA à 200/300/500 km : VarDyn 0,075657/0,497749/0,783813, MIOST 0,072446/0,505872/0,799309. Spectres courants complexes u+iv avec sens rotatifs et vingt bandes de latitude valides.
- **Contrôles :** validation.json passed ; empreintes des trois entrées annuelles, originaux et scripts locaux, modules DataChallenge inchangés. Champs diagnostiques exactement égaux au NetCDF sur douze dates, appariement coordonnées/références et RMSE recomputées directement, mêmes segments et PSD de référence. Première validation échouée sur un chemin relatif depuis le cwd systemd ; contrôleur corrigé et relancé seul, journaux précédents conservés. Cartes, séries, spectres et aperçu inspectés ; marges des cartes de courant corrigées sans recalcul des statistiques. Compilation massh py_compile, bash -n et git diff --check réussis.
- **Livrables :** REPORT.md et index.html dans performance_skill_20261110 ; trois cartes spatiales, séries temporelles, cartes de variance, figures spectrales, NetCDF et JSON dans annual/diag_datachallenge_lat15 ; MP4, premier PNG et manifest dans annual/animations_glorys_miost_vardyn_2024. Rapport spectral natif conservé à la fin de rapport.md.
- **Limites et mémoire :** convergence de l’OSE sur coût uniquement ; AL exclue de son coût mais MIOST contribue aux frontières. Gains descriptifs, sans test de significativité. Les diagnostics antérieurs du domaine complet et les sorties de la variante SLA−LWE restent préservés ; aucun cœur/source original modifié, aucune nouvelle assimilation, aucun commit/push. CONTEXT.md actualisé.

## 2026-11-10 — Diagnostics du skill demandés pour l’expérience SLA−LWE

- **Cible :** `runs/ose_surface_2024_minus_lwe_cfl01_20261110/annual`, export du dernier contrôle accepté à l’itération 352, non convergé. `export_complete.json` vérifié : export réussi sans nouvelle optimisation, états marins finis, épaisseur minimale 97,14745 m, g′ fixe. Aucun marqueur de convergence fabriqué.
- **Préparation :** réutilisation des copies locales adaptées et déjà autorisées ; même lecture directe de SLA physique du NetCDF et centrage des vitesses. Animation lit le marqueur d’export, exige le contrôle de finitude existant et affiche non-converged. Rapport qualifié explicitement non convergé ; aucune source originale ni cœur modifié.
- **Périmètre :** méthodes et entrées du skill identiques à la référence du 9 novembre : AL SLA−LWE/drifters/MIOST, 60°W–14°E et 15°S–15°N du 15 janvier au 15 décembre ; animation 60°W–12°E sur 366 jours. Dossier séparé performance_skill_20261110 ; sorties précédentes préservées.
- **Exécution :** compilation massh py_compile et bash -n réussis. Unités persistantes diagnostics vardyn-minus-lwe-perf-diagnostics-20261110 (invocation cfb77bc8e9ba414b8184000b8deca491) et aperçu vardyn-minus-lwe-perf-preview-20261110 (5e000941e55b486da968e05155c6d72a), enfants nohup, logs/PID/exit. Empreintes des entrées annuelles et modules DataChallenge enregistrées. Calculs en cours ; aperçu à inspecter avant vidéo complète, puis validation finale des scores et artefacts.

- **Aperçu validé :** preview.exit=0 ; disposition, unités, limites communes et mention non-converged inspectées. Animation complète lancée sous unité vardyn-minus-lwe-perf-video-20261110 ; validation finale programmée sous vardyn-minus-lwe-perf-validation-20261110, après diagnostics et vidéo. Aucun résultat final encore revendiqué.

## 2026-11-10 — Diagnostics SLA−LWE non convergée achevés et validés

- **Achèvement :** diagnostics.exit=0, preview.exit=0, video.exit=0, finish.exit=0 ; unités de calcul et finition réussies. Animation complète de 366 images à 12 fps, durée 30,5 s vérifiée par ffprobe, statut non-converged visible. Jobs et invocations dans performance_skill_20261110/jobs.json ; aucun nouveau calcul d’optimisation.
- **RMSE sur support commun :** 580 548 AL et 38 805 drifters (sur 596 243 et 249 112 candidats). VarDyn/MIOST : SLA−LWE 3,172408/3,111595 cm (+1,9544 % d’erreur), u 11,573694/11,190908 cm/s (+3,4205 %), v 11,882630/11,118013 cm/s (+6,8773 %). Résultats du checkpoint accepté à l’itération 352, non convergé après échec de recherche de pas.
- **Spatiaux/spectres :** 1581 cases SLA et 144 par courant ; VarDyn meilleur dans 519 cases SLA, 68 pour u, 43 pour v. Sur 1603 cases spectrales appariées, résolution VarDyn−MIOST médiane +11,962765 km, 294 améliorations/1309 dégradations. Scores SLA à 200/300/500 km : 0,069672/0,474394/0,760963 contre MIOST 0,072446/0,505872/0,799309. Spectres des courants u+iv par sens rotatif, vingt bandes de latitude valides.
- **Comparaison vérifiée au 9 novembre :** égalité exacte des dates, coordonnées, références AL et drifters/identifiants, donc même support. RMSE de cette variante par rapport à l’assimilation SLA brute : AL +3,3639 %, u −0,8911 %, v +8,5572 %. comparison_with_raw_assimilation.json conserve valeurs et statuts de convergence distincts.
- **Validation :** validation.json passed ; empreintes des entrées annuelles et du checkpoint/historique/résultat/provenance d’export inchangées, originaux/modules DataChallenge inchangés, champs exacts au NetCDF sur douze dates, RMSE recomputées et segments/PSD référence identiques. Première validation bloquée par np.isnan sur sensor_id textuel dans le contrôle inter-expériences ; comparaison exacte corrigée et contrôle seul relancé, logs conservés. Marges des cartes corrigées par export bbox tight, sans recalcul des statistiques ni de la vidéo. Figures anglaises/aperçu/spectres inspectés ; py_compile massh, bash -n et git diff --check réussis.
- **Livrables et mémoire :** REPORT.md et index.html dans performance_skill_20261110 ; trois cartes spatiales, cartes de variance, RMSE quotidiennes, spectres/NetCDF/JSON et rapports natifs dans annual/diag_datachallenge_lat15 ; MP4/aperçu/manifest sous annual/animations_glorys_miost_vardyn_2024. Rapport natif qualifié non convergé après la section spectrale ; aucun complete.json de convergence créé. CONTEXT.md actualisé.
- **Limites :** courants évalués sur 15,58 % des candidats, frontières dépendantes de MIOST, aucun test de significativité ; non-convergence maintenue. Aucun cœur ou script original modifié, aucune nouvelle assimilation, données de référence et résultats du 9 novembre préservés ; aucun commit/push.

## 2026-11-10 — Lecture du mode LWE MIOST fourni

- **Demande :** résumer `default.py` et `lwe.py` avant discussion de l’implémentation dans VarDyn. Lecture des documents canoniques et des consignes MASSH retrouvées dans `/home/flo/MASSH/AGENTS.md` ; branche VarDyn_Tropics, modifications documentaires et fichiers non suivis préexistants préservés.
- **Constats :** composante SLA par mission ajoutée à l’inversion commune PCG ; fenêtres le long des traces, segmentation aux trous de 600 s, segments d’au moins sept points, centres espacés de tdec_lw/facnlt. Défauts 0,001 jour et 4, soit échelle 86,4 s et espacement 21,6 s. Variance des coefficients facQ × std_lwe² issue des cartes auxiliaires ; std_lw=0.02 transmis mais non utilisé explicitement par la classe LWE fournie. Polynômes transversaux d’ordres 0/1/2 possibles.
- **Validation et limites :** analyse statique intégrale des deux fichiers avec lignes repérées ; aucune exécution ni test scientifique nécessaire. Window, Component, observation_index et solve_pcg importés restent hors de ce périmètre : forme exacte des fenêtres et signe des corrections exportées non établis. Contexte complété ; aucun code VarDyn/MIOST modifié, aucun calcul, commit ou push.

## 2026-11-10 — Mode LWE précisé depuis le code MIOST complet

- **Source fournie :** /home/flo/miost/miost_core/miost_core ; branche g_fragmentation et nombreux changements locaux préexistants, consultation seule. default.py identique à la copie fournie ; lwe.py fourni ajoute le bornage des latitudes, un argument inutilisé return_index et l’alias OrbitError.
- **Vérifications :** Window(None) utilise cos²(pi*u/2), nulle hors [-1,1] ; tdec_lw est une demi-largeur (86,4 s), support complet 172,8 s et centres espacés de 21,6 s. solve_pcg résout (Q^-1 + GᵀR^-1G)eta=GᵀR^-1y avec préconditionnement sqrt(Q) ; bruit R diagonal.
- **Signe et sorties :** exemple examples/miost_valid_60_lwe.py : entrée sla_unfiltered−lwe et comparaison de −e_MIOST à la correction DUACS. ScatterOutput exporte directement e_MIOST sans inversion de signe. Les sorties sont séparées par composante et filtrables via only_components ; sélectionner explicitement les composantes océaniques pour les cartes physiques.
- **Portée :** analyse statique des fenêtres, sélection des observations, solveur, projections et exemple ; aucun calcul scientifique ni code modifié. Contexte actualisé ; intégration VarDyn encore à discuter, aucun commit/push.

## 2026-11-10 — Plan d’intégration LWE dans VarDyn

- **Demande :** établir un plan ; aucune implémentation ni expérience autorisée par cette seule demande. Documents du projet relus, branche VarDyn_Tropics et changements documentaires préexistants vérifiés.
- **Interfaces examinées :** inv.py (Inv_4Dvar, Variational, adjoints Python/scan, contrôle et Xres), obs.py (temps conservés dans les observations préparées), obsop.py (concaténation missions, caches et résidus normalisés), config_default.py ; pilote annuel et sw_learning_joint_forward/sw_joint_optax. Cartes std_lwe et facQ présentes dans /home/flo/miost/miost_sad/miost_sad, correspondance des alias à expliciter.
- **Proposition :** contrôle LWE blanchi propre aux observations, ajouté après conversion SSH ; géométrie aux temps natifs et prior indépendant. Première version nadir p=0 avec fenêtres/échelles/prior MIOST, sans nouvelle dépendance runtime MIOST et sans nouveau module src. Conserver les métadonnées par ligne et versionner les caches ; raccordement explicite aux gradients du cœur et à l’autodiff du pilote annuel.
- **Validation prévue :** parité des projections et variances MIOST, tests transposés/gradients, cas multimis­sion/vides/bords/doublons, optimum gaussien analytique, régressions désactivé, reprise/rejeu et cohérence des deux chemins ; pilote puis comparaison trois traitements de la correction avec AL exclue et diagnostics SSH/courants communs.
- **Statut :** propositions documentées dans le contexte et présentées à l’utilisateur ; aucun fichier source édité, aucun calcul ni test exécuté, aucun commit/push. Les détails de configuration et le traitement des observations agrégées seront arrêtés pendant l’implémentation.

## 2026-11-10 — Étape 1 LWE : cadrage avec grill-me/grilling

- **Demande :** implémenter l’étape 1 (configuration et conventions) en invoquant grill-me, qui renvoie à grilling. La compétence impose de confirmer la compréhension partagée avant mise en œuvre ; deux choix sont soumis : conventions explicites avant/après correction DUACS, paramètres MIOST surchargeables par mission.
- **Exploration déléguée en lecture seule :** configuration dans config_default.py, validation après fusion dans Exp (exp.py), avant effets de bord. Les clés inconnues sont ignorées par la fusion ; les exceptions de merge_configs sont absorbées. Bloc LWE par source nadir recommandé. Transformation des variables déjà assurée par name_var/combine_var : la convention doit décrire l’entrée sans seconde soustraction.
- **Cartes vérifiées :** C2N→C2, J3N→J3 ; S6A/SWON réutilisent J3 par alias MIOST documentés. H2B/S3A/S3B/AL ont leurs cartes ; facQ commun disponible. Ces alias ne prouvent pas une calibration propre à S6A/SWON.
- **Statut :** branche et changements préexistants préservés ; aucune source éditée, aucun test/calcul lancé. Questions de conception en attente ; autorisation d’implémenter l’étape 1 reçue, confirmation de compréhension demandée conformément à la compétence invoquée.

## 2026-11-10 — Étape 1 LWE MIOST implémentée et validée

- **Décisions utilisateur :** « Seulement avant correction DUACS » et « comme dans miost ». Confirmation du périmètre de configuration/conventions ; aucune estimation LWE ni nouvelle expérience demandée. Paramètres nommés comme MIOST, valeurs par défaut identiques et portées par source.
- **Implémentation :** `config_default.py` déclare LWE_MIOST et la clé facultative lwe dans OBS_SSH_NADIR. `exp.py` résout les valeurs et valide types, paramètres, convention unique, expression SLA−LWE, labels explicites et existence des cartes. Validation avant le merger historique qui absorbe les exceptions, et avant création/copie de fichiers ; nettoyage du module après erreur. Aucun ajout automatique de correction, aucune dépendance runtime MIOST ni nouveau module src.
- **Limite explicite :** `obsop.py` refuse par NotImplementedError une exécution avec LWE activé jusqu’au raccordement de la projection et de l’inversion. Anciennes configurations et appels sans bloc OBS conservés. Aucun noyau de modèle, opérateur scientifique ou minimiseur modifié.
- **Documentation/tests :** README décrit l’usage, les unités, le signe, le prior facQ×std_lwe², les alias de cartes et le statut configuration seule. Nouveau test local `tests/test_lwe_config.py` (ignoré, conformément à l’organisation du projet) : 23 cas, dont fusion Exp mono/multi-source, erreurs avant effets de bord, invariance des expressions/defaults, compatibilité et garde d’exécution.
- **Validation :** première passe 22 tests réussis ; dernière passe 38 tests réussis en 2,86 s (23 nouveaux et 15 tests de référence mapping/tests), verify.exit=0. Avertissement binaire NumPy/NetCDF dans un test existant, aucun échec. Syntaxe AST de quatre fichiers et résolution des cartes alg/c2n/h2b/j3n/s3a/s3b/s6a/swon vérifiées ; les alias J3 de S6A/SWON sont documentés, contenu des cartes non évalué.
- **Exécution détachée :** nohup sous unités vardyn-lwe-config-20261110 (invocation 1c656917b4514697918af48c956a12c5) puis vardyn-lwe-config-final-20261110 (8308653291fd4b1c9d2a6c87b7e91a0e). Dernière unité inactive/dead, Result=success, ExecMainStatus=0. Logs/PID/exit/XML initiaux et finaux, pilote et validation.json dans runs/lwe_config_20261110. Accès au bus utilisateur refusé dans le sandbox ; exécution autorisée via escalation pour respecter la persistance demandée.
- **Mémoire et suite :** contexte actualisé ; étape 2 (temps natifs/mission/segment et caches) reste à implémenter, puis opérateur et coûts. Branche VarDyn_Tropics, changements documentaires et fichiers non suivis antérieurs préservés ; aucun commit/push, aucune assimilation ni calcul scientifique.


## 2026-11-10 — Répartition de la configuration LWE entre observations et opérateur

Demande explicite de ne pas modifier `exp` et de placer les paramètres LWE
propres à chaque mission dans les observations `OBS_SSH_NADIR`, les paramètres
communs dans le nouvel opérateur `OBSOP_INTERP_LWE_L3`.
Retrait des seuls ajouts de l’étape précédente dans `mapping/src/exp.py` :
le fichier correspond exactement à HEAD, y compris son absence de saut de
ligne final. Le chargeur historique est utilisé tel quel.

Suppression du bloc `LWE_MIOST` et du sous-bloc `lwe`. Ajout aux observations
de `lwe_enabled`, `lwe_mission`, `path_lwe`, `name_var_lwe`.
Déclaration du nouvel opérateur dans config_default.py, reprenant L3 et les
paramètres MIOST communs (fenêtres, segmentation, convention, carte facQ).
Validation et résolution des cartes dans `Obsop_interp_lwe_l3.resolve_config`,
sans mutation de la configuration. Sélection par `name_obs` pour qu’une
mission LWE ne bloque pas un opérateur ordinaire visant une autre mission.
Maintien du refus explicite d’exécuter LWE avant les étapes de projection et
de coût conjoint ; pas de soustraction supplémentaire de la correction DUACS.
README et CONTEXT actualisés pour ce contrat.

Vérification CPU détachée par systemd/nohup : unité
`vardyn-lwe-layout-20261110.service`, invocation
`92b9aea3f85640319e34b4cd7f6c3233`. PID et journaux conservés dans
`runs/lwe_config_layout_20261110/`. **39 tests réussis** (24 de configuration,
15 de référence), `verify.exit=0`. Vérifications : fusion des deux blocs avec
Exp inchangé, valeurs MIOST, convention d’entrée, chemins manquants, cartes
par mission, sélection et dispatch, refus explicite du coût non implémenté.
Un avertissement NumPy de compatibilité binaire dans le test NetCDF, sans
échec. `git diff --check` propre ; identité binaire de exp.py avec HEAD
contrôlée. Aucun calcul d’assimilation, commit ou nouvelle étape du portage.
La géométrie native, la projection, le prior et les sorties LWE restent à faire.


## 2026-11-10 — Revue Plannotator de la configuration LWE

Compétence plannotator-review invoquée explicitement. Revue du checkout
VarDyn_Tropics lancée puis limitée au mode unstaged, session locale port 19443.
Le lancement initial sur 19432 présentait un périmètre plus large ; session
fermée. Ouverture automatique du navigateur impossible (affichage localhost:11.0
indisponible), lien local fourni. Revue finalement soumise et commande terminée
avec code 0 ; retour conservé dans /tmp/vardyn-lwe-plannotator-review.log.

Quatre annotations examinées uniquement sur les lignes concernées de
config_default.py et leur consommation dans obsop.py : supprimer lwe_mission
et la déduction des noms de cartes au profit de chemins explicites ; supprimer
input_convention (convention unique déjà retenue) ; supprimer window (cos²
unique) ; déplacer sad_path dans OBS_SSH_NADIR, car mission spécifique.
Constats confirmés comme corrections de conception des ajouts courants,
aucun défaut préexistant invoqué. Le retour demande explicitement de discuter
et valider les verdicts avant modification du code : corrections présentées,
sources, tests et configuration inchangés pendant cette revue. CONTEXT conserve
le contrat implémenté tant que les corrections ne sont pas appliquées.


## 2026-11-10 — Corrections LWE validées après revue Plannotator

Le « oui » utilisateur autorise les quatre corrections discutées.
Suppression de lwe_mission, input_convention et window de config_default.py
et de leur consommation/validation dans obsop.py. Déplacement de sad_path
dans OBS_SSH_NADIR. Chaque source fournit path_lwe explicitement, relatif à
son sad_path optionnel ou absolu ; aucun nom de fichier n’est déduit d’un
label mission. facQ_file est lui aussi explicite et reste commun à l’opérateur.
La convention SLA−LWE reste exigée via name_var/combine_var ; cos² reste la
fenêtre scientifique fixée pour les étapes suivantes. exp.py non modifié.
README, CONTEXT et tests locaux mis à jour pour ces choix.

Tests CPU détachés sous unité vardyn-lwe-review-20261110.service,
invocation 8fbeaff9c525454198808a1e84250d04 ; PID/log/exit/XML dans
runs/lwe_config_review_20261110. 38 tests réussis en 3,52 s : 23 cas de
configuration et 15 de référence, verify.exit=0. Cas ajoutés/adaptés :
fichiers obligatoires même si sad_path existe, noms arbitraires sans mission,
répertoires différents pour un même nom, chemin absolu prioritaire, fusion
Exp inchangée et absence de mutation. Avertissement binaire NumPy/NetCDF
inchangé, sans échec. Identité binaire exp.py/HEAD et git diff --check vérifiés.
Aucune assimilation ni nouvelle étape scientifique, aucun commit/push.


## 2026-11-10 — Étape 2 LWE : cadrage demandé avec grill-me/grilling

- **Demande :** implémenter l’étape 2 avec le skill grill-me, qui renvoie à grilling. Lecture des documents canoniques et des consignes MASSH ; branche VarDyn_Tropics et changements préexistants de l’étape 1 vérifiés et préservés.
- **Décision utilisateur :** périmètre confirmé : temps natif, mission, segment, identité et alignement des lignes dans les observations préparées et les caches, avec tests. Fenêtres LWE et coût conjoint restent aux étapes suivantes.
- **Inspection déléguée en lecture seule :** MIOST segmente globalement par mission avant découpage temporel, sur temps uniques triés (seuil 600 s, minimum sept temps uniques pour la future base). Le coarsen actuel moyenne valeurs, positions et temps ; les caches opérateur par date v2 et empilés v1 ne portent pas la géométrie native.
- **Choix proposés, non encore validés :** refuser subsampling>1 pour les sources LWE actives ; espace de caches LWE distinct, versionné et signé, reconstruction depuis les sources lorsque les métadonnées manquent. Préserver le chemin ordinaire.
- **Statut :** aucune source modifiée ni expérience lancée. Confirmation de la compréhension partagée requise par le skill avant implémentation ; terminer les choix puis coder et tester.


## 2026-11-10 — Étape 2 LWE : Q2 et Q3 validées, Q4 à préciser

- **Décisions utilisateur :** refus de subsampling>1 pour LWE actif (Q2) et caches LWE distincts/versionnés/signés avec reconstruction depuis les sources si nécessaire (Q3) validés.
- **Q4 :** explication demandée avant décision. Le chemin nominal _obs_alti utilise une sélection inclusive aux deux bornes ; une observation exactement à mi-distance de deux checkpoints peut être assimilée deux fois. Proposition de lots semi-ouverts [début, fin[, en conservant temps natifs et lignes distinctes de même timestamp. Cette affectation ne doit pas couper les segments LWE, définis par mission avant les lots.
- **Statut :** CONTEXT.md actualisé ; aucune source modifiée, aucun calcul ni test lancé. Attente du choix Q4 selon le questionnement demandé.

## 2026-11-10 — Étape 2 LWE : géométrie native et caches implémentés

- **Contrat confirmé :** Q4 validée, puis conservation explicite du temps natif indépendamment du checkpoint et accord de poursuite. Affectation [début, fin[, midpoint au checkpoint suivant, observations distinctes de même temps conservées. Q2/Q3 précédemment validées.
- **Sources :** obs.py conserve temps int64 nanosecondes Unix et identifiant de ligne avant sélection/découpage pour les sources nadir LWE actives ; refuse subsampling hors None/1. Espace de caches préparés distinct signé par sources, fichiers/taille/mtime, sélection exacte et cadence. Validation/reconstruction des fichiers préparés incomplets depuis les entrées natives. Chemin ordinaire préservé.
- **Opérateur :** prepare_geometry prépare H spatial et métadonnées sans autoriser l’inversion LWE. Segmentation globale par mission sur temps uniques triés de tous les lots retenus ; coupure strictement >600 s, éligibilité sur le nombre de temps uniques. Segments courts conservés dans les observations océaniques. Table des sources, identité de ligne, temps natif, segment, éligibilité, coordonnées et masque de padding restent sur hôte, alignés aux lignes H. Caches par date et empilés distincts/versionnés/signés, vérification des dimensions et empreinte du contenu ; cache incomplet reconstruit.
- **Premier contrôle :** 42 succès/12 échecs ont révélé le cast float64 de xr.where appliqué aux métadonnées entières. Sélection LWE remplacée par isel pour préserver les nanosecondes exactes. Passe corrigée : **54 tests CPU réussis** (16 géométrie, 23 configuration, 15 référence), code 0 ; avertissement binaire NumPy/NetCDF déjà connu. Invocation f25e0379b6c44db1ab42afb764f0db50 puis 9be6be1a6df043f1979d8e30790c5726, logs/PID/XML dans runs/lwe_geometry_20261110 ; première passe conservée sous verify_initial.log/pytest_initial.xml.
- **Suite :** renforcer les cas multifichiers/entrées devenues obsolètes, vérifier régressions désactivé et documentation, puis validation finale. Aucun noyau dynamique, exp.py, coût ou prior LWE modifié ; aucune assimilation relancée, aucun commit/push.

## 2026-11-10 — Étape 2 LWE validée et documentée

- **Résultat :** géométrie native opérationnelle via `Obsop_interp_lwe_l3.prepare_geometry` ; temps exact conservé indépendamment des dates de checkpoint, métadonnées de mission/ligne/segment alignées aux observations et à H, caches préparés/par-date/empilés séparés et validés. Segmentation globale des lots retenus, temps uniques pour le seuil minimal ; doublons temporels et lignes sans H conservés. Les segments courts sont signalés inéligibles au futur opérateur LWE sans retirer leurs observations.
- **Validation finale :** **59 tests CPU réussis en 4,19 s** (21 géométrie, 23 configuration, 15 référence), verify.exit=0. Cas multifichiers non triés, passage inter-checkpoints, seuil strict 600 s, minimum de temps uniques, midpoint exact, précision à la nanoseconde, multimission/padding/masquage H, vide, caches chauds sans réouverture NetCDF, caches abîmés, données et configurations devenues obsolètes, sous-échantillonnage interdit et régression LWE désactivé. Un avertissement binaire NumPy/NetCDF connu, sans échec.
- **Correction complémentaire :** une passe intermédiaire (58 succès/1 échec) a montré que le dictionnaire d’observations conserve des références aux configurations : comparer les seuls blocs courants ne détectait pas un fichier préparé avec un ancien varmax. Signature de définition source persistée dans le NetCDF et vérifiée à la préparation ; reconstruction depuis les sources réussie. Journaux/XML intermédiaires conservés sous verify_stale_config.log et pytest_stale_config.xml.
- **Traçabilité :** unité finale `vardyn-lwe-geometry-complete-20261110.service`, invocation `ac78ce5be4a24640b9500191416a2e7e`, inactive/dead avec Result=success et ExecMainStatus=0. PID, log, code de sortie, XML, scripts et validation.json dans `runs/lwe_geometry_20261110/`. Compilation de cinq fichiers avec cache /tmp et git diff --check réussis ; exp.py identique octet pour octet à HEAD.
- **Documentation :** README décrit l’API préparatoire, unités/identités, règles de segmentation/affectation, cache et limite de l’estimation ; CONTEXT.md actualisé. Aucun nouveau module dans src, aucun noyau dynamique ni exp.py modifié ; les changements préexistants de configuration/étape 1 sont conservés. Les tests racine restent locaux et ignorés, conformément à l’organisation du projet.
- **Limites/suite :** préparation seulement ; projection LWE et prior/coût conjoint restent aux étapes 3–4, exports/reprises et validation scientifique aux étapes suivantes. Création standard avec LWE actif et utilisation du coût de l’objet préparatoire toujours refusées. Aucune assimilation, aucun commit/push.


## 2026-11-10 — Correction des deux défauts de revue BMaux/LWE

- **Demande :** résoudre les deux constats du sous-agent : changement de disposition BMaux par simple chemin de fond et absence de reconstruction après disparition physique d’un NetCDF préparé LWE. Sources explicitement autorisées ; branche VarDyn_Tropics, changements préexistants conservés. Skill diagnosing-bugs appliqué, causes déjà reproduites par la revue.
- **Régression avant correction :** quatre tests ciblés échouent comme attendu (deux BMaux dont construction complète, deux LWE avec/sans cache opérateur). Unité détachée vardyn-two-review-red-20261110, invocation d092efb86ca843dca4bfd25322bcefa3 ; red.exit=1, logs/PID/XML dans runs/review_two_fixes_20261110.
- **BMaux :** sélection historique uniquement depuis métadonnées explicites des fonds/initialisations. Signature des fonctions spatiales/temporelles et disposition persistées dans Xres/X_it ; tailles, signature et offsets contrôlés avant lecture du fond global, fond de composante, initialisation et reprise. Suppression du prélèvement implicite des derniers coefficients d’un fond de composante. Les anciens fichiers non signés sont refusés avec erreur de compatibilité explicite. Aucune déduction du format depuis la taille ou la présence du chemin.
- **LWE :** prepare_geometry intercepte aussi FileNotFoundError et reconstruit depuis les entrées natives. Régression vérifie temps exacts et IDs, avec/sans cache opérateur préexistant.
- **Première validation corrigée :** 71 tests CPU réussis en 9,50 s ; unité vardyn-two-review-green-20261110, invocation 08afcfb45d7348beadd80f21c4911f98, green.exit=0. Inclut fonds BMaux actuels/historiques explicites, projection inchangée après reprise, rejet de géométrie différente à taille identique, offsets composés, tailles incorrectes et fichiers non signés, ainsi que LWE/configuration/reprise et tests de référence. Un avertissement binaire NumPy/NetCDF connu.
- **Mémoire/suite :** README et CONTEXT actualisés ; vérification finale après ajout des seuils d’atténuation bathymétrique à la signature BMaux, puis compilation et contrôles Git. Aucune assimilation, aucun noyau dynamique ou exp.py modifié, aucun commit/push.


## 2026-11-10 — Corrections BMaux/LWE : validation finale réussie

- **Résultat :** 71 tests CPU réussis en 11,04 s, final.exit=0. L’unité vardyn-two-review-final-20261110.service est inactive/dead, Result=success, ExecMainStatus=0 ; invocation 19fcfbfcc33d48b6b4d1588005d88fed. Logs/PID/XML et validation.json dans runs/review_two_fixes_20261110.
- **Contrôles :** compilation de basis.py, inv.py, obsop.py et des deux fichiers de régression avec cache isolé /tmp ; git diff --check réussi et exp.py identique à HEAD. Métadonnées conservées pour les contrôles finaux et les checkpoints SciPy/Optax ; lecture soumise à validation BMaux sans modifier le chemin des autres bases.
- **Mémoire et limites :** CONTEXT/README actualisés, erreur de format BMaux ancien en suspens désormais corrigée. Anciens fonds BMaux sans métadonnées refusés explicitement ; migration scientifique non inventée. Aucun calcul d’assimilation, aucune modification des résultats existants ou du noyau dynamique ; changements non committés, aucun push.

## 2026-11-10 — Étape 3 LWE : ouverture du cadrage grilling

- **Demande :** étape 3 LWE VarDyn avec grill-me, redirigé vers grilling. Documents canoniques relus, consignes MASSH retrouvées dans /home/flo/MASSH/AGENTS.md ; branche VarDyn_Tropics et changements non committés préservés.
- **Périmètre proposé, à confirmer :** opérateur temporel creux, transposée, prior diagonal issu des cartes et projection blanchie ; coût conjoint et assimilation aux étapes suivantes.
- **Recherche :** inspection en lecture seule déléguée conformément au skill pour préciser la construction MIOST aux extrémités des segments, l’interpolation des cartes et les interfaces existantes.
- **Statut :** première question sur le périmètre ; décisions techniques à soumettre après inspection. Aucune source modifiée, aucun calcul ni test lancé. CONTEXT actualisé en distinguant proposition et choix confirmé.

## 2026-11-10 — Étape 3 LWE : périmètre confirmé et choix techniques identifiés

- **Décision utilisateur :** oui à la projection L, sa transposée, Q=facQ×std_lwe², la projection blanchie et son adjoint, avec tests d’équivalence/alignement/adjoint. Coût conjoint à l’étape 4.
- **Inspection déléguée :** MIOST aligne les centres sur une référence fixe et étend historiquement de trois espacements de chaque côté ; sa sélection utilise quatre espacements plutôt que tdec, équivalents au réglage facnlt=4 uniquement. Les positions aux centres sont interpolées/extrapolées le long de la trace après déroulement des longitudes ; la copie fournie borne aussi les latitudes. Cartes interpolées linéairement, avec des comportements hors domaine différents selon le backend SAD.
- **Questions ouvertes :** compatibilité exacte de placement, politique explicite d’interpolation/hors domaine et traitement des valeurs nulles/invalides du prior. Proposition : erreur explicite sur valeurs non finies/négatives, suppression des coefficients de variance nulle conformément au principe MIOST.
- **Statut :** cadrage en cours, aucune source modifiée ni assimilation/test lancé ; changements préexistants conservés. CONTEXT distingue le périmètre confirmé des décisions restantes.

## 2026-11-10 — Étape 3 LWE : Q2–Q4 confirmées

- **Décisions utilisateur :** oui à tous les choix : placement exact MIOST aux bords, interpolation bilinéaire avec périodicité des cartes globales et erreur hors domaine/manquants, retrait des coefficients nuls et refus des valeurs std_lwe/facQ négatives ou non finies.
- **Vérification ciblée :** la copie lwe.py fournie extrapole linéairement les positions puis borne la latitude à ±90°, contrairement au module MIOST de référence qui ne la borne pas ; les temps uniques conservent la première occurrence pour la géométrie des centres.
- **Suite du cadrage :** confirmer positions des centres, gestion des doublons pour leur géométrie et interface creuse/cache. Aucune source modifiée ni exécution de calcul/test ; branche et changements préexistants préservés. CONTEXT actualisé.

## 2026-11-10 — Étape 3 LWE : Q5–Q7 confirmées, contrat récapitulé

- **Décisions utilisateur :** oui aux positions extrapolées le long de la trace et latitudes bornées comme dans lwe.py fourni ; première ligne par temps unique pour les positions, conservation de toutes les lignes de projection et séparation stricte mission/segment ; préparation dédiée creuse dans obsop.py avec projection blanchie/adjoint, métadonnées, vues checkpoint et cache signé couvrant les cartes. Pondération par sigma à l’étape 4.
- **Contrat :** Q1–Q7 désormais confirmées. Récapitulatif final présenté pour confirmer la compréhension partagée exigée par grilling avant modification source. Périmètre fonctionnel et tests définis ; aucune assimilation prévue dans cette étape.
- **Vérification :** documents canoniques relus, branche VarDyn_Tropics et changements non committés vérifiés ; API prepare_geometry inspectée. Aucune source modifiée ni calcul/test lancé. CONTEXT mis à jour ; mise en œuvre en attente de confirmation finale du contrat récapitulé.

## 2026-11-10 — Étape 3 LWE : implémentation et première validation

- **Autorisation :** compréhension partagée du contrat Q1–Q7 confirmée explicitement par l’utilisateur ; implémentation source autorisée.
- **Travaux :** prepare_projection dans obsop.py, fenêtres et phase MIOST aux temps relatifs natifs, positions extrapolées/bornées, cartes bilinéaires NetCDF ou Zarr v2 ZIP MIOST avec préservation des zéros, prior diagonal et suppression des coefficients nuls. Projection/adjoint JAX creux avec vues par checkpoint et métadonnées, cache versionné signé par géométrie/configuration et contenu des cartes. Coût joint toujours refusé.
- **Tests :** nouvelle suite locale : parité des helpers MIOST réels par extraction AST, facnlt=2/4/6, prior, doublons/segments/missions, JIT/autodiff/adjoint, padding/vues, cartes réelles et synthétiques, cache chaud/abîmé/cartes modifiées, entrées invalides/zéros/vides. Compilation initiale réussie.
- **Exécution :** première passe CPU détachée via unité vardyn-lwe-projection-20261110, nohup avec PID/log/exit/XML dans runs/lwe_projection_20261110. Accès bus utilisateur refusé dans le sandbox, lancement demandé hors sandbox. Résultats à vérifier ; aucune assimilation lancée.

## 2026-11-10 — Étape 3 LWE : premières passes réussies et contrôle final

- **Résultats :** 80 tests CPU réussis en 8,46 s, puis 84 en 8,78 s après contrôles supplémentaires des métadonnées et cas de bords ; verify.exit=0 pour ces deux passes. Helpers MIOST réels extraits en lecture seule (aucune dépendance runtime), cartes SAD réelles décodées correctement, adjoint/autodiff/JIT et vues checkpoint validés. Avertissement binaire NumPy/NetCDF connu sans échec.
- **Renforcement :** contrôles d’identité mission/segment des coefficients et indices creux ; cas longitude 0°, latitudes bornées, extrapolation et doublons, grands supports sans fuite inter-segments, carte modifiée à taille/mtime identiques et segments courts. Lecture des cartes mutualisée entre segments, avec régression dédiée pour éviter une redécompression par segment.
- **Traçabilité :** invocations 63a80f8e8a1c45fab150f7f73f21007c et 0bcff37064264b938b0889481e2cc94d ; logs/XML initiaux conservés. Passe finale unité vardyn-lwe-projection-complete-20261110, invocation d9f982dbe6fb4ee7aed285934f1a8a20. Compilation obsop/test et git diff --check réussis ; validation finale à contrôler.
- **Documentation :** README décrit les méthodes, unités, format des sorties, placement/positions, prior, cartes et caches ; CONTEXT actualisé. Coût joint et création standard LWE active restent refusés à l’étape 4. Aucun noyau, exp.py, minimiseur ou assimilation modifié/lancé ; travaux antérieurs préservés.

## 2026-11-10 — Étape 3 LWE achevée et validée

- **Résultat final :** 85 tests CPU réussis en 8,75 s ; verify.exit=0, aucune erreur/échec. Un avertissement binaire NumPy/NetCDF connu. Unité vardyn-lwe-projection-complete-20261110.service inactive/dead, Result=success, ExecMainStatus=0 ; invocation de lancement d9f982dbe6fb4ee7aed285934f1a8a20. Rapport, validation.json avec empreintes, PID/log/exit et XML dans runs/lwe_projection_20261110 ; passes 80/84 conservées.
- **Contrôles :** compilation obsop.py et test avec cache isolé /tmp, git diff --check et identité exp.py/HEAD réussis. Lecture des cartes une fois par préparation vérifiée. Projection, prior et métadonnées restent distincts du coût et des contrôles dynamiques.
- **Documentation :** README et CONTEXT décrivent le contrat effectivement implémenté, méthodes/unités, géométrie MIOST, cartes, zéros/invalides, caches et limites. Le test reste local/ignoré conformément à l’organisation du projet ; aucun nouveau module src.
- **Limites et suite :** préparation et validation CPU seulement ; coût conjoint, sorties/reprises et pilote scientifique restent aux étapes 4–6. Aucune assimilation ni validation GPU, aucun noyau/exp/minimiseur modifié, aucun commit/push. Changements antérieurs de BMaux et des étapes LWE 1–2 préservés.

## 2026-11-10 — Étape 4 LWE : ouverture du cadrage grilling

- **Demande :** étape 4 LWE avec grill-me, redirigé vers grilling. Documents canoniques relus et consignes MASSH consultées ; branche VarDyn_Tropics et changements non committés vérifiés et préservés.
- **Périmètre proposé, à confirmer :** coût conjoint et gradients du contrôle LWE blanchi, dans le cœur générique à adjoint explicite et le chemin SW annuel local à autodiff ; tests synthétiques de gradient et optimum analytique. Sorties/reprises et pilote scientifique aux étapes 5–6.
- **Recherche :** inspection en lecture seule déléguée selon le skill pour vérifier pondération sigma, disposition des contrôles/prior et interfaces des deux chemins. Les décisions techniques dépendantes seront soumises après confirmation du périmètre et retour de l’inspection.
- **Statut :** aucune source modifiée, aucun calcul ou test lancé ; CONTEXT actualisé en distinguant proposition et décision. Confirmation de compréhension partagée requise par le skill avant implémentation.

## 2026-11-10 — Revue Plannotator de l’étape 3 LWE ouverte

- **Demande :** revue des modifications de l’étape 3 avec le skill plannotator-review.
- **Périmètre :** patch statique de la projection/adjoint dans obsop.py, tests locaux test_lwe_projection.py et documentation README/section étape 3 du CONTEXT. Le patch reconstruit l’état antérieur aux seuls ajouts de l’étape 3 pour exclure BMaux, étapes LWE 1–2 et cadrage concurrent de l’étape 4 ; aucun état Git modifié.
- **Session :** plannotator review --patch-file runs/lwe_projection_20261110/review_step3.patch --json ; interface http://localhost:19445, sortie /tmp/vardyn-lwe-step3-review.log. Lancement hors sandbox pour permettre le serveur local. Session en attente de soumission ; aucune annotation ni approbation encore reçue.
- **Limite :** mode patch statique sans navigation/édition/staging du checkout depuis l’interface ; annotations sur le diff disponibles. Aucune modification source ni nouvelle exécution scientifique.

## 2026-11-10 — Étape 4 LWE : périmètre confirmé et interfaces vérifiées

- **Décision utilisateur :** Q1 confirmée (« ok ») : coût conjoint et gradients dans le cœur générique Python/scan et le chemin SW annuel autodiff ; tests synthétiques, sorties/reprises à l’étape 5 et pilote à l’étape 6.
- **Inspection en lecture seule :** contrôle océanique générique éventuellement blanchi par B ; le contrôle LWE doit garder son prior identité indépendamment de prec/sigma_B. Le chemin local Optax inclut déjà le prior identité de tout son contrôle ; éviter un ajout double. Les vues temporelles LWE utilisent des dictionnaires Python : dans scan, employer une projection complétée et des indices JAX. Préserver l’ordre multi-opérateurs, padding et jours sélectionnés.
- **Défaut vérifié :** inv.py applique R.inv au coût mais pas aux résidus transmis aux adjoints explicites, alors que obsop.py ne fournit que la pondération errobs. Coût/gradient incohérents si sigma_R diffère de l’identité. Correction proposée pour les deux gradients, à valider explicitement avant édition.
- **Suite du cadrage :** disposition et initialisation du contrôle LWE, politique sigma_R, activation limitée avant persistance signée. Aucune source modifiée ni calcul lancé ; documents actualisés, changements préexistants et revue Plannotator concurrente préservés.

## 2026-11-10 — Étape 4 LWE : Q3–Q4 validées, clarification Q2

- **Décisions utilisateur :** correction de la pondération R.inv dans les adjoints acceptée (Q3), avec précision que le protocole pratique utilise seulement sigma_noise par satellite ; activation scientifique reportée à la persistance signée de l’étape 5 (Q4).
- **Clarification Q2 :** z_lwe est déjà blanchi par la transformation des coefficients physiques a=sqrt(Q_lwe)z_lwe ; son prior identité doit rester le même quel que soit prec pour l’océan. Une transformation numérique supplémentaire est possible si elle est répercutée partout ; aucune indépendance postérieure n’est affirmée. La question reste ouverte jusqu’à la réponse utilisateur.
- **Vérification :** sigma_noise alimente errobs dans obsop.py ; les résidus sont divisés par errobs, tandis que INV.sigma_R définit une covariance supplémentaire. Pour le protocole indiqué, conserver cette dernière à l’identité et éviter une seconde pondération par sigma_noise. Lecture seule du code ; aucun calcul ni modification source. CONTEXT actualisé.

## 2026-11-10 — Étape 4 LWE : Q2 validée, contrat récapitulé

- **Décision utilisateur :** Q2 confirmée (« ok ») après explication de la disposition du vecteur : contrôles océaniques existants puis coefficients LWE, tous optimisés ensemble. Blocs LWE identifiés par opérateur, initialisation nulle et prior blanchi propre au LWE ; aucune transformation par la covariance océanique.
- **Contrat complet :** coût/gradients génériques Python/scan et SW annuel autodiff, correction R.inv autorisée, sigma_noise par satellite pour le protocole pratique, tests de gradients/optimum gaussien et régressions ; activation scientifique/sorties/reprises aux étapes suivantes. Q1–Q4 arrêtées.
- **Statut :** récapitulatif soumis pour confirmation de compréhension partagée requise par grilling avant implémentation. CONTEXT actualisé ; branche et changements préexistants vérifiés. Aucune source modifiée, aucun calcul/test ni commit/push.

## 2026-11-10 — Étape 4 LWE : implémentation autorisée, première validation

- **Autorisation :** contrat final Q1–Q4 confirmé explicitement par l’utilisateur (« ok »).
- **Travaux :** API prepare_joint et disposition LweControl dans obsop.py ; projection/transpose normalisées par errobs, sélection temporelle JAX sans dictionnaire tracé. Variational sépare le préfixe océanique et le suffixe LWE, ajoute une fois le prior identité et son gradient, et transmet R.inv aux adjoints Python/scan. Chemin SW local : packing, groupes et binding du résidu, correction avant sélection des jours, états physiques séparés. Gardes du factory/driver générique et Optax local jusqu’à l’étape 5.
- **Validation en cours :** tests synthétiques d’optimum MAP, gradients séparés/conjoints, Python/JIT/scan/autodiff, padding/multi-opérateurs/jours exclus et gardes. Test historique SW corrigé pour retirer la seconde pondération R qui compensait l’ancien défaut. Compilation initiale réussie.
- **Exécution :** CPU détaché systemd/nohup, unité vardyn-lwe-joint-cost-20261110, invocation adebb8c5da394a38921be4afa7032826 ; PID/log/exit/XML dans runs/lwe_joint_cost_20261110. Bus utilisateur refusé dans le sandbox, lancement hors sandbox autorisé. Première passe en cours avec des échecs à diagnostiquer ; aucun résultat de validation revendiqué. Aucun noyau, exp.py ni assimilation modifié/lancé.

## 2026-11-10 — Étape 4 LWE : contrôles ciblés réussis et régressions en cours

- **Première passe :** 93 tests réussis/6 échecs en 70,92 s. Cinq échecs viennent du faux State de test sans setvar ; le sixième attend encore le texte de la garde étape 4 au lieu de la garde étape 5. Montage et assertion corrigés ; aucune défaillance scientifique démontrée par ces six cas. Journaux/XML initiaux conservés sous verify_initial.log/pytest_initial.xml.
- **Validation ciblée corrigée :** 8 tests réussis en 21,95 s, focus.exit=0 ; Python/JIT/scan, optimum MAP, gradients, sélection et trajectoire du chemin local, gardes et blocs multiples. Invocations diagnostic 38d0b75b5de8421ea9c4dd0d5235393f puis 3515a7f4be264241834e0ce8e4eff691.
- **Renforcement :** parité directe coût/gradient entre évaluateur générique et scan SW local sur la même trajectoire linéaire ; erreurs satellites différentes et sources LWE désactivées, équivalence n_lwe=0 et précision coût mixte. Les opérateurs LWE désactivés ne créent aucun bloc ; garde de préparation explicite conservée.
- **Passe complète en cours :** unité vardyn-lwe-joint-green-20261110, invocation a5f178cd8d67444688a6f6635bdf4ce9, mêmes logs/PID/XML. README décrit la nouvelle API et le périmètre limité ; les résultats finaux restent à vérifier. Aucun calcul d’assimilation ni commit/push.

## 2026-11-10 — Étape 4 LWE : 101 tests réussis, contrôle final lancé

- **Résultat :** 101 tests CPU réussis en 78,50 s, verify.exit=0 ; unité green inactive/success avec ExecMainStatus=0. Parité directe générique/SW autodiff, gradients et optimum gaussien, erreurs par satellite, régressions LWE/SW et trois tests de référence inclus. Logs/XML conservés sous verify_green.log/pytest_green.xml.
- **Derniers cas :** factory LWE entièrement désactivé (aucun suffixe ni garde de production), précision coût mixte avec LWE non nul et absence de prior océanique. La garde Variational préc=True sans B conserve l’absence de transformation océanique ; le prior LWE reste identité.
- **Exécution finale :** unité vardyn-lwe-joint-final-20261110, invocation c0b118ebb6e4496daba5a370a8c76461, PID/log/exit/XML dans le même dossier. Le marqueur exit ancien est supprimé au début du contrôleur pour ne pas confondre une passe en cours avec la précédente.
- **Contrôles :** compilation de huit fichiers Python avec cache /tmp, bash -n et git diff --check réussis ; exp.py identique à HEAD. README/CONTEXT actualisés ; résultat final à vérifier. Aucun noyau ni assimilation modifié/lancé, aucun commit/push.

## 2026-11-10 — Étape 4 LWE achevée et validée

- **Résultat final :** 103 tests CPU réussis en 81,07 s, verify.exit=0 ; unité vardyn-lwe-joint-final-20261110.service inactive/dead, Result=success et ExecMainStatus=0. Invocation c0b118ebb6e4496daba5a370a8c76461. Rapport, validation.json passed avec hashes, jobs.json, PID/log/exit/XML dans runs/lwe_joint_cost_20261110.
- **Contrat livré :** prepare_joint et LweControl ; contrôle joint océan/LWE, prior LWE identité indépendant du choix de coordonnées océaniques, résidu en mètres SSH avant normalisation ; gradients explicites cohérents avec R.inv et autodiff SW local. Prior unique dans le chemin local ; seuls les contrôles océaniques influencent la trajectoire physique. Initialisation LWE nulle, offsets par opérateur et groupes locaux exposés.
- **Couverture :** optimum gaussien analytique et mini-optimisations BFGS, dérivées directionnelles océaniques/LWE/conjointes, Python/JIT/scan/autodiff et parité directe SW local, sigma_R non identité et erreurs satellites distinctes, padding/missions actives-inactives/multi-opérateurs/temps absents, jours exclus, dernier checkpoint partiel, LWE nul/factory désactivé, précision mixte et absence de prior océanique. Régressions LWE et tests de référence, bases SW/modèle QGSW et physique locale inclus. Avertissements NumPy/NetCDF, précision/dépréciation JAX et vent de petits cas connus, sans échec.
- **Contrôles et documentation :** huit fichiers compilés avec cache /tmp ; bash -n, git diff --check et identité exp.py/HEAD réussis. README/CONTEXT décrivent les API et la limite d’activation ; changements antérieurs BMaux/configuration/géométrie/projection préservés. Aucun nouveau module src ni modification des noyaux.
- **Limites/suite :** aucune assimilation ni validation GPU. Gardes conservées avant écriture/reprise de contrôles joints non signés dans le factory, Inv_4Dvar et le driver local recevant un résidu joint déclaré. Étape 5 : sorties/reprises/corrections ; étape 6 : pilote scientifique. Modules locaux/tests restent ignorés conformément à l’organisation demandée ; aucun commit/push.

## 2026-11-10 — Vérification demandée de LWE avec Obsop_multi

- **Demande :** vérifier le fonctionnement de l’opérateur avec Obsop_multi. Documents canoniques et consignes MASSH relus ; branche VarDyn_Tropics et changements préexistants vérifiés.
- **Travaux :** ajout local de tests d’intégration dépassant le précédent test de disposition : deux LWE préparés avec dates/tailles/bruits différents, L3 ordinaire, L4 actif et vide, permutations et composition imbriquée. Coût conjoint et gradients Python/JIT/scan comparés à la somme indépendante des feuilles différentiée par JAX ; concaténation et adjoints réels multi vérifiés, séparation des blocs et factory désactivé couverts.
- **Limite explicitement testée :** le constructeur standard Obsop_multi refuse encore une composante LWE active avec la garde étape 5, conformément au contrat validé. Composition des opérateurs préparés utilisée pour les évaluateurs de l’étape 4 ; pas de prétention de production/reprise.
- **Exécution :** CPU détaché systemd/nohup, unité vardyn-lwe-multi-check-20261110, invocation 6047ae7447c34904a272601e5b36364c ; scripts/PID/log/exit/XML dans runs/lwe_multi_check_20261110. Résultats à vérifier ; aucune modification source ni assimilation lancée.

## 2026-11-10 — Compatibilité Obsop_multi vérifiée sur les opérateurs préparés

- **Résultats :** 22 tests CPU réussis en 56,47 s, verify.exit=0. Six cas multi (trois dispositions × deux choix de blanchiment océanique) vérifient la somme des coûts et les gradients contre l’autodiff indépendante des feuilles, les adjoints/concaténations réels, les dates absentes, le padding, les opérateurs L3/L4 avec ou sans LWE et les permutations/imbriquements. Les régressions coût joint et L4 vide passent aussi.
- **Prétraitement :** 1 test supplémentaire réussi en 3,09 s, process.exit=0 ; appel réel Obsop_multi.process_obs sur LWE préparé + L3 ordinaire, identité exacte des tableaux données/erreurs/indices/poids/comptes et de la projection LWE. Unité vardyn-lwe-multi-process-check-20261110, invocation 9c213e5da53b42b3a2ab9f1a85108ff0.
- **Clôture :** deux unités inactive/success, ExecMainStatus=0. REPORT.md, validation.json passed avec empreintes, jobs.json et XML/log/PID conservés dans runs/lwe_multi_check_20261110. Compilation du nouveau test et git diff --check réussis ; avertissements connus NumPy/NetCDF et précision JAX sans échec.
- **Conclusion et limite :** composition des opérateurs préparés compatible avec les évaluateurs de l’étape 4. Le constructeur standard appelle le factory enfant et refuse toujours LWE actif jusqu’aux sorties/reprises signées de l’étape 5 ; cette garde et le constructeur désactivé sont testés. Aucun changement de source nécessaire : hashes du cœur et des modules locaux identiques à validation étape 4. Test supplémentaire local/ignoré ; aucune assimilation/GPU, aucun commit/push. CONTEXT actualisé.

## 2026-11-10 — Étape 5 LWE : mise en œuvre autorisée

- **Demande :** implémenter l’étape 5 du plan : persistance signée des contrôles joints, reprises/rejeu et sorties LWE séparées des états physiques, dans le cœur générique et le chemin SW local. Le pilote scientifique reste à l’étape 6.
- **Inspection :** branche VarDyn_Tropics ; modifications antérieures BMaux et LWE 1–4 présentes et préservées. Documents canoniques et consignes MASSH relus ; snapshots des cinq modules concernés dans runs/lwe_outputs_20261110. Aucun nouveau module src prévu.
- **Choix :** conserver les coordonnées océaniques historiques de Xres et ajouter le suffixe LWE blanchi ; les checkpoints gardent les coordonnées du minimiseur. Signer disposition, opérateurs, géométrie native, projection/prior, erreurs et conversion SSH ; refuser les reprises LWE absentes ou incompatibles même à taille égale. Exporter l’erreur additive e et la correction −e, ainsi que les résidus avant/après, sans padding.
- **Validation prévue :** aller-retour et rejeu, changement de carte/géométrie/ordre, multimission, zéros/vides/désactivé, chemins générique et local, régressions des étapes précédentes. Aucune assimilation scientifique lancée.

## 2026-11-10 — Étape 5 LWE : persistance et première validation

- **Implémentation :** constructeur/factory actif raccordé à prepare_joint ; préparation native préservée lors de process_obs, sans substitution par les caches L3 ordinaires. Signatures versionnées des blocs, géométrie/projection/prior, données/erreurs/conversion et ordre des feuilles ; validation des coordonnées analyse/minimiseur et du prior océanique avant chargement.
- **Sorties :** Xres contient l’analyse océanique puis z_lwe ; X_it et checkpoint local conservent le contrôle du minimiseur. Export atomique NetCDF séparé lwe_diagnostics.nc : identités/temps int64 natifs, sources/segments/checkpoints, erreur additive, correction opposée, résidus avant/après, masque assimilé et coefficients blanchis/physiques. Aucun LWE transmis aux bases ou aux sorties physiques.
- **Tests :** première passe 10 réussites/5 échecs : quatre révèlent une signature qui inspectait __dict__ vide d’un tableau JAX au lieu de ses octets ; le cinquième révèle l’absence du rejeu maxiter=0 dans le driver local. Deux causes corrigées. Passe ciblée corrigée : 15 tests réussis en 5,95 s, verify.exit=0. Avertissement NumPy/NetCDF connu sans échec.
- **Traçabilité :** invocations 878211c982694bd4919e3b07e1693f72 puis 8a1e639c2c5a46818a2cbf4ecf58ebf4 ; logs/XML initiaux et ciblés préservés dans runs/lwe_outputs_20261110. Régressions et tests des trois minimiseurs effectivement optimisés lancés (invocation ef6249379ffc4f2aa865f47b6d96d8da). Aucun pilote scientifique ni GPU lancé.

## 2026-11-10 — Étape 5 LWE : régressions et chemin SciPy

- **Régressions :** 140 tests CPU réussis/1 échec en 120,19 s. L’échec dans la nouvelle optimisation SciPy expose un import local time après le premier appel à fun : il masque l’import global et provoque NameError. Retrait de l’import local redondant dans inv.py ; nécessaire pour exécuter ce chemin autorisé.
- **Contrôle ciblé :** l’optimisation devient exécutable ; l’assertion suivante montre que le seuil de convergence synthétique se déclenche dès l’évaluation initiale avec la patience 1 historique. Le test SciPy désactive ce critère personnalisé pour tester une optimisation effective et sa persistance ; la sémantique du minimiseur n’est pas modifiée. Le lanceur ciblé initial nommé scipy.py masquait le paquet ; renommé focus_scipy.py, sans incidence sur les sources.
- **Renforcement :** signature observatoire calculée une fois par driver générique pour éviter de transférer/hacher tous les tableaux à chaque checkpoint écrit. Cas carte réelle modifiée à taille égale, observations vides et rejeu local d’un checkpoint non convergé ajoutés. Sorties/reprises locales vérifient également le préconditionneur et conservent la non-convergence au rejeu.
- **Validation finale :** unité vardyn-lwe-outputs-final-20261110, invocation 42a51637ee524121adac2cd6f15e4bbc ; CPU/nohup, logs/PID/XML/exit dans runs/lwe_outputs_20261110. Résultat final à vérifier. Aucune assimilation scientifique ni validation GPU.

## 2026-11-10 — Étape 5 LWE : cas limites précisés

- **Vérifications :** les trois optimiseurs génériques produisent des contrôles joints rechargeables et des corrections cohérentes avec le suffixe sauvegardé. Rejeu local d’un checkpoint non convergé sans nouvelle optimisation vérifié, sans changement du statut de convergence.
- **Cas vide :** le nouveau test utilisait un fichier brut physiquement vide, rejeté avant l’opérateur par une réduction min/max du chargeur historique. Le périmètre testé est corrigé vers un opérateur sans lots retenus ; aucune modification du chargeur hors étape 5. Le test couvre maintenant changement réel de carte à taille égale et export NetCDF avec deux dimensions vides : 1 test réussi en 2,17 s, edge.exit=0, invocation 4b56c251a45c4d69bec4bd95a22e10df.
- **Limite existante :** la prise en charge des fichiers bruts physiquement vides n’est pas ajoutée. Les opérateurs sans observations retenues sont pris en charge. Rapports ciblés conservés ; aucune assimilation scientifique lancée.

## 2026-11-10 — Étape 5 LWE achevée et validée

- **Résultats finaux :** 143 cas CPU distincts validés : 123 régressions réussies dans la passe élargie plus les 20 tests de sorties/reprises réussis dans la passe finale en 10,18 s. outputs.exit=0 ; unité vardyn-lwe-outputs-complete-20261110 inactive/success, ExecMainStatus=0, invocation 0afb8fd8b1f545b6a3c28e4e5e056d6f. La passe élargie précédente reste conservée avec 142 succès/1 échec de montage du cas vide en 120,05 s ; son échec est résolu dans la passe ciblée finale. validation.json décrit explicitement cette agrégation, sans annoncer une invocation unique de 143 tests sans échec.
- **Contrat livré :** constructeur standard/multi actif, préparation native préservée ; contrôles joints signés dans Xres/X_it et checkpoints/préconditionneurs locaux ; vérification des identités, tailles/offsets, cartes/géométrie/ordre, données/erreurs/conversion et coordonnées, plus prior générique et provenance locale. Export atomique lwe_diagnostics.nc séparé : temps exacts, IDs natifs, erreur additive et correction opposée, résidus avant/après, sigma, masque d’assimilation et coefficients blanchis/physiques. Seul le préfixe océanique entre dans les bases et états physiques.
- **Couverture :** optimisation et rejeu des trois minimiseurs génériques ; chargement analyse/checkpoint, prior océanique, cas taille égale incompatible, multimission/multi-opérateurs, doublons temporels, cartes changées, contrôles nuls/vides et absence de lots, optimisation/reprise/rejeu local et conservation de non-convergence. Régressions LWE 1–4, Obsop_multi, BMaux, reprises, physique/forçage SW et tests de référence réussies.
- **Traçabilité :** REPORT.md, validation.json passed et hashes, source_checks.json, jobs.json, snapshots avant édition et XML/log/PID/exit dans runs/lwe_outputs_20261110. Neuf fichiers Python compilés ; bash -n, git diff --check et exp.py identique à HEAD vérifiés. NumPy/NetCDF et avertissements JAX connus sans échec final.
- **Limites/suite :** CPU synthétique seulement ; aucune assimilation réelle, validation GPU ni pilote scientifique. Fichiers bruts physiquement vides soumis à la limite existante du chargeur ; opérateur sans lots retenus validé. Provenance des bases/physique locales toujours fournie par le pilote. Étape 6 : pilote scientifique puis comparaison OSE. README/CONTEXT actualisés ; modules locaux et tests supplémentaires ignorés par Git comme demandé. Changements préexistants préservés ; aucun commit/push.

## 2026-11-10 — Revue des modifications d’intégration LWE

- **Périmètre :** modifications non committées contre HEAD `5e414dfc`, branche VarDyn_Tropics ; configuration, observations, géométrie/projection/coût, sorties/reprises génériques et modules SW locaux. Documents canoniques et consignes MASSH relus ; changements BMaux antérieurs préservés. Revue standards et spécification déléguée selon la compétence code-review ; aucun code source modifié.
- **Défaut P1 reproduit :** LweControl.correction et son adjoint masquent le padding, mais pas les lignes SSH rejetées/NaN ni les erreurs NaN. Les lignes rejetées restent dans les données préparées et leur résidu océanique est neutralisé ; ajouter e/sigma y crée une contrainte LWE fictive. Avec toutes les observations manquantes : coût 1,52119899 au lieu du seul prior 0,08500001, écart maximal du gradient LWE 5,40677. Une erreur NaN rend le coût NaN. Le défaut partagé concerne les chemins générique et local.
- **Défaut P2 reproduit :** analysis_control puis export_lwe rejouent X0+Xb alors que les coûts utilisent X0−Xb avec B=None. Avec Xb=0,1, écart maximal des résidus exportés 0,1199997 m. La reprise analyse préconditionnée dans ce même cas échoue aussi sur inverse_sqrt de None ; ce dernier comportement historique n’est pas compté comme constat nouveau distinct.
- **Standards :** aucune violation documentaire bloquante ; deux pistes heuristiques de maintenance : parcours répétés de l’arbre d’opérateurs et duplication du helper d’écriture NetCDF atomique. Une exploration de signature après changement d’état initial n’est pas retenue comme constat : le contrat de reprise n’établit pas suffisamment cette exigence générale.
- **Validation :** 125 tests CPU existants réussis en 71,15 s, trois avertissements connus, verify.exit=0 ; git diff --check réussi. Reproductions supplémentaires terminées, check.exit=0. Exécutions détachées systemd/nohup : suite invocation `10e6c1f37d114b9db92786c4d1dfdb66`, reproductions finales `c2efc4befb4e48b9a9b63086f4748e4a` ; première reproduction `182ca59a22c849b6b8a7c187ca7fd32b`. Scripts/logs/PID/XML, JSON des résultats et rapport dans runs/lwe_review_20261110.
- **Suite et limites :** constats présentés pour correction ultérieure explicitement autorisée ; CONTEXT actualisé sans effacer les validations antérieures. Aucune assimilation réelle, validation GPU, modification des sources, commit ni push.

## 2026-11-10 — Corrections des deux constats LWE autorisées et implémentées

- **Demande :** corriger les deux défauts de la revue, avec préservation des changements antérieurs et des expériences. Branche VarDyn_Tropics et fichiers concernés vérifiés ; copies des deux sources avant édition dans runs/lwe_review_fixes_20261110. Compétence diagnosing-bugs appliquée ; causes et reproductions déjà établies à la revue, sans nouvelle phase spéculative.
- **Régression avant correction :** 13 tests échouent en 6,77 s sur les symptômes attendus : SSH manquantes/varmax/erreurs NaN dans Python/scan, gradients avec lignes partiellement manquantes, autodiff locale et export sans prior avec préc activé/désactivé. red.exit=1, invocation `9008beb88c2c4df0b0f518656c78d7f8`.
- **Masquage :** observation_mask dans l’opérateur LWE associe padding, SSH finie et erreur finie positive. Résidu océanique, correction normalisée, adjoint et diagnostics partagent ce masque ; les divisions sont assainies avant calcul pour préserver l’autodiff. Les observations natives restent identifiables ; sorties assimilated=0 et résidus NaN pour les lignes exclues, correction physique conservée. L3 ordinaire inchangé.
- **Coordonnées :** helper _ocean_control commun aux deux coûts et à analysis_control/export_lwe ; B=None conserve la convention historique X0−Xb. Le chargement applique X+Xb, sans appeler une covariance absente. Suffixe LWE et chemins avec covariance conservés.
- **Premier résultat :** 13 tests réussis en 7,74 s, green.exit=0, invocation `759ca2fcf8df42acb3880bfea8ee5972`. Six cas supplémentaires d’erreurs nulles/négatives/infinies ajoutés ; la division d’export évite également ces valeurs pour les lignes exclues. Suite élargie à lancer et vérifier, validation non encore close. Logs/PID/XML/exit conservés ; aucune assimilation/GPU, aucun commit/push.

## 2026-11-10 — Corrections LWE : parité arithmétique préservée

- **Régression élargie :** la première passe complète révèle un échec de parité exacte entre LWE de variance nulle et L3 ordinaire. Reproduction ciblée : coût 61718,617 contre 61718,610 (écart relatif 1,27e−7), provenant de residual/sigma au lieu de (1/sigma)*residual ; invocation `2460a0a9da0945fab7ebee1c8cdf55a1`, equivalence.exit=1.
- **Correction :** ordre arithmétique historique restauré, masquage et assainissement conservés. **20 tests ciblés réussis en 12,18 s** : les 19 cas des deux corrections et la parité LWE nul/L3 avec précision mixte ; focused.exit=0, invocation `c1cb82e2f3c346c2be9e5f392fb684d4`.
- **Rejeu original :** erreurs SSH NaN : coût exactement égal au prior 0,0850000083 et écart du gradient LWE nul ; sigma NaN : même coût fini ; export sans prior : écart maximal 2,93e−7 m et chargement accepté. Preuves de revue antérieures préservées, copie du rejeu et sorties dans le nouveau dossier ; invocation `28e789100a8e41a0af8e83b8cd7ec667`.
- **Validation finale en cours :** première passe complète interrompue après identification/correction de sa régression pour éviter de terminer sur la version chargée antérieure. Nouvelle suite intégrale sous unité vardyn-lwe-review-fixes-final-20261110, invocation `8b338e0d04b1401ea5a6b735652bc426`. Compilation des deux sources et du test, bash -n et git diff --check réussis ; exp.py inchangé. Aucun noyau dynamique ni module local modifié.

## 2026-11-10 — Deux corrections LWE validées sur la suite complète

- **Résultat final :** 236 tests CPU réussis en 262,02 s, full_final.exit=0 ; les 19 cas du nouveau fichier tests/test_lwe_review_fixes.py sont inclus. NumPy/NetCDF, précisions JAX et dépréciations génèrent 368 avertissements, sans échec. Unité vardyn-lwe-review-fixes-final-20261110 inactive/success, ExecMainStatus=0, invocation `8b338e0d04b1401ea5a6b735652bc426`.
- **Couverture :** SSH manquante/rejetée par varmax, erreurs NaN/nulles/négatives/infinies, contrôle LWE nul/non nul, gradients explicites Python/scan et autodiff locale, transposée avec lignes manquantes, diagnostics natifs des lignes exclues ; conversion et export sans covariance océanique avec Xb non nul, préc activé/désactivé, résidus confrontés au coût et aller-retour signé. Régressions LWE 1–5, multi-opérateurs, sorties/reprises, BMaux, modèles/bases SW, physique/forçages et apprentissage local inclus.
- **Traçabilité :** validation.json, source_checks.json, jobs.json, patch limité fixes.patch, snapshots et logs/PID/XML/exit dans runs/lwe_review_fixes_20261110. Reproductions originales avant correction conservées dans runs/lwe_review_20261110 ; résultats après correction dans le nouveau dossier. La première passe complète a ignoré SIGINT puis a été arrêtée avec systemctl stop ; elle est explicitement marquée interrompue, sans verdict complet.
- **Contrôles :** compilation des deux sources et du nouveau test avec cache /tmp, bash -n, git diff --check et exp.py identique à HEAD réussis. README et CONTEXT décrivent le masque commun et la conversion sans prior ; journal à jour. Seuls obsop.py et inv.py modifiés pour ces corrections, en plus de la documentation et du test local ignoré comme demandé. Changements source antérieurs préservés.
- **Limites/suite :** aucune assimilation réelle, validation GPU, modification de noyau ou de module learning_local, commit ni push. Pilote scientifique de l’étape 6 restant à réaliser.

## 2026-11-10 — Enregistrement Git demandé des modifications LWE et BMaux

- **Demande :** créer un commit local après correction des deux constats LWE.
- **Périmètre :** huit fichiers suivis modifiés : configuration et observations LWE, géométrie/projection/coût conjoint, sorties/reprises signées et masquage corrigé dans obsop.py/inv.py, protection des reprises BMaux dans basis.py, README/CONTEXT/JOURNAL. Ces changements, déjà présents et validés ensemble, constituent l’état intégré du cœur ; les deux derniers correctifs dépendent de cette intégration non encore committée.
- **Vérification :** branche VarDyn_Tropics ; index initial vide, diff relu, git diff --check réussi. Empreintes des sources finales et du test de régression identiques à la validation réussie de 236 tests CPU ; aucune relance nécessaire en l’absence de changement source depuis cette validation.
- **Préservation :** tests/, scripts/ et learning_local/ restent ignorés conformément à l’organisation demandée ; aucun ajout forcé. Runs, références MIOST default.py/lwe.py, skills locaux .agents/ et bytecode non suivis restent hors du commit. Aucun push ni nouveau calcul scientifique.
