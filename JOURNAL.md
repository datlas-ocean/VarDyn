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
