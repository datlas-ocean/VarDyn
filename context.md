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

Le projet est un worktree Git dans `/home/flo/Workspace/VarDyn_Tropics`. La branche active est désormais **`VarDyn_Tropics`**, à la demande de fusion du 2026-11-09. La fusion de **`Dev-SW`** est validée sans conflit : 81 tests CPU de la suite complète et 2 tests supplémentaires du pilote Optax réussis. Les corrections locales SW précédentes ont été enregistrées sur `Dev-SW` au commit `66c8aded` pour les préserver avant fusion. Les correctifs propres à Tropics, notamment L4 vide, sont conservés. Le dépôt principal `/home/flo/VarDyn` reste indépendant dans ses fichiers de travail.

**Complétude de l’apprentissage SW :** 16 modules historiques manquants ont été récupérés dans `mapping/src/` pour MAP1, les statistiques locales, la réduction temporelle MAP2, les contrôles physiques, les diagnostics EM/CG et la composante barotrope optionnelle. Les imports AR(1) pointent vers `basis.py` ; les anciens modules de bases déjà regroupés ne sont pas réintroduits. Quatre pilotes `learn_sw_*` sont présents dans `mapping/scripts/`, avec `--config` explicite et empreintes adaptées aux fichiers actuels. Les configurations scientifiques et données historiques restent à fournir. Documentation : `docs/sw_learning.md` ; inventaire et validation : `runs/merge_dev_sw_20261109/`.

**Validation de l’intégration :** 29 modules src importés depuis ce worktree, aucun import relatif manquant, quatre CLI --help valides, 36 fichiers compilés avec py_compile (cache dans /tmp), git diff --check réussi. Suite complète : 81 tests en 130,57 s. Deux tests supplémentaires du pilote Optax en 2,03 s : optimum gaussien, modes prior/groupé et reprise/signature. Le mode prior du pilote récupéré a été corrigé pour utiliser une transformation identité lorsque les groupes sont absents. Empreintes de provenance versionnées dans docs/sw_learning_source_inventory.json ; logs et PID dans runs/merge_dev_sw_20261109. Pas de validation GPU ni apprentissage annuel.

Comparaison directe de Dev-SW au main local `350fa951` : 11 fichiers, 2 770 ajouts et 345 suppressions ; extensions du modèle SW, bases de forçage, covariances multivariées, fenêtres de forçage et minimiseur par blocs. Rapport : [comparaison Dev-SW/main](runs/dev_sw_comparison_20261106/report.md). La branche Dev-SW est locale et absente du distant configuré, y compris après fetch ; aucune version distante plus récente n'a été identifiée.

**Corrections SW du 2026-11-09, enregistrées dans `66c8aded` avant fusion :** les remarques Plannotator sont implémentées. Les bases SW journalière, générique et calibrée et leurs opérateurs spatiaux sont réunis dans `mapping/src/basis.py`. Les helpers spécifiques de compatibilité BMaux sont dans sa classe. Le module partagé `mapping/src/covariance.py` contient uniquement les opérateurs indépendants/diagonaux utilisés par les bases et l'inversion. Les imports locaux n'ont plus besoin des modules non suivis du checkout historique. Aucun fichier de `/home/flo/VarDyn` n'a été modifié.

**Décisions et comportement :** autocorrélations temporelles conservées (AR(1), ACF et fenêtres), bases de forçage par fenêtres conservées, corrélations croisées BMaux et signatures multivariées retirées. Le minimiseur block-coordinate, ses configurations, checkpoints, évaluateurs locaux et métadonnées de fenêtres sont supprimés ; les minimiseurs SciPy et Optax existants restent utilisables. Les contrôles enregistrés par l'ancien mode bloc ne sont plus pris en charge par ce mode supprimé.

Les frontières caractéristiques sont disponibles sur x et y (masques humides, caractéristiques entrantes, flux, intégrateurs sans/avec traceurs) via `open_boundary_x`/`open_boundary_y`, fermées par défaut. La divergence impose un flux nul sur les faces extérieures fermées, y compris lorsqu'un autre axe est ouvert. Les cartes auxiliaires peuvent fournir g_prime en configurant son nom dans `name_var_params_aux` ; elles alimentent la fermeture physique, les conversions SSH/interface, la CFL, l'affichage et les sorties. La transformation log commune conserve les planchers/plafonds H et Hwind et les cas particuliers H/c.

**Validation :** 31 tests du checkout réussis, puis deux tests supplémentaires ciblés réussis (intégration frontières y/xy avec traceur constant et variance/adjoint de la base calibrée), sans chargement de sources historiques. Couverture : bases/AR(1), géométrie et masse, RMS, opérateurs et gradients SW Python/scan/différences finies, frontières et budget de flux, carte g_prime/SSH/c, contrôles-log, reprises et refus du minimiseur supprimé. Syntaxe, py_compile de mod.py/sw.py et git diff --check réussis. Logs, PID, invocations et résultats : `runs/review_changes_20261109/`. Tests CPU ; aucune nouvelle OSE annuelle ni validation GPU. Le correctif L4 de main, hors annotations traitées lors de cette validation historique, est maintenant conservé par la fusion dans VarDyn_Tropics.

**Trois défauts de revue corrigés le 2026-11-09 :** le calcul de profondeur du vent reçoit aussi g_prime_control dans les intégrateurs avec/sans traceurs, y compris lorsque le plancher H*g_prime couple les deux contrôles. La gravité de référence QGSW résolue est portée par l'état (et conservée par copie) pour valider les calibrations contre les champs réellement utilisés, y compris cartes ou diagnostic H/c ; les calibrations incompatibles restent refusées. Les concentrations extérieures des traceurs sont utilisées sur les faces ouvertes entrantes, les valeurs intérieures sur les faces sortantes ; sans concentration extérieure, l'extrapolation précédente est conservée. La donnée c_b traverse les étapes d'intégration et fonctionne sans éponge. Quatre régressions ajoutées ; suite complète **37 tests CPU réussis**, imports locaux autonomes, syntaxe/py_compile et git diff --check réussis. Tests des quatre faces, quatre schémas, plusieurs traceurs, intégration/JIT/gradient de bord, référence diagnostiquée/cartographiée, copie d'état et cohérence profondeur/gradient du vent. Logs/PID : `runs/review_fixes_20261109/`. Pas de nouvelle OSE annuelle ni de commit/push.

**Suite :** rattacher les configurations/données des expériences originales puis préparer un pilote tropical avant de reproduire les slides. Les sources et pilotes sont présents et autonomes ; les tests synthétiques CPU ne reproduisent pas les OSE annuelles. La fusion est enregistrée localement, sans push.

Les spectres altimétriques locaux restent disponibles ; les expériences d'assimilation des slides sont rapportées par leur source, sans reproduction dans ce worktree.

Réplication spectrale Atlantique terminée (2026-11-09), à la demande de l'utilisateur, dans `runs/tropical_atlantic_spectrum_20261109/` : **6°S–6°N, 44°W–10°E**, mêmes dates, huit missions et trois variables SLA/LWE/SLA−LWE que dans le Pacifique. Le domaine est la plus longue bande contiguë satisfaisant >=90 % de couverture par bande de latitude et >=95 % en moyenne, après extraction large 60°W–16°E ; ces critères évitent les colonnes entières non observées mais ne constituent pas un masque terrestre. Les bandes côtières peuvent ne représenter que leur portion marine. Même grille de 2 jours × 2° × 4° et même méthode Welch ; **97,4283 %** de cellules observées, 752 trous interpolés localement. Pas spectral zonal **0,018519 degré⁻¹**, plus grossier que dans le Pacifique du fait des 54° de largeur. Variances après traitement : SLA 7,850020 cm², LWE 0,813764 cm², SLA−LWE 9,321354 cm². Parseval, linéarité, signe est/ouest et réouverture NetCDF vérifiés ; figures en anglais inspectées. Rapport et limites dans `report.md`, données `spectra_atlantic.nc`. Les modifications locales VarDyn sur `Dev-SW` sont préservées ; aucune identification de mode ni validation des hautes fréquences revendiquée.

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
avec `main` sont consignées dans `journal.md` ; elles ne décrivent plus l'état
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

- `context.md` décrit les objectifs, le périmètre et les choix structurants du projet. Le mettre à jour lorsque ces éléments évoluent.
- `journal.md` consigne chronologiquement les avancées, décisions, expériences, résultats et difficultés. Ajouter une entrée datée à chaque avancée significative et conserver l'historique.

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
