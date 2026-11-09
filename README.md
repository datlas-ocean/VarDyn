# VarDyn Tropics — branche `VarDyn_Tropics`

Les documents de référence à lire à chaque nouveau prompt sont [AGENTS.md](AGENTS.md),
[CONTEXT.md](CONTEXT.md), [README.md](README.md) et [JOURNAL.md](JOURNAL.md).

VarDyn reconstruit la hauteur de surface de la mer (SSH) et les courants en
combinant des observations, un modèle dynamique et une assimilation
variationnelle dans un espace de contrôle réduit. Cette branche développe
particulièrement la reconstruction tropicale avec le modèle shallow-water
(SW) à gravité réduite, le vent et des corrections apprises sur GLORYS.
Elle conserve les autres modèles, observations et outils de cartographie de
VarDyn.

Ce README décrit le code de cette branche et son organisation après la fusion
de `Dev-SW`. Les expériences de `/home/flo/VarDyn` servent d'exemples de
protocoles scientifiques ; leurs configurations, données et résultats ne sont
pas tous inclus dans ce dépôt.

## 1. Ce qui est disponible dans cette branche

| Ensemble | Disponibilité | Usage |
| --- | --- | --- |
| Cœur dynamique et 4DVar | Versionné dans `mapping/src/` et `mapping/models/` | Intégration, assimilation, bases réduites, contrôles physiques, sorties et diagnostics |
| Bases de forçage SW, y compris journalière et calibrée | Versionnées dans `mapping/src/basis.py` | Appliquer un prior générique ou une calibration existante, sans dépendance aux scripts d'apprentissage |
| Pilotes d'apprentissage MAP1/MAP2 et EM | Quatre scripts locaux dans `scripts/`, ignorés par Git | Préparer et lancer un apprentissage avec une configuration et des données explicites |
| Implémentations spécialisées de l'apprentissage | Locales dans `learning_local/`, ignorées par Git | Disponibles dans ce worktree, absentes d'une nouvelle copie Git |
| Expériences et données historiques | Dans `/home/flo/VarDyn` et les chemins de données indiqués par leurs configurations | Références de protocole, préparation des entrées et comparaison scientifique |

**Les fichiers suivis dans `mapping/src/` sont les mêmes 13 fichiers que sur
`main`, mais leur contenu comporte les fonctionnalités propres à cette branche.**
L’ensemble de `mapping/` reprend l’inventaire de `main`, y compris ses trois
tests de référence ; les fichiers du code conservent les changements SW.
Le cœur n'importe aucun des 17 modules déplacés dans `learning_local/`.
Les opérateurs de covariance partagés sont dans `tools.py`.

## 2. Principe de reconstruction

Une expérience définit une grille, un état initial, une dynamique, des
observations et une base réduite. La base transforme un vecteur de coefficients
en corrections de l'état ou des paramètres du modèle. La trajectoire obtenue
est comparée aux observations aux checkpoints de la fenêtre d'analyse.

Le 4DVar minimise un coût combinant l'écart au prior et les résidus
d'observation pondérés par leurs erreurs. Les gradients utilisent les chemins
adjoints et la différentiation JAX des opérateurs compatibles. Sans bloc
d'inversion, le même ensemble permet une intégration directe.

Le code distingue :

- l'état de l'océan, par exemple SSH, `u`, `v` et les traceurs configurés ;
- les paramètres physiques, par exemple `H`, `g_prime` et `h_wind` ;
- les forçages d'erreur de modèle, corrections additives des tendances ;
- les coefficients réduits et leur prior, qui ne sont pas directement les
  champs physiques reconstruits.

## 3. Modèles dynamiques

| Configuration | Fonctionnalités |
| --- | --- |
| `MOD_DIFF` | Propagation simplifiée par diffusion ; cas statiques selon les paramètres, expériences de référence et contrôles de flux |
| `MOD_QG1L` | Modèle quasi géostrophique à une couche et demie, évolution de SSH et courants diagnostiqués ; variantes du noyau QG selon la configuration |
| `MOD_QGSW` | Interface SW monocouche pour SSH, vitesses sur grille C, vent, dissipation, forçages et traceurs passifs |
| `MOD_CSW1L` | Modèle d'ondes internes SW, profondeurs équivalentes et contrôles de conditions aux limites de marée interne |
| `MOD_BMIT` | Composition d'un modèle de mouvement équilibré et d'un modèle de marée interne, avec couplages configurables |
| Composition de modèles | Plusieurs blocs partageant un état et une configuration d'expérience |

Les noyaux se trouvent dans `mapping/models/model_qg1l/`, `model_sw1l/` et
`model_qgsw/`. Certaines possibilités internes de ces noyaux dépassent
l'interface publique. En particulier, **`Model_qgsw` exige actuellement `nl=1`** :
la présence de routines multicouches ou QG projeté dans `model_qgsw/` ne rend
pas ces modes accessibles par cette interface.

### Modèle SW tropical

La dynamique représente l'advection, Coriolis, le gradient de pression,
le transport de l'épaisseur, le vent et les termes dissipatifs configurés.
Elle utilise une grille C avec cellules de hauteur, faces de vitesse et
masques côtiers. Le SW conserve Coriolis dans ses équations sans imposer une
inversion géostrophique à l'équateur.

Deux conventions de hauteur sont disponibles via `height_representation` :

- `ssh` : convention de hauteur historique ;
- `interface_displacement` : déplacement d'interface `eta`, avec conversion
  en SSH physique par `SSH = (g_prime / physical_gravity) * eta`.

Dans la seconde convention, la profondeur totale est `H + eta` et la vitesse
d'onde vérifie `c² = H * g_prime`. Les forçages de hauteur sont exprimés dans
la coordonnée du modèle : un mètre d'interface n'est pas un mètre de SSH.
La convention d'une reprise est spécifiée par `restart_height_coordinate`.

Les principales options numériques sont :

- pas fixe `dtmodel` ou sélection par `cfl` ; une CFL configurée peut remplacer
  le pas demandé, ce qui doit être vérifié dans la préparation ;
- intégrateurs `rk3`, `rk2` et `rk2_ssp` ;
- advection WENO, variantes linéaires upwind et flux diffusifs selon la
  variable ;
- viscosité, diffusion de hauteur, diffusion des traceurs et frottement de fond ;
- traitement des masques, du glissement latéral et des éponges côtières ;
- traitement direct des forçages ou correction de quantité de mouvement
  associée à une source de masse via `forcing_momentum`.

### Vent, frontières et traceurs

Le vent peut être prescrit au noyau ou préparé depuis `u10`/`v10` NetCDF.
La conversion utilise une formule de tension de vent, puis une accélération
proportionnelle à `tau / (rho_water * h_wind)`. Les options de densité, de
coefficient de traînée et de cadence `wind_timestep` sont configurables.

`h_wind` est la profondeur de distribution du vent. Elle peut être distincte
de la profondeur dynamique `H`, prescrite ou contrôlée. Le choix d'utiliser
l'épaisseur instantanée est explicite ; il ne remplace pas un modèle
prognostique de couche mélangée.

Les frontières peuvent utiliser un fichier commun ou des fichiers distincts
par variable (`bc_file`, `bc_files`), avec interpolation spatiale et temporelle.
Le SW monocouche propose des frontières fermées ou caractéristiques sur les
deux axes : `open_boundary_x` et `open_boundary_y`. Les masques des faces humides,
les caractéristiques entrantes, les flux et les éponges doivent être cohérents.
Les combinaisons incompatibles avec les anciennes options de frontière sont
rejetées par le modèle.

L'advection de traceurs passifs est disponible. Aux frontières ouvertes, les
flux utilisent la concentration extérieure sur une face entrante et la
concentration intérieure sur une face sortante. Sans donnée extérieure,
l'extrapolation historique est conservée. Cela permet des expériences avec
SST ou SSS comme traceurs, mais **ne constitue pas une thermodynamique active**
ni une représentation verticale de la couche mélangée.

## 4. Paramètres physiques et bases réduites

### Contrôles physiques SW

Les champs `H`, `g_prime`, `h_wind`, `bottom_drag_coef`, `diff_coef` et
`visc_coef` peuvent être utilisés comme paramètres contrôlés selon le problème.
Leur référence peut être scalaire, cartographiée ou issue d'une reprise.
`file_params_aux` et `name_var_params_aux` permettent de charger des cartes,
y compris de gravité réduite.

Les coordonnées logarithmiques maintiennent la positivité, avec les
planchers/plafonds propres aux paramètres. La référence physique et la
correction logarithmique sont deux quantités distinctes. La fermeture
`c² = H * g_prime` intervient dans la dynamique, la conversion SSH et la CFL.
Une option permet aussi un contrôle borné de `c` à `g_prime` fixé ; elle ne se
combine pas avec des contrôles simultanés de `H` ou `g_prime`.

Le fond de paramètres d'un état peut être réutilisé par les bases compatibles
avec `use_state_background`. Une reprise doit conserver la convention de
hauteur, les références physiques et la disposition du contrôle.

### Familles de bases

| Famille | Représentation |
| --- | --- |
| `BASIS_OFFSET` | Corrections uniformes ou partagées entre paramètres configurés |
| `BASIS_GAUSS2D` | Champs spatiaux lisses, notamment pour les paramètres physiques ; support des grilles de hauteur et de faces |
| `BASIS_GAUSS3D` | Corrections localisées en espace et en temps, avec amplitudes éventuellement issues de cartes auxiliaires |
| `BASIS_BM` / `BASIS_BMaux` | Mouvement équilibré et fonctions multirésolution, avec lecture de statistiques auxiliaires |
| `BASIS_HBC` | Contrôles d'ondes aux limites, notamment de marée interne |
| `BASIS_SW_FORCING` | Forçages SW divergents, rotationnels et de hauteur, génériques ou lus depuis une calibration |
| `BASIS_SW_DAILY` | Atomes spatiaux SW fixes et coefficients quotidiens à covariance temporelle AR(1) |
| Composition / fenêtres | Combinaison de bases et séparation des contrôles de forçage par fenêtres, avec paramètres physiques partagés |

Pour SW, la quantité de mouvement est décomposée en gradient d'un potentiel
et gradient tourné d'un autre potentiel : `F = grad(phi) + ez × grad(psi)`.
Les atomes de hauteur compensés ont une intégrale nulle par bassin ; des modes
de hauteur moyenne par bassin peuvent être ajoutés séparément.

La base générique propose un biais, une partie lente et des fenêtres
éventuellement modulées par des périodes choisies explicitement. La base
calibrée lit la géométrie, les amplitudes, les échelles temporelles et le fond
prescrit d'un artefact compatible avec la grille et la référence physique.
La base journalière interpole les coefficients entre les dates et applique
une covariance AR(1), sans construire une matrice temporelle dense.

Les fonctions spatiales, projections et transposées restent dans `basis.py`.
Les covariances diagonales et par blocs indépendants sont dans `tools.py`.
Les corrélations croisées BMaux retirées lors de la revue ne sont pas
réintroduites ; conserver l'autocorrélation temporelle ne les rétablit pas.

## 5. Observations et diagnostics

Les entrées comprennent les champs L4, l'altimétrie nadir et l'altimétrie
swath. Les noms de variables, coordonnées, missions, erreurs, corrections MDT
et options de bruit sont configurables. Le chargement sélectionne les fichiers
et observations par dates et domaine, avec réutilisation de caches.

`OBSOP_INTERP_L3` et `OBSOP_INTERP_L4` projettent les états vers les mesures ;
plusieurs opérateurs peuvent être combinés. Les conversions entre hauteur
interne et SSH physique sont prises en compte. Un opérateur L4 sans observation
est neutre dans les chemins compilés de coût et d'adjoint.

`DIAG_OSSE` compare une reconstruction à une simulation de référence.
`DIAG_OSE` permet l'évaluation avec observations réelles et produits de
référence, selon les données fournies : erreurs, cartes, séries temporelles,
diagnostics spectraux, figures et animations. Une mission exclue de
l'assimilation peut servir de validation indépendante du coût.

Les scripts spécialisés de bootstrap, scores drifters ou comparaison MIOST des
expériences historiques ne sont pas tous inclus ici. Un score rapporté dans
leurs comptes rendus doit conserver son domaine, sa période, ses corrections
altimétriques, son support commun et son statut de convergence.

## 6. Inversion, reprises et passage à l'échelle

`INV_4DVAR.minimizer` accepte trois valeurs :

| Valeur | Exécution |
| --- | --- |
| `scipy` | L-BFGS-B historique, piloté par SciPy |
| `optax-decoupled` | Contrôle, gradient et historique L-BFGS sur le périphérique JAX ; recherche de pas Armijo pilotée par Python |
| `optax-full-gpu` | Itérations et recherche Armijo dans un programme compilé ; diagnostics transférés après le calcul, sans callbacks par itération |

Les chemins Optax exigent `jit_cost_and_grad=True` et
`cost_and_grad_schedule='scan'`. La présence de `optax-full-gpu` dans le code
ne signifie pas qu'une expérience tropicale annuelle ait été validée avec ce
mode dans ce worktree. Le minimiseur block-coordinate antérieur est retiré.

La mémoire et l'arrêt sont réglés notamment par `lbfgs_history_size`, `maxiter`,
`ftol`, `gtol`, `convergence_nit`, `minimum_iterations` et
`timestep_checkpoint`. Les chemins Python et `scan` offrent des compromis
différents de compilation et de stockage. Un arrêt sur variation du coût,
un seuil de gradient atteint et un budget d'itérations épuisé doivent être
rapportés séparément.

Les contrôles sauvegardés permettent la reprise ou le rejeu d'une trajectoire.
Le rejeu avec `maxiter=0` exige un contrôle chargé ; il ne relance pas une
optimisation. Un vecteur incompatible avec la base, le prior ou la convention
d'état ne doit pas être réutilisé comme une reprise valide.

La cartographie peut être découpée en tuiles spatiales et fenêtres temporelles
chevauchantes. `run_assimilation.py` gère la préparation, les processus,
l'affectation GPU et les fusions pondérées des sorties. Ce découpage est
distinct d'une base de forçage fenêtrée au sein d'une trajectoire continue.
Les scripts [SLURM](slurm/README.md) distribuent les tuiles entre GPU et réalisent
les fusions spatiales puis temporelles.

## 7. Apprentissage SW et exemples tirés de `/home/flo/VarDyn`

### Chaîne MAP1 → statistiques locales → MAP2

Les travaux historiques ont conduit à la chaîne suivante :

1. **Préparation** : interpoler GLORYS sur la grille C ; préparer état initial,
   SSH et éventuellement U/V, vent ERA5 et conditions aux limites. Les défauts
   d'équation journaliers peuvent initialiser les amplitudes du prior.
2. **MAP1** : estimer les forçages à coefficients quotidiens et, selon la
   configuration, les champs physiques. Une trajectoire annuelle continue
   conserve l'état entre les jours.
3. **Statistiques locales** : estimer moyenne, écart-type centré et ACF de
   chaque atome spatial ; conserver les indicateurs de variance non résolue,
   de plancher et de qualité d'ajustement.
4. **MAP2 réduit** : ajuster des fenêtres temporelles compactes cos²,
   normalisées en partition de l'unité ; construire un nombre de contrôles
   temporels propre à chaque atome. L'espacement, la demi-largeur et le temps
   de passage à `1/e` de l'ACF sont des grandeurs distinctes.
5. **Reconstruction et validation** : repartir de forçages physiques nuls,
   transférer ou réoptimiser les paramètres physiques selon le pilote, puis
   comparer trajectoires, forçages, coûts d'observation et convergence.

Les statistiques empiriques des coefficients MAP ne sont pas une covariance
d'erreur de modèle corrigée de l'incertitude postérieure. Les outils locaux
conservent aussi l'ancienne approche EM avec moments postérieurs et diagnostics
CG ; elle constitue un autre protocole que le MAP1/MAP2 empirique.

Les pilotes disponibles localement sont :

- `learn_sw_daily.py` : apprentissage journalier, MAP seul, EM et diagnostics
  suivant les options ;
- `learn_sw_joint_physics.py` : MAP conjoint forçages/physique ;
- `learn_sw_reduced_joint_physics.py` : MAP2 réduit avec transfert et contrôles
  physiques actifs ;
- `learn_sw_windowed.py` : apprentissage historique par fenêtres et statistiques
  communes.

Le dossier `scripts/` et le dossier racine `tests/` sont conservés
localement et ignorés par Git ; une nouvelle copie ne contient ni ces pilotes
ni ces tests complémentaires. Les trois tests de référence de `main` restent
versionnés dans `mapping/tests/`. Les pilotes exigent `--config` et leurs modules locaux. Voir
[le guide d'apprentissage](docs/sw_learning.md). Les garde-fous de signature,
reprise et finitude font partie de ces outils ; la présence d'un checkpoint
ne prouve pas la convergence d'un résultat.

### Expériences de référence

| Travaux documentés dans `/home/flo/VarDyn` | Exemple d'usage et enseignement pour cette branche |
| --- | --- |
| SW Atlantique vent seul, été 2020 | Intégration depuis le repos, vent ERA5, masque GLORYS, diagnostics SSH/U/V et propagation. Les structures équatoriales observées ne suffisent pas à identifier un mode ni à démontrer la reproduction des TIW. |
| Apprentissage annuel GLORYS, juillet 2020–juin 2021 | Trajectoire continue, forçages multiscalaires et compression temporelle MAP2. Ne pas confondre réduction du nombre de contrôles et réduction proportionnelle du coût du modèle. |
| Apprentissage conjoint de la physique | Estimation de `H`, `g_prime`, `h_wind` et, selon l'expérience, du frottement ; contrôle de la cohérence SSH/interface, du prior et de la CFL. |
| OSE nadir 2024 | Assimilation de SSH réelle, mission AL exclue du coût, SLA MIOST aux limites et courants GLORYS. L'OSE peut réoptimiser la physique autour du MAP1 appris. |
| Comparaison des cibles de courant | Apprentissage avec courants de surface, moyenne entre `mlotst` et `mlotst + 30 m`, ou courant à `mlotst + 50 m`. Les conditions aux limites U/V doivent suivre la même définition que la cible d'apprentissage. |

Ces expériences sont **des références documentées dans le checkout historique**,
non des reproductions réalisées ici. Les pas de temps, éponges, priors et
critères d'arrêt ont évolué entre essais ; ils ne constituent pas des valeurs
par défaut interchangeables. Les cartes et spectres altimétriques de ce worktree
restent des analyses distinctes de la validation d'une assimilation.

Pour retrouver les détails sur cette machine :

- `/home/flo/VarDyn/mapping/examples/tropical_sw/README.md` ;
- `/home/flo/VarDyn/docs/sw_reduced_local_map2_20261007.md` ;
- `/home/flo/VarDyn/docs/sw_ose_miost_sla_glorys_currents_2024.md` ;
- `/home/flo/VarDyn/docs/sw_submixed_consistent_bc_ose_20261015.md`.

Ces documents contiennent aussi des états de processus datés : ils ne doivent
pas être interprétés comme le statut actuel d'un calcul.

## 8. Organisation des fichiers

```text
mapping/                Exactement les 43 fichiers présents sur main
  src/                  Cœur versionné : 13 fichiers, contenus propres à la branche
  models/               Noyaux dynamiques et opérateurs numériques
  examples/             Notebooks et configurations QG, WHIRLS et CSWM/BMIT
  aux_files/            Champs auxiliaires livrés dans le dépôt
  tests/                Trois tests de référence présents sur main
scripts/                Quatre pilotes locaux, ignorés par Git
learning_local/         17 modules locaux, ignorés par Git
tests/                  15 fichiers de tests complémentaires, ignorés par Git
slurm/                  Préparation, distribution des tuiles et fusion HPC
docs/                   Guide SW et inventaire historique de provenance
runs/                   Analyses, journaux et vérifications locales
AGENTS.md               Consignes et lecture obligatoire à chaque prompt
CONTEXT.md              Contexte Tropics et vocabulaire technique fusionnés
JOURNAL.md              Historique des travaux et validations
pyproject.toml          Métadonnées et dépendances Python déclarées
```

Les 13 modules de `src` sont `__init__.py`, `basis.py`, `config.py`,
`config_default.py`, `diag.py`, `exp.py`, `inv.py`, `mod.py`, `obs.py`,
`obsop.py`, `run_assimilation.py`, `state.py` et `tools.py`.

## 9. Environnement et exécution

Le projet déclare Python ≥ 3.10. Les vérifications locales récentes utilisent
l'environnement `/home/anaconda/envs/massh` avec Python 3.13 et JAX sur CPU.
Une installation GPU doit disposer d'une distribution JAX compatible avec le
matériel ; sa disponibilité se vérifie dans un processus frais avant un calcul.

Les dépendances comprennent NumPy, SciPy, JAX, Optax, xarray, NetCDF4, pandas,
Matplotlib, xrft et les bibliothèques déclarées dans `pyproject.toml`.
`pyinterp` est également nécessaire au cœur. L'orchestration importe
`astropy` et `cartopy` ; les sorties Zarr nécessitent `zarr` et `numcodecs`.
Ces dépendances complémentaires ne sont pas toutes déclarées dans les
métadonnées actuelles. `pytest` est nécessaire pour les tests.

Depuis la racine, rendre les modules accessibles à un environnement préparé :

```bash
export PYTHONPATH="$PWD:$PWD/mapping${PYTHONPATH:+:$PYTHONPATH}"
```

Les configurations sont des fichiers Python dont les blocs sélectionnés par
`NAME_GRID`, `NAME_MOD`, `NAME_OBS`, `NAME_OBSOP`, `NAME_BASIS`, `NAME_INV` et
`NAME_DIAG` héritent des familles `super` de `config_default.py`. Les grilles
peuvent être géographiques, cartésiennes, lues depuis un fichier ou une reprise.
Les données, masques, cartes auxiliaires et sorties doivent être explicitement
raccordés à leurs chemins locaux.

Un pilote utilisateur minimal charge une configuration complète et appelle le
cœur :

```python
from src import exp, state, mod, inv

cfg = exp.Exp('/absolute/path/to/config_experiment.py')
ocean = state.State(cfg)
model = mod.Model(cfg, ocean)
inv.Inv(cfg, State=ocean, Model=model)
```

Pour sélectionner Optax dans le bloc d'inversion d'une configuration :

```python
myINV = dict(
    super='INV_4DVAR',
    minimizer='optax-decoupled',
    jit_cost_and_grad=True,
    cost_and_grad_schedule='scan',
    lbfgs_history_size=10,
    maxiter=100,
    ftol=5e-5,
    gtol=1e-3,
    convergence_nit=2,
    minimum_iterations=5,
)
```

Ces valeurs illustrent la syntaxe ; elles ne définissent pas un protocole
scientifique validé. Pour explorer le pilote d'apprentissage sans calcul :

```bash
python scripts/learn_sw_joint_physics.py --help
```

Une fois la configuration et les données préparées, lancer les calculs en
arrière-plan avec un journal et un PID, ou utiliser SLURM :

```bash
mkdir -p runs/my_experiment
nohup env JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=0 \
  HDF5_USE_FILE_LOCKING=FALSE XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python -u /absolute/path/to/run_experiment.py \
  > runs/my_experiment/run.log 2>&1 < /dev/null &
echo $! > runs/my_experiment/run.pid
```

Choisir le GPU disponible et contrôler la fin du calcul. Prévoir d'abord un
pilote sur une durée et une grille réduites, avec vérification des épaisseurs,
forçages, gradients et sorties finies sur les points océaniques.

### Exemples livrés

- `mapping/examples/2021a_SSH_mapping_OSE/` : notebook et configuration d'OSE
  SSH Gulf Stream avec QG, à adapter aux chemins de données ;
- [WHIRLS](mapping/examples/WHIRLS/README.md) : exemples QG et SW avec SSH,
  courants et SST selon la configuration ;
- [CSWM/BMIT](mapping/examples/CSWM/README.md) : ondes internes et couplage
  avec le mouvement équilibré ;
- [SLURM](slurm/README.md) : lancement distribué, préparation et fusion.

Les configurations tropicales et leurs données du checkout historique ne sont
pas automatiquement disponibles dans ces exemples.

## 10. Sorties, validation et limites

Les états et paramètres sont sauvegardés selon la configuration en NetCDF ou
Zarr, avec gestion des reprises, écritures groupées et fusions de tuiles.
Les contrôles, dont `Xres.nc` pour les chemins concernés, et les diagnostics de
minimisation doivent être conservés avec la configuration résolue et les
références des données. Les outils locaux d'apprentissage ajoutent notamment
statistiques, métadonnées de base temporelle, checkpoints et diagnostics MAP.

Validation de la réorganisation du 9 novembre 2026 :

- **40 tests CPU du cœur réussis**, avec les imports des modules déplacés
  explicitement interdits ; huit fichiers de tests optionnels ignorés ;
- **43 tests d'apprentissage local réussis** dans un processus séparé ;
- audit statique sans dépendance du cœur vers `learning_local`, compilation de
  34 fichiers, aide des quatre CLI et contrôles Git réussis.

Les vérifications couvrent les projections et adjoints, AR(1), normalisations,
conservation de masse, frontières x/y, traceurs, contrôles physiques, gradients
contre différences finies, L4 vides, solutions MAP synthétiques et reprises.
Les rapports locaux sont dans `runs/src_main_layout_20261109/`.

Lorsque les tests locaux sont présents, pour lancer la suite dans un environnement
préparé, conserver la même convention
d'exécution détachée :

```bash
mkdir -p runs/tests
nohup env JAX_PLATFORMS=cpu PYTHONPATH="$PWD:$PWD/mapping" \
  python -m pytest tests mapping/tests -q \
  > runs/tests/pytest.log 2>&1 < /dev/null &
echo $! > runs/tests/pytest.pid
```

En l'absence de `learning_local`, les huit fichiers d'apprentissage sont ignorés.
Ces tests ne constituent ni un benchmark GPU, ni une reproduction d'un
apprentissage GLORYS annuel, ni une validation indépendante de la SSH et des
courants. Une couche mélangée dynamique, les rétroactions thermodynamiques de
SST/SSS et une validation complète des futurs protocoles tropicaux restent des
évolutions du projet. Tous les textes des figures produites doivent être en
anglais.

Voir aussi [les consignes](AGENTS.md), [le contexte et le vocabulaire](CONTEXT.md),
[le journal](JOURNAL.md) et [le guide d'apprentissage SW](docs/sw_learning.md).

Après déplacement des outils locaux à la racine, les mêmes 40 tests du cœur
et 43 tests d’apprentissage ont réussi. L’inventaire complet des 43 fichiers
de `mapping/` correspond à `main`, et leurs contenus ont été conservés pendant
le déplacement. Les quatre pilotes ont aussi été vérifiés depuis un autre
répertoire. Rapports : `runs/root_local_layout_20261109/`.
