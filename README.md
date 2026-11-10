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

### Configuration, coût conjoint et sorties LWE MIOST — étapes 1–5

Les paramètres propres aux missions sont déclarés dans les blocs `OBS`
de type `OBS_SSH_NADIR` : `lwe_enabled` (désactivé par défaut), `sad_path`,
`path_lwe` et `name_var_lwe` (coordonnées et variable de la carte std).
Les paramètres communs appartiennent au nouvel opérateur `OBSOP_INTERP_LWE_L3`.
`exp.py` et son mécanisme de fusion restent inchangés.

```python
C2N = dict(
    super='OBS_SSH_NADIR',
    # Ajouter path, name_time, name_lon, name_lat et les erreurs de mesure.
    name_var={'SSH': ['sla_unfiltered', 'lwe']},
    combine_var={'SSH': [1, -1]},
    lwe_enabled=True,
    sad_path='/path/to/miost_sad',
    path_lwe='std_lwe_c2n_v2024.zarr.zip', # nom explicite propre à cette mission
    # name_var_lwe={'lon': 'lon', 'lat': 'lat', 'var': 'std_lwe'},
)
NAME_OBSOP = 'SSH_LWE'
SSH_LWE = dict(
    super='OBSOP_INTERP_LWE_L3',
    name_obs=['C2N'],
    facQ_file='/path/to/miost_sad/facQ_lwe.zarr.zip',
)
```

Seule la convention **avant correction DUACS** est admise :
`sla_unfiltered - lwe`, via `name_var`/`combine_var`. Le chargeur existant
évalue cette expression ; aucune correction supplémentaire n'est soustraite.
L'ajout éventuel de MDT reste indépendant. Employer des caches d'observations
propres à cette expression, comme pour les expériences SLA−LWE existantes.

Les valeurs communes reprennent MIOST : `tdec_lw=0.001` **jour**, demi-largeur
86,4 s ; `facnlt=4`, soit un espacement de 21,6 s ; `gap_seconds=600` et
`min_segment_size=7`. Fenêtres cos² compactes non renormalisées, terme nadir
`p_param=0`. La fenêtre cos² et la convention avant correction DUACS sont
fixées, sans options `window` ou `input_convention`. Aucun paramètre de temps n'est exprimé en pas du modèle.

Chaque source fournit explicitement le fichier `path_lwe`. Un chemin relatif
est joint au répertoire `sad_path` de cette source, lorsqu'il est renseigné ;
un chemin absolu est utilisé directement. Aucun nom n'est déduit de la mission.
L'opérateur reçoit le chemin explicite `facQ_file` de la carte commune ;
`name_var_facQ` vaut `{'lon':'lon','lat':'lat','var':'facQ'}`.
Les écarts-types sont en mètres de SSH et facQ est sans dimension ; le prior
attendu est `Q = facQ * std_lwe**2`, distinct du bruit des observations.
La validation et la résolution des chemins se font dans
`Obsop_interp_lwe_l3.resolve_config(config)`, pour les sources sélectionnées
par `name_obs`. Le contenu scientifique des cartes sera contrôlé lors du
raccordement de la projection.

Dans les auxiliaires MIOST locaux, les fichiers std_lwe utilisent les labels
`alg`, `c2n`, `h2b`, `j3n`, `s3a`, `s3b`, `s6a` et `swon`.
C2N est un alias de C2 ; J3N, S6A et SWON réutilisent la carte J3.
Ces alias ne représentent pas des calibrations indépendantes. Le choix du
fichier reste explicite dans chaque source VarDyn.

Le constructeur standard et le factory préparent désormais le coût conjoint,
y compris dans `Obsop_multi`. `prepare_joint` reste utilisable directement.
Un opérateur L3 ordinaire refuse une source sélectionnée avec `lwe_enabled=True`.
Les expériences existantes restent exécutables avec LWE désactivé. La validation
scientifique par un pilote court puis une OSE comparative appartient à l’étape 6.

L’étape 2 fournit une préparation explicite sans inversion :

```python
from src.obs import Obs
from src.obsop import Obsop_interp_lwe_l3

observations = Obs(cfg, ocean)
geometry = Obsop_interp_lwe_l3.prepare_geometry(cfg, ocean, observations, model)
# geometry.source_names : table des missions (clés des blocs OBS)
# geometry.native_geometry[t] : lignes réelles du checkpoint modèle t
# geometry.native_geometry_arr : tableaux hôte complétés + masque "valid"
```

Chaque ligne conserve `time_ns` (int64, nanosecondes depuis l’époque Unix),
`source_id`, `row_id`, `segment_id`, `lwe_eligible`, `lon` et `lat`.
Pour une mission LWE active, `row_id` est l’index dans son jeu d’entrée signé,
avant sélection spatiale et découpage temporel ; le couple mission/ligne
identifie l’observation. Les métadonnées restent sur CPU NumPy, sans passage
par les flottants JAX. Les fichiers NetCDF préparés gardent `lwe_time_ns`,
`lwe_row_id`, le nom de source et les signatures de provenance.

La date du checkpoint sélectionne l’état océanique ; elle ne remplace jamais
le temps natif. Les lots LWE utilisent `[checkpoint − Δt/2, checkpoint + Δt/2[` :
un point exactement au milieu appartient uniquement au checkpoint suivant.
Deux observations distinctes au même temps sont conservées. `subsampling`
est limité à `None` ou `1` lorsque `lwe_enabled=True`.

La préparation de l’opérateur réunit tous les lots retenus d’une mission
avant de définir les segments sur les temps uniques triés : séparation
strictement au-delà de `gap_seconds`, même entre fichiers/checkpoints.
`min_segment_size` compte les temps uniques. Les observations des segments
courts restent présentes, avec `lwe_eligible=False` ; aucune base LWE n’est
construite par `prepare_geometry`. Les missions désactivées ont `segment_id=-1`.

Les caches LWE préparés, par checkpoint et empilés ont un espace distinct et
un schéma versionné. Les signatures couvrent les définitions des sources,
l’inventaire des fichiers (chemin, taille, mtime), les sélections et, pour H,
les coordonnées/masques de grille, les checkpoints et les paramètres de
l’opérateur. Les jeux fournis directement en mémoire sont aussi empreintés.
Les caches d’opérateur valident dimensions, identité et empreinte des tableaux,
y compris la géométrie et le masque de padding. Une géométrie préparée absente
ou obsolète est reconstruite depuis les sources ; aucun temps natif n’est
déduit du checkpoint. `prepare_geometry` expose uniquement la préparation :
un appel de coût depuis cette préparation reste refusé. Le constructeur
standard prépare également la projection et le coût décrits ci-dessous.

L’étape 3 expose une préparation séparée de la projection, sans inversion :

```python
operator = Obsop_interp_lwe_l3.prepare_projection(cfg, ocean, observations, model)
z = np.zeros(operator.n_lwe)  # coefficients blanchis, sans unité
correction = operator.project_lwe(z)  # mètres SSH, tableau complété comme varobs_arr
gradient = operator.adjoint_lwe(residual)  # même forme que z
# Vues par checkpoint : correction réelle, sans padding ; adjoint global.
correction_t = operator.project_lwe(z, t=operator.t_obs[0])
gradient_t = operator.adjoint_lwe(residual_t, t=operator.t_obs[0])
```

`project_lwe(z)` applique `L sqrt(Q) z` ; `adjoint_lwe(r)` applique
`sqrt(Q) L.T r`. Ces deux fonctions sont compatibles avec JIT et autodiff JAX.
Elles n’appliquent pas les erreurs d’observation : la division par sigma et le
prior `0.5 * ||z||²` appartiennent au coût conjoint (étape 4). Les lignes de padding,
missions désactivées et segments courts contribuent zéro. Chaque couple
mission/segment possède ses coefficients ; aucune fenêtre ne contribue à un
autre segment, même si son support temporel le recouvre.

`lwe_matrix` est la matrice CSR hôte **L non blanchie**, dont les lignes sont les
lignes réelles concaténées dans l’ordre des checkpoints (sans padding).
`lwe_sqrtq` et `lwe_coefficients` exposent la racine du prior, les temps des
centres (nanosecondes Unix), leurs coordonnées, source, segment et variance.
Les calculs temporels utilisent des différences locales depuis les temps natifs.
Le placement suit MIOST : phase fixe au 31 août 1982 (jour 11930 depuis 1950),
extension historique de trois espacements et sélection avec quatre espacements.
Ce dernier seuil égale la demi-largeur de la fenêtre uniquement si `facnlt=4`.
Les fenêtres cos² ne sont pas renormalisées.

Les positions des centres sont interpolées/extrapolées linéairement le long de
chaque trace, après déroulement des longitudes, puis longitude modulo 360 et
latitude bornée à ±90°, comme dans le `lwe.py` fourni. La première occurrence de
chaque temps unique définit cette géométrie ; toutes les observations, y compris
les doublons temporels, conservent leur propre ligne de projection.

Les cartes NetCDF ou Zarr v2 ZIP MIOST sont interpolées bilinéairement. La lecture
ZIP accepte les attributs `dimensions` et `_ARRAY_DIMENSIONS`, les chunks
compressés et les masques optionnels, en préservant les zéros physiques. Une
carte globale est périodique en longitude ; une carte régionale n’est pas
extrapolée. Un centre hors domaine ou une valeur std_lwe/facQ négative ou non
finie produit une erreur précisant source, carte et centre. Une valeur nulle
retire le coefficient. Les cartes ne sont pas renormalisées selon la densité
des fenêtres : `Q = facQ * std_lwe**2`.

Le cache de projection est séparé/versionné/signé par la géométrie, la
configuration et le SHA-256 du contenu des cartes. Son intégrité et les identités
des lignes/coefficients sont vérifiées ; un cache absent ou abîmé est reconstruit.
Les signatures sont vérifiées même à chaud ; les cartes ne sont pas décodées
lorsque le cache est valide. Aucun import runtime du paquet MIOST.

L’étape 4 prépare explicitement un opérateur pour évaluer le coût et les gradients :

```python
operator = Obsop_interp_lwe_l3.prepare_joint(cfg, ocean, observations, model)
# Fournir operator comme H à Variational, avec les bases/prior océaniques habituels.
# evaluator.X0 : initialisation nulle du contrôle optimisé complet.
# evaluator.ocean_size : nombre de contrôles océaniques.
# evaluator.control_size : nombre total, incluant evaluator.lwe.size.
# value, gradient = evaluator.cost_and_grad(control)
```

Le vecteur est `[contrôles océaniques, z_lwe]`, tous optimisés conjointement.
`LweControl` dans obsop.py attribue les blocs LWE dans l’ordre des opérateurs,
avec des offsets explicites ; les contrôles océaniques conservent leurs indices.
Seul le préfixe océanique passe dans B et les bases physiques. Le suffixe LWE
est toujours blanchi par son propre prior et reçoit `0.5 * ||z_lwe||²` une seule
fois, quel que soit `prec` ou `sigma_B` pour l’océan.

La correction `L sqrt(Q_lwe) z_lwe` est ajoutée **en mètres SSH**, après la
conversion de hauteur océanique, avant division par `errobs` :
`r = (H(x) + L sqrt(Q_lwe) z_lwe - y) / errobs`.
`sigma_noise` par satellite alimente `errobs` en l’absence d’un champ d’erreur
configuré. `INV.sigma_R` est une covariance supplémentaire sur ces résidus
déjà normalisés ; la laisser à `None` (identité) pour utiliser seulement les
erreurs par satellite. Coût et adjoints Python/scan appliquent le même R inverse.
Le gradient LWE est `z_lwe + sqrt(Q_lwe) L.T ((R.inv(r)) / errobs)`.

Dans le mode LWE actif, le résidu océanique, la correction normalisée et son
adjoint utilisent le même masque : ligne native réelle, observation finie et
erreur finie strictement positive. Les lignes rejetées par `varmax` ou avec
valeurs manquantes ne créent aucune contrainte sur le LWE. Leurs identités et
leur correction physique restent disponibles dans les diagnostics, avec
`assimilated=0` et des résidus NaN.

Les projections complétées sont calculées une fois par évaluation, puis
sélectionnées par indices JAX aux checkpoints. Les transposes creuses utilisent
le temps tracé sans dictionnaire Python. L’ordre de concaténation multi-opérateurs,
les temps absents et les lignes de padding sont respectés. Les blocs de variance
nulle n’ajoutent aucun coefficient ; un opérateur désactivé garde le contrôle ordinaire.

Dans le chemin SW local, `JointControl(..., observations=operator)` ajoute les
blocs après les contrôles de forçage et de physique ; `initial()` initialise le
suffixe à zéro, `decode()` conserve sa sortie océanique et `lwe_coefficients(z)`
extrait le suffixe. `make_daily_residual` accepte un troisième argument explicite
`z_lwe` et ajoute la correction avant la sélection des jours. `control.bind_residual`
relie ces deux interfaces en un résidu fonction du contrôle complet ; la fonction
coût canonique conserve son unique prior identité sur tout le vecteur.
`return_states=True` fournit la trajectoire physique sans argument LWE.

L’étape 5 raccorde `Inv_4Dvar` (SciPy et les deux chemins Optax) et le driver
local `joint_optax_map` aux sorties et reprises signées. Aucun état océanique
ne contient le LWE.

`Xres.nc` contient dans `res` l’analyse océanique dans les coordonnées
historiques, puis le suffixe **blanchi** `z_lwe`. `X_it.nc` contient le vecteur
du minimiseur. Coût, export et reprise partagent la même conversion ; pour
l’évaluateur sans prior océanique (`B=None`), la convention historique est
`X = X0 - Xb` et son inverse est `X0 = X + Xb`, indépendamment de `prec`.
Les attributs `lwe_control_layout`, `control_coordinates` et
`lwe_ocean_signature` décrivent blocs/offsets, sources, signatures des
observations/projection/cartes, coordonnées et prior océanique ; les checkpoints
précisent aussi le préconditionnement de l’océan. La lecture refuse les fichiers
non signés, les dispositions ou conventions différentes, les coefficients non
finis, les changements de données/erreurs/cartes ou d’ordre des opérateurs,
y compris à nombre de coefficients identique. Les métadonnées BMaux restent
vérifiées sur le préfixe océanique. Un contrôle joint s’utilise en initialisation
ou reprise ; il est refusé comme `path_background` pour ne pas déplacer le
prior LWE centré en zéro. Les anciennes reprises sans LWE gardent leur format.

`lwe_diagnostics.nc`, écrit atomiquement dans le dossier des contrôles, est
séparé des sorties physiques. Il contient deux dimensions sans padding :

- `observation_row` : `time_ns` int64 exact (ns Unix), `source_name`, `source_id`,
  `row_id`, `segment_id`, `operator_id`, coordonnées et checkpoint, observation
  préparée et sigma, `lwe_error=e`, `lwe_correction=-e`, `residual_before=H(x)-y`,
  `residual_after=H(x)+e-y` en mètres, résidu après normalisation par sigma et
  masque `assimilated` ;
- `coefficient` : `coefficient_z_lwe`, amplitude physique `sqrt(Q)*z`, variance,
  temps/position du centre, identités mission/segment/opérateur et index dans
  le contrôle complet.

Les résidus suivent exactement les checkpoints du coût. Les lignes exclues ou
absentes de la trajectoire évaluée restent identifiables, avec `assimilated=0`
et résidus NaN. Leur correction projetée reste disponible. Les sources
désactivées et segments courts ont une correction nulle. Les temps natifs ne
passent jamais par JAX float32. Le champ `observation` suit les transformations
configurées, dont MDT et retrait de frontière en mode anomalie ; la correction
DUACS n’est pas soustraite une seconde fois.

Dans le chemin local, utiliser `control.bind_residual(raw_residual)` : cette
fonction transmet disposition signée, sélection journalière et export natif au
driver. `map_checkpoint.npz` conserve le vecteur joint blanchi, et
`map_preconditioner.npz` ses coordonnées numériques ; les deux portent la même
disposition et la signature de provenance fournie au driver. Cette signature
reste la responsabilité du pilote pour ses bases et sa physique. `resume=True`
ou `restart_from` vérifient ces métadonnées pour LWE. `maxiter=0` exige un
checkpoint et rejoue sans optimisation ; son statut de convergence est conservé.
Un arrêt non convergé peut aussi exporter les diagnostics sans être déclaré
convergé. Le pilote scientifique appartient à l’étape 6.

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

Pour BMaux, renseigner un chemin de fond ne change plus la grille de bandes.
La disposition bornée actuelle est conservée, sauf indication explicite d’une
disposition historique dans les métadonnées du fichier. Les nouveaux fichiers
`Xres.nc` et `X_it.nc` portent `bmaux_control_layouts` : disposition, signature
des fonctions spatiales/temporelles, variables ciblées, tailles et offsets des
composantes. Le chargement d’un fond, d’une initialisation ou d’une reprise
vérifie ces informations avant d’affecter les coefficients, y compris dans
une base composée. Un fond propre à une composante est extrait par ses
offsets validés, sans sélection implicite des derniers coefficients.
Les anciens fichiers BMaux sans ces métadonnées sont refusés explicitement :
leur compatibilité doit être établie avant migration ; aucun format n’est
déduit de la seule présence du chemin ou du seul nombre de coefficients.

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
