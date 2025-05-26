# ELMY_ENS_Data_Challenge

Ce dépôt GitHub présente ma participation au [Challenge ENS × ELMY](https://challengedata.ens.fr/participants/challenges/140/), visant à prédire le spread **Spot/Intraday** sur le marché de l’électricité à l’aide de différentes méthodes de machine learning, tout en m’appuyant sur les connaissances métiers acquises dans le secteur de l’énergie.

## Objectif de ce projet

Déterminer le **sens du spread** est une information critique pour la prise de décision. Le marché *Day-Ahead* (ici abusivement appelé *Spot*) fixe le prix de l’électricité livrée 24h plus tard, heure par heure, à partir d’un *clearing price* basé sur les ordres agrégés déposés avant 12h. À partir de 15h, s’ouvre le marché *Intraday*, permettant d’ajuster en temps quasi réel sa balance d’électricité, jusqu’à 5 minutes avant livraison.

L’objectif est donc d’anticiper si le prix Intraday sera supérieur ou inférieur au prix Spot, afin d’optimiser l’arbitrage d’achat entre ces deux marchés.

La variable cible `spot_id_delta` peut être modélisée soit par une **régression** (prédiction de la valeur exacte du spread), soit par une **classification** (prédiction de son signe ou de sa classe).\
Cependant, dans une logique d’analyste énergie — et au-delà du challenge académique — je ferai le choix de me concentrer principalement sur l’approche **régressive**, plus représentative d’un contexte métier.

## Étapes de ce projet

### 1. Construction de la pipeline

*Une EDA rigoureuse permettra de diagnostiquer et corriger les problématiques de distribution, de valeurs manquantes et de données aberrantes avant modélisation.*

#### 1.a : Encodage des données

Les variables catégorielles ou cycliques (issues de `DELIVERY_START`) seront atomisée et potentiellement encodée via des stratégies adaptées : - `OneHotEncoder` pour les types simples, - Encodage sinusoïdal `sin/cos` pour modéliser des cycles, des saisonalités.

La première version de ce projet consistera en la mise en place d'un modèle à arbre tel que l'**XGBoost** qu'on essaiera d'optimiser avec ***Optuna***. Ce modèle étant plutôt robuste face aux variables qu'on lui fournit, il ne sera pas nécessaire d'encoder. Si un encodage a été fait, c'est que je suis passé à un autre modèle, potentiellement plus performant.

#### 1.b : Gestion des valeurs manquantes

Plusieurs variables du jeu de données présentent un taux élevé de valeurs manquantes. C’est le cas de `predicted_spot_price` (\~82%), mais aussi de `solar_power_forecasts_std` (\~22%) et `load_forecast` (\~12%).\
L’imputation sera adaptée à chaque cas, à l’aide de méthodes simples (moyenne, médiane) ou plus avancées (`KNNImputer`, `IterativeImputer`), selon la structure et la corrélation des variables.

#### 1.c : Gestion des données aberrantes

Des valeurs aberrantes peuvent exister dans ce jeu de données, de manière **univariée ou multivariée**.\
Étant donné le rôle fondamental de la **saisonnalité** dans la consommation et la production électrique (solaire, éolien), le champ `DELIVERY_START` sera décomposé en variables temporelles (heure, jour, mois, saison, heure de pointe, etc.).\
La détection d’anomalies multivariées s’appuiera notamment sur un modèle de type `Isolation Forest`.

#### 1.d : Scaling ou Normalisation

Les variables numériques seront standardisées ou normalisées selon leur distribution. Plusieurs approches seront testées :\
`StandardScaler`, `MinMaxScaler`, `RobustScaler`, `PowerTransformer` ou `Box-Cox`, en fonction de la sensibilité aux outliers et à la distribution (symétrique ou non).

#### 1.e : Cross-Validation

Pour respecter la structure temporelle des données, la validation croisée s’effectuera via `TimeSeriesSplit`. Cette méthode garantit que l’entraînement se fait toujours sur le passé, et la validation sur le futur, simulant des conditions réelles de prédiction.

#### 1.f : Matrice de corrélation

Des matrices de corrélation seront affichées afin : - d’identifier les redondances ou co-dépendances fortes, - de guider la sélection de variables, - et de vérifier la pertinence des variables explicatives vis-à-vis de la cible.
