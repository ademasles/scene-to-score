# Plan du projet scene-to-score

Version du 15 octobre 2026. Hypothèse de travail : six semaines, trois ou quatre personnes. À
recaler dès que la date de rendu et la taille du groupe sont fixées.

## La question

À partir de la représentation d'une scène de film, peut-on prédire celle de sa musique assez bien
pour la retrouver parmi d'autres, et pour en générer une qui lui ressemble ?

Une même scène accepte plusieurs musiques. Une prédiction naïve renvoie la musique moyenne. Le
projet compare les façons d'éviter cet effondrement.

## La chaîne

1. Extrait de film, découpé en segments de 10 s.
2. Encodeur vidéo gelé : un encodeur d'images appliqué image par image (DINOv2), et V-JEPA 2.
3. Projection entraînée vers l'espace musical. Trois objectifs comparés : régression,
   contrastif, et en option un petit modèle génératif sur les vecteurs.
4. Vecteur musical CLAP, celui du modèle CLAP livré avec MusicLDM.
5. MusicLDM gelé, qui génère 10 s d'audio à partir du vecteur prédit.

Référence sans entraînement : décrire la scène en une phrase, puis donner cette phrase au même
générateur.

## Où tourne quoi

| Machine | Usage |
|---|---|
| Portable (RTX 4070, 8 Go ; environ 30 Go libres sous WSL) | Développement et tests sur 20 à 30 films |
| Tech lab (jusqu'à 48 Go de mémoire GPU) | Téléchargement complet (180 Go), extraction complète, entraînements à plusieurs graines, génération |

Le code est identique sur les deux machines. Seuls changent l'emplacement des données
(`STS_DATA_DIR`) et la taille des lots.

À préciser sur le tech lab avant la semaine 2 :

- système et ordonnanceur de tâches (accès direct en SSH ou file d'attente) ;
- quota disque, pour les 180 Go de données et quelques gigaoctets de caractéristiques ;
- accès à internet depuis les machines de calcul, pour Hugging Face et uv ;
- modèle exact de la carte.

## Calendrier

| Semaine | Tâches | Machine | Jalon |
|---|---|---|---|
| 1 | Socle du dépôt et téléchargement d'un sous-ensemble (faits) ; écoute de 50 extraits ; environnement sur le tech lab | Portable, tech lab | La piste audio contient surtout de la musique |
| 2 | Extraction des vecteurs vidéo et musique, mesure du temps par segment ; référence par le texte | Portable puis tech lab | Vecteurs en cache pour tout le jeu |
| 3 | Projections par régression et contrastive ; évaluation par le rang ; diagnostic d'effondrement | Tech lab | Premier tableau objectifs × encodeurs |
| 4 | Branchement sur MusicLDM ; génération sur le jeu de test ; début du rapport | Tech lab | Démo, première version |
| 5 | Évaluation complète à trois graines ; test hors domaine sur V2M-bench ; test d'écoute | Tech lab | Toutes les figures |
| 6 | Rapport, soutenance, démo finale, nettoyage du dépôt | Portable | Code gelé en début de semaine |

En cas de retard, couper dans cet ordre : le modèle génératif sur les vecteurs, le test
d'écoute, le test hors domaine. Le tableau objectifs × encodeurs ne se coupe pas.

## Répartition, à attribuer

| Rôle | Contenu |
|---|---|
| Données et évaluation | Sous-ensembles, écoute de contrôle, métriques de rang et de dispersion, test d'écoute |
| Encodeurs et extraction | DINOv2, V-JEPA 2, CLAP, mise en cache, mesures de temps |
| Projection et entraînement | Objectifs, sélection de modèle, balayages au tech lab |
| Génération et démo | MusicLDM, référence par le texte, démo |

Tout le monde écrit le rapport à partir de la semaine 4.

## Calcul estimé

Estimations, pas des mesures, pour tout OSSL-v2 (environ 85 000 segments de 10 s) sur le
portable. Le tech lab va plusieurs fois plus vite.

| Étape | Temps estimé |
|---|---|
| Encodeur d'images (1 image par seconde) | 1 à 2 h |
| V-JEPA 2 ViT-L (16 images par segment) | 2 à 4 h |
| CLAP sur la musique | moins de 30 min |
| 18 entraînements de projection (3 objectifs × 2 encodeurs × 3 graines) | 2 à 5 h au total |
| Génération d'environ 1 000 extraits | 1 à 4 h |

Les goulots sont le disque et la bande passante, pas le GPU. Mesurer le temps par segment sur
une centaine de segments en semaine 2, puis extrapoler.

## Notes techniques relevées

- **Identifiants d'extraits** : début et fin en centièmes de seconde, vérifié sur un extrait.
- **Audio d'OSSL-v2** : mono, 44,1 kHz. Le CLAP de MusicLDM attend du 48 kHz : rééchantillonner.
- **MusicLDM** : le pipeline est rangé dans `diffusers.pipelines.deprecated` (diffusers 0.41).
  Avec transformers 5, `ClapModel.get_text_features` renvoie un objet de sortie
  (`.pooler_output`) et non un tenseur, ce que le pipeline n'attend pas. Il faut lui passer
  soi-même `prompt_embeds` et `negative_prompt_embeds`, ce dernier étant le vecteur du texte
  vide. Constaté à la lecture du code des versions installées, pas encore exécuté.
- **Vecteurs CLAP** : texte et audio sortent normalisés (norme 1) dans transformers 5.
- **V-JEPA 2** : `get_vision_features(pixel_values_videos)` dans transformers. Moyenner les
  jetons sur l'espace avant de les stocker : environ 4 Mo par segment sinon, contre 16 Ko.
- **Décodage vidéo** : ffmpeg n'est pas installé sous WSL. PyAV embarque ses propres
  bibliothèques et évite d'en dépendre.
- **Segments d'un même extrait** : ils partagent leur musique. À l'évaluation, n'en garder
  qu'un par extrait ; à l'entraînement contrastif, ils créent de faux négatifs.

## Risques

1. Films anciens (62 % des années 1930 et 1940) : le transfert vers des vidéos récentes est
   incertain. Le test sur V2M-bench le mesure.
2. La musique séparée garde des sons non musicaux, selon les auteurs du jeu.
3. La régression s'effondrera probablement vers la moyenne : c'est l'objet de la comparaison.
4. MusicLDM a appris sur de la musique récente ; la mesure par le rang n'en dépend pas.
5. La similarité entre musiques est calculée dans l'espace qui sert à prédire ; le test
   d'écoute sert de contrôle indépendant.

## Terminé quand

- Le tableau objectifs × encodeurs existe, sur OSSL-v2 et sur V2M-bench, à trois graines.
- La démo tourne : une scène, la musique produite, la musique d'origine et celle de la
  référence par le texte.
- Le dépôt se relance de zéro avec les commandes du README.
- Le rapport fait quatre à six pages.
