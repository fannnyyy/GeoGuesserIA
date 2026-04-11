Abstract !

Contexte métier !
	secours 

Problématique !
    hypothèse

etat l'art
	ce qui existe déja et se fait de mieux modèle
    (dataset si y'a)

dataset
	OSV5M
		analyse

Approche théorique cf problématique !
        Stratégie Classif, Reg, ....
        métrique d'éval
        Schéma de tous les modèle a tester
        répartition tache Méthodologie des tâches ? (pt bizz : avancement pratique) + orga travail dataset

modèle
	Resnet
		architecture
		modélisation
			reg x 2
                Résultats
                    Analyse
			classif + reg
                Résultats
                    Analyse
			classif x 2
				impacte discrétisation
                Résultats
                    Analyse
        Résumé

	Module d'attention 
		Définition 
		intégration au restnet 
		Visualiosation de l'impact de l'attention GradCAM 
            Résultats !
                Analyse
        Résumé

	ViT
        Objectif de l'approche
        choix de la réprésentation
        Architecture
        Fonction de perte 
        Evaluation
        Résultat
            Analyse
        Résumé  

    StreetClip (base de comparaison cf etat de l'art dispo sur HF)
		Description test
		Résultats

    Dinov2 (regrouper des points)
        Objectif
        Principe 
        Intérêt KNN
        Construction 
        Stratégie 
        Ajustement
        Sélection
        Evaluation
        Résultat
            Analyse 
        Résumé


Analyse du tableau des différents modèles !
    Comparaison uniquement
    Positionnement 

Réponse à notre problématique + contexte métier !
    Streamlit (qui montre :)
        Réponse claire (cf meilleur modèle)
        XIA : citer vite fait gradCAM + feature couche
        Limite
            performance faible (cf analyse dataset)
    

Annexe
    Dataset abandonné 
        (Kaggle + Flair
            description + impossibilité de matching)
    
    Etat de l'art moindre

    Problème rencontré Resnet50 + CBAM

    implémentation Streamlit



             


--------------------------------------------------


**Abstract**

**1. Contexte et problématique**
- 1.1 Contexte métier — application aux secours et à la localisation d'urgence
- 1.2 Problématique et hypothèses de travail

**2. État de l'art**
- 2.1 Modèles existants et meilleures performances actuelles
- 2.2 Datasets disponibles

**3. Dataset — OSV5M**
- 3.1 Présentation et caractéristiques
- 3.2 Analyse exploratoire — land cover, road index, déséquilibre géographique

**4. Approche méthodologique**
- 4.1 Stratégies envisagées — classification, régression, approches hybrides
- 4.2 Métriques d'évaluation — Haversine, F1-score
- 4.3 Vue d'ensemble des modèles testés — schéma global
- 4.4 Répartition des tâches

**5. Modèles**

- 5.1 ResNet — architecture et modélisation
    - 5.1.1 Régression pure — ResNet18 et ResNet50
        - Résultats et analyse
    - 5.1.2 Classification pays et régression GPS conditionnée
        - Résultats et analyse
    - 5.1.3 Classification par cellules géographiques k-means
        - Impact de la discrétisation
        - Résultats et analyse
    - 5.1.4 Résumé et comparaison des variantes ResNet

- 5.2 Module d'attention — ResNet50 + CBAM
    - 5.2.1 Définition et principe de CBAM
    - 5.2.2 Intégration au ResNet50 — wrapping et curriculum
    - 5.2.3 Visualisation de l'impact de l'attention — GradCAM
    - 5.2.4 Résultats et analyse
    - 5.2.5 Résumé

- 5.3 Vision Transformer — ViT-B/16
    - 5.3.1 Objectif et choix de représentation GPS
    - 5.3.2 Architecture et fonction de perte
    - 5.3.3 Résultats et analyse
    - 5.3.4 Résumé

- 5.4 DINOv2 + KNN — approche self-supervised
    - 5.4.1 Objectif et principe
    - 5.4.2 Intérêt du KNN géographique
    - 5.4.3 Construction de la banque de features
    - 5.4.4 Stratégie de prédiction et ajustements
    - 5.4.5 Résultats et analyse
    - 5.4.6 Résumé

**6. Comparaison des modèles**
- 6.1 StreetCLIP — base de comparaison issue de l'état de l'art
    - 6.1.1 Description et protocole de test
    - 6.1.2 Résultats
- 6.2 Tableau récapitulatif des performances
- 6.3 Analyse comparative et positionnement

**7. Réponse à la problématique**
- 7.1 Application Streamlit — démonstration de la solution
    - Présentation de l'interface et des fonctionnalités
    - Démonstration sur cas réels — succès et échecs
    - Explicabilité des prédictions — GradCAM et feature maps
- 7.2 Limites et perspectives
    - Performances limitées par la nature du dataset 
    - Pistes d'amélioration

**8. Conclusion**

**Annexes**
- A. Datasets abandonnés — Kaggle et FLAIR One
- B. État de l'art complémentaire
- C. Problèmes rencontrés — ResNet50 + CBAM et NaN
- D. Détails d'implémentation Streamlit


--------------
métrique f1 score dans ma partie

