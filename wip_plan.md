Abstract

Contexte métier
	secours 

Problématique
    hypothèse

etat l'art
	ce qui existe déja et se fait de mieux modèle
    (dataset si y'a)

dataset
	OSV5M
		analyse

Approche théorique cf problématique 
        Stratégie Classif, Reg, ....
        métrique d'éval
        Schéma de tous les modèle a tester
        répartition tache Méthodologie des tâches ? (pt bizz : avancement pratique)

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
		Problème rencontré
		Visualiosation de l'impact de l'attention GradCAM
            Résultats
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
        (Positionnement) à bouger
        Résumé


Analyse du tableau des différents modèles
    Comparaison uniquement
    Positionnement 

Réponse à notre problématique + contexte métier
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



             



