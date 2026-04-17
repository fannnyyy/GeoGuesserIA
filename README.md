# GeoGuesserIA


## Prérequis
- Python 3.9
- Conda : `conda env create -f environment.yml`
            puis `conda activate geoguesseriassh`

## Installations
1. Cloner le dépôt
2. Placer le dossier "saved/" dans "GeoGuesserIA/model/"
3. Activer l'environnement conda

Ensuite, il y a plusieurs moyens de lancer l'application streamlit :
- en local : 
```
	cd visualisation
	streamlit run streamlit_app.py
```

- via un job : 
```
	sbatch job/streamlit_job.sh
	si un pont ssh-localhost est requis :
		ssh -L 8501:<nodelist>:8501 dce-login
		Ouvrir http://localhost:8501
```

Dans le repo il est possible de trouver des fichiers de notes, nommés wip_xxx.md, qui indiquent l'avancement et les conclusions de chaque partie (modèles, streamlit…).

