# GeoGuesserIA


## Prérequis
- Python 3.9
- Conda : `conda env create -f environment.yml`
            puis `conda activate geoguesseriassh`

## Installations

1. Cloner le dépôt :
git clone https://github.com/fannnyyy/GeoGuesserIA.git
cd GeoGuesserIA

2. Placer le dossier saved/ dans GeoGuesserIA/model/.

3. Activer l'environnement Conda :
conda activate geoguesseriassh


## Lancer l'application Streamlit

Il est possible de lancer l'application Streamlit de deux manières.

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

## Notes de développement

Le dépôt contient plusieurs fichiers de notes au format wip_xxx.md.

Ils permettent de suivre l'avancement et les conclusions des différentes parties du projet, notamment les modèles et l'application Streamlit.
