# Installation
## Librairies pour faire tourner les modèles sur GPU:
Guide dont s'inspire les étapes suivantes: https://www.tensorflow.org/install/gpu
Mais normalement, les explications ci-dessous sont suffisantes.

### 0. Installer Python 3.10.9:
https://www.python.org/ftp/python/3.10.9/...
-> Prendre amd64 pour 64 bits sur windows

### 1. Installation de cuda toolkit 11.2 (bien respecter la version, sinon incompatible avec tensorflow) :
https://developer.nvidia.com/cuda-11.2.0-download-archive
Next next next

### 2. Installation de cuDNN 8.1:
https://developer.nvidia.com/rdp/cudnn-archive
-> Il suffit de décompresser l'archive téléchargée et d'ajouter le /bin au path

### 2.1. Redémarrer le PC

## L'interpréteur python, et des dépendances
### Sans Conda
1) Installer la dernière version de Python 3.10 et la dernière versio nde Python 3.11
   On veillera à installer Python avec le gestionnaire de package 'pip', et à ajouter
   'python.exe' au path du système.
2) Installer les dépendances suivantes:
```
py -3.10 -m pip install "tensorflow<2.11"
py -3.10 -m pip install "matplotlib<3.8"
py -3.10 -m pip install "scipy<1.13"
py -3.10 -m pip install "pandas<1.6"
py -3.10 -m pip install "cloudpickle<3.1"
py -3.10 -m pip install "opencv-python<4.10" 
py -3.10 -m pip install "statsmodels<0.15"
py -3.10 -m pip install "scipy<1.14"
```
3) Installer les dépendances nécessaires pour faire tourner les notebooks Jupyterlab sur Python 3.11
```
py -3.11 -m pip install "jupyterlab<3.7"
```

Il est nécessaire d'installer 2 runtime de Python, car tensorflow 2.10 n'est pas compatible avec jupyterlab.
La librairie qui pose problème est 'typing_extensions': tensorflow 2.10 dépend d'une ancienne version
de typing_extensions, alors que Jupyterlab ne fonctionne qu'avec des version plus récentes de 'typing_extensions'.

Il existe cependant des versions plus récentes de tensorflow, mais ces versions ne permettent pas de faire tourner
un modèle sur carte graphique sur Windows (uniquement sur Linux). Mais à défaut d'avoir les compétences Docker, ou 
bien un système d'exploitation linux sous la main, il n'est pas possible de faire autrement que d'avoir 2 
runtime Python pour faire tourner les notebooks.

### Avec Conda
Il s'agit de réaliser la même chose: il faut créer 2 runtime Conda. Pour installer les dépendances, je recommande
également 'pip' qui est installé par défaut sur les runtime Conda. De fait, Conda m'a posé certains soucis
avec les librairies opencv et tensorflow, alors que l'utilitaire 'pip' les installe correctement.

Une fois chacun des 2 runtimes créée, on installe donc les dépendances sur chacun des environnement:
- Runtime 1 (python 3.10)
```
conda activate <nomRuntime1>
python -m pip install "tensorflow<2.11"
python -m pip install "matplotlib<3.8"
python -m pip install "scipy<1.13"
python -m pip install "pandas<1.6"
python -m pip install "cloudpickle<3.1"
python -m pip install "opencv-python<4.10" 
python -m pip install "statsmodels<0.15"
python -m pip install "scipy<1.14"
python -m pip install "scikit-optimize<0.11"
```
- Runtime 2 (python 3.10 ou 3.11, qu'importe)
```
conda activate env <nomRuntime2>
python -m pip install "jupyterlab<3.7"
```
(Je recommande de créer les environnements conda depuis le navigateur conda)

## Un kernel pour un notebook
Un notebook tourne sur un kernel (un kernel peut-être vu comme un environnement conda). Pour créer un tel kernel,
on utilise la ligne de commande :
```
python -m ipykernel install --user --name=NameOfKernel
```
Il faut que 'python' qui est utilisé dans cette commande soit le python de l'environnement que l'on souhaite
convertir en un kernel.

D'ailleurs, pour exécuter cette commande, il est nécessaire d'installer 'ipykernel', d'où la ligne 
'py -3.10 -m pip install "ipykernel==6.3.1"' dans le fichier d'installation du projet.

Si bien installé, un nouveau répertoire au nom de <NameOfKernel> devrait être apparu dans :
C:\Users\<your_name>\AppData\Roaming\jupyter\kernels

## Vérification
Pour vérifier que tout fonctionne bien, exécuter le code suivant dans un notebook :
```
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```
Ce code devrait lister les GPUs disponibles de la machine. Si aucun GPU n'apparait, c'est que l'installation 
est un échec ;(