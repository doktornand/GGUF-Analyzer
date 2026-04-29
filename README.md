# GGUF-Analyzer

GGUF-Analyzer est un outil conçu pour analyser les fichiers GGUF (format utilisé notamment pour les modèles LLM optimisés comme ceux de llama.cpp).

## 🚀 Fonctionnalités

- Analyse détaillée des fichiers `.gguf`
- Extraction des métadonnées
- Inspection des tenseurs et paramètres
- Affichage structuré et lisible

## 📦 Installation

```bash
git clone https://github.com/doktornand/GGUF-Analyzer.git
cd GGUF-Analyzer
pip install -r requirements.txt
```

## ▶️ Utilisation

```bash
python main.py <chemin_du_fichier.gguf>
```

### Exemple

```bash
python main.py model.gguf
```

## 🧠 À propos du format GGUF

GGUF est un format de fichier utilisé pour stocker des modèles de langage optimisés, souvent associés à llama.cpp. Il permet une meilleure portabilité et efficacité.

## 📁 Structure du projet

- `main.py` : point d'entrée principal
- `parser/` : logique d'analyse des fichiers GGUF
- `utils/` : fonctions utilitaires

## 🛠️ Dépendances

- Python 3.8+
- Voir `requirements.txt`

## 🤝 Contribution

Les contributions sont les bienvenues !

1. Fork le projet
2. Crée une branche (`git checkout -b feature/ma-feature`)
3. Commit (`git commit -m 'Ajout de ma feature'`)
4. Push (`git push origin feature/ma-feature`)
5. Ouvre une Pull Request

## 📄 Licence

Ce projet est sous licence MIT.

## 👤 Auteur

- doktornand

---

⭐ N'hésitez pas à mettre une étoile au repo si ce projet vous est utile !
