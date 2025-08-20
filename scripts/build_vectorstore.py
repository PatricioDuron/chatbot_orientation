import os
import json
import re
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from tqdm import tqdm


data_folder = os.path.join("data")
vectorstore_path = os.path.join("vectorstore")


chunk_pattern = re.compile(r"^chunks\.json$")
# Function to clean text
def clean_chunk(text: str) -> str:
    patterns = [
    r"Retour en haut",
    r"Dernière mise à jour\s*:.*",
    r"Navigation de pied de page.*",
    r"Gouvernement du Québec",
    r"Abonnements.*?Flux RSS",
    r"En savoir plus",
    r"À propos de l’organisation",
    r"S'abonner.*?",
    r"^Page \d+ sur \d+",
    r"Page \d+(\n\d+)?",                   
    r"Footer navigation \| Québec\.ca",    
    r"Back to top",                        
    r"Pagination\npage n°\d+ sur \d+",      
    r"Joindre les responsables des relations avec les médias",
    r"Liste des responsables des relations avec les médias",
    r"Voir plus",
    r"Page courante",
    r"Cabinet du premier ministre",
    r"Flux RSS",
    r"©\s*Gouvernement du Québec.*",
    r"Politique de confidentialité",
    r"Plan du site",
    r"Contact",
    r"Accueil",
    r"Voir aussi",
    r"Plus d['’]infos",
    r"Tous droits réservés",
    r"Cookies",
    r"FAQ",
    r"Conditions d’utilisation",
    r"Accessibilité",
    r"Aide",
    r"Recherche",
    r"Newsletter",
    r"Langue",
    r"Suivez-nous",
    r"©.*\d{4}",
    r"Page précédente",
    r"Page suivante",
    r"Fermer",
    r"Navigation principale",
    r"Menu",
    r"Connexion",
    r"Déconnexion",
    r"Version imprimable",
    r"Fil d’ariane",
    r"Imprimer",
    r"Partager",
    r"Sitemap",
    r"Informations légales",
    r"Mentions légales",
    r"Conditions générales",
    r"Réseaux sociaux",
    r"Nouvelle fenêtre",
    r"\n+",
    r"Tous droits réservés",
    r"Déclaration de confidentialité",
    r"Énoncé de confidentialité",
    r"Mise à jour le \d{1,2} \w+ \d{4}",
    r"Impossible de trouver la page.*",
    r"English",
    r"Español",
    r"Português",
    r"日本語",
    r"Italiano",
    r"Deutsch",
    r"한국어",
    r"中文, 汉语",
    r"中文, 漢語",
    r"Nederlands",
    r"Українська",
    r"Veuillez préciser la nature du problème",
    r"Don't fill this field!",
    r"Votre commentaires serviront à bonifier Québec\.ca.*",
    r"Metiers Québec",
    r"Image",
    r"La pandémie de la COVID-19",
    r"Cabinet du ministre délégué à l'Économie.*",  
    r"Page \d+",                            
    r"page n°0 sur 0",                       
    r"Page \d+\n\d+",                        
    r"Joindre les responsables des relations .*", 
    r"Page \d+\nPage \d+",
    r"Joindre les responsables des relations avec les médias\s*Liste des responsables des relations avec les médias\s*Gouvernement du Québec\s*Navigation de pied de page de Québec\.ca\s*Retour en haut",
    r"Pagination\s*page n°\d+ sur \d+\s*(?:\.\n)?(?:Page \d+\n?)+- Page courante\n(?:Page \d+\n?)+\.?",
    r"(Cabinet|Commission)[^\n]+\n(?:.|\n)+?Retour en haut",
    r"Page \d+\n\d+",
    r"(Cabinet du|Commission|Délégation)[^\n]+\n(?:.|\n)+?Retour en haut",
    r"Délégation générale du Québec à Bruxelles",
    r"Cabinet du ministre de l'Agriculture, des Pêcheries et de l'Alimentation",
    r"JavaScript is disabled\s+In order to continue, we need to verify that you're not a robot\.\s+This requires JavaScript\. Enable JavaScript and then reload the page\.",
    r"Would you like to add any details\?\s+Do not include any personal information .+?\.\s+Your comments will be used to improve Québec\.ca and may be used for statistical purposes\. Please note that we will not be able to provide individual responses\.\s+Please describe the problem\s+Don't fill this field!",
    r"MISE EN GARDE À LA POPULATION",
    r"Lire le communiqué",
    r"Agriculture, Pêcheries et Alimentation",
    r"Conseil des arts et des lettres du Québec",
    r"Tourisme",
    r"Régie des marchés agricoles et alimentaires du Québec",
    r"Centre d'acquisitions gouvernementales",
    r"Bureau du Québec à Barcelone",
    r"Archives des nouvelles relatives à Forêts, Faune et Parcs",
    r"Langue française",
    r"Office des personnes handicapées du Québec",
    r"Délégation générale du Québec à Bruxelles",
    r"CABINET DE LA MINISTRE DE L'ÉCONOMIE, DE L'INNOVATION ET DE L'ÉNERGIE ET MINISTRE RESPONSABLE DU DÉVELOPPEMENT ÉCONOMIQUE RÉGIONAL",
    r"Lire le communiqué",
    r"Autorité des marchés financiers",
    r"Contact information for media relations officials",
    r"Media relations officials in the departments and agencies \(in French only\)",
    r"Footer navigation \| Québec\.ca",
    r"Paginate",  
    r"- Current page",
    r"MONTREAL, Jan\. \d{1,2}, \d{4}",  
    r"MONTRÉAL, le \d+ \w+ \d{4} /CNW/ - .+",
    r"QUÉBEC, le \d+ \w+ \d{4} – .+",
    r"Héma-Québec",
    r"Secrétariat aux relations avec les Premières Nations et les Inuit",
    r"Institut national de la recherche scientifique",
    r"Drummondville centre key to plasma self-sufficiency strategy MONTRÉAL, April \d+, 2024 /CNW/ - .+",
    r"Paginate\s+page n°0 of 0",  
    r"Page \d+",
    r"Contact information for media relations officials",
    r"Media relations officials in the departments and agencies \(in French only\)",
    r"Footer navigation \| Québec\.ca",
    r"Back to top",
    r"- Current page",
    r"Paginate",
    r"Société du Plan Nord",
    r"Palais des congrès de Montréal",
    r"Protecteur du citoyen",
    r"Centre intégré universitaire de santé et de services sociaux de l'Est-de-l'Île-de-Montréal",
    r"Québec Government Office in Brussels",
    r"Institut national de la recherche scientifique",
    r"Voir plus",
    r"Dernière mise à jour : \d{1,2} \w+ \d{4}",
    r"À propos de l’organisation",
    r"Abonnements",
    r"S'abonner au fil de presse",
    r"Flux RSS",
    r"Suivez-nous",
    r"Cabinet de la vice-première ministre et ministre des Transports et de la Mobilité durable",
    r"Chacun de\nnous a un rôle à jouer dans l'amélioration du bilan routier.*",
    r"Le nouveau gouvernement du Québec travaillera en collaboration avec la Ville de Montréal",
    r"Se renseigner sur .+",
    r"Mise à jour le \d{1,2} \w+ \d{4}",
    r"Paginate\s*page n°0 of 0",
    r"Page \d+",
    r"Pagination\s*page n°0 sur 0",
    r"Page \d+",
]
    unique_patterns = []
    seen = set()
    for pattern in patterns:
        if pattern not in seen:
            unique_patterns.append(pattern)
            seen.add(pattern)

    patterns = unique_patterns

    for pattern in unique_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r" +", " ", text)
    return text.strip()

# Read and clean all chunk files
all_documents = []

for filename in tqdm(os.listdir(data_folder), desc="Loading files"):
    if chunk_pattern.match(filename):
        filepath = os.path.join(data_folder, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                # Progress bar for each chunk in the file
                for chunk in tqdm(data, desc=f"Processing {filename}", leave=False):
                    clean_text = clean_chunk(chunk["content"])
                    if clean_text:
                        if "source" not in chunk["metadata"]:
                            chunk["metadata"]["source"] = filename 
                        all_documents.append(
                            Document(
                                page_content=clean_text,
                                metadata=chunk["metadata"]
                            )
                        )
            except json.JSONDecodeError as e:
                print(f"⚠️ Error reading {filename}: {e}")

print(f"📄 Total documents charged for vectorstore: {len(all_documents)}")

# Create embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create vectorstore with all documents
db = FAISS.from_documents(all_documents, embedding_model)
db.save_local(vectorstore_path)

# Save vectorstores
print(f"✅ Vectorstore saved in: {vectorstore_path}")
