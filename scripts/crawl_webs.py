import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, urldefrag, parse_qs, urlencode, urlunparse
import os
import json
import time

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from tqdm import tqdm

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from fake_useragent import UserAgent

# Selenium setup
options = Options()
options.add_argument("--headless=new")
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# Splitter
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# Base domains
BASE_URLS = ["https://metiers-quebec.org/accueil.html"]
visited_urls = set()
all_chunks = []
chunk_counter = 0
save_every = 3000
output_dir = "data"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "pdfs_temp"), exist_ok=True)


def normalize_url(url):
    url, _ = urldefrag(url)  # remove fragments
    if url.startswith("http://"):
        url = "https://" + url[7:]

    parsed = urlparse(url)
    # remove tracking query parameters
    clean_query = {k: v for k, v in parse_qs(parsed.query).items()
                   if not k.startswith(("utm_", "fbclid", "gclid", "ref", "sessionid"))}

    # remove trailing slash unless it's the root
    path = parsed.path.rstrip("/") if parsed.path != "/" else parsed.path
    cleaned = parsed._replace(query=urlencode(clean_query, doseq=True), path=path)
    return urlunparse(cleaned)

def is_valid_url(url, base_domain):
    parsed = urlparse(url)
    return parsed.netloc.endswith(base_domain)

def extract_text_with_selenium(url):
    try:
        driver.get(url)
        time.sleep(3)
        html = driver.page_source
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return "\n".join(lines)
    except Exception as e:
        print(f"[Error Selenium] {url}: {e}")
        return None

def process_pdf(url):
    try:
        filename = url.split("/")[-1]
        save_path = os.path.join(output_dir, "pdfs_temp", filename)

        headers = {"User-Agent": UserAgent().random}
        r = requests.get(url, headers=headers)
        with open(save_path, 'wb') as f:
            f.write(r.content)

        loader = PyPDFLoader(save_path)
        docs = loader.load()
        for doc in docs:
            doc.metadata = {"source": url}
        return docs
    except Exception as e:
        print(f"[Error PDF] {url}: {e}")
        return []

def save_chunks_periodically(force=False):
    global chunk_counter, all_chunks
    if len(all_chunks) >= save_every or force:
        part_number = chunk_counter // save_every + 1
        output_path = os.path.join(output_dir, f"web_chunks_part_{part_number}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                [{"content": chunk.page_content, "metadata": chunk.metadata} for chunk in all_chunks],
                f,
                ensure_ascii=False,
                indent=2
            )
        print(f"\n📁 Saved {len(all_chunks)} chunks to: {output_path}")
        chunk_counter += len(all_chunks)
        all_chunks = []

def crawl(base_url):
    base_domain = urlparse(base_url).netloc
    to_visit = [base_url]

    with tqdm(desc=f"Crawling {base_domain}") as pbar:
        while to_visit:
            current_url = to_visit.pop(0)
            normalized_url = normalize_url(current_url)

            if not is_valid_url(normalized_url, base_domain) or normalized_url in visited_urls:
                continue

            visited_urls.add(normalized_url)
            pbar.update(1)

            if normalized_url.lower().endswith(".pdf"):
                docs = process_pdf(normalized_url)
                all_chunks.extend(docs)
                save_chunks_periodically()
                continue

            text = extract_text_with_selenium(normalized_url)
            if text:
                docs = splitter.create_documents([text])
                for doc in docs:
                    doc.metadata = {"source": normalized_url}
                all_chunks.extend(docs)

                save_chunks_periodically()

                soup = BeautifulSoup(driver.page_source, "html.parser")
                for link in soup.find_all("a", href=True):
                    new_url = urljoin(normalized_url, link['href'])
                    normalized_link = normalize_url(new_url)
                    if is_valid_url(normalized_link, base_domain) and normalized_link not in visited_urls:
                        to_visit.append(normalized_link)
                        print(f"🔗 Found link: {normalized_link}")


for base_url in BASE_URLS:
    crawl(base_url)

# Save the last chunks
save_chunks_periodically(force=True)

print(f"\n✅ Finished crawling! Total saved chunks: {chunk_counter}")
