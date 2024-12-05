import json
import os
import re  # Import the re module
import requests
from bs4 import BeautifulSoup

DATASETS_PATH = r"/Users/ashwin/Desktop/LLM_Hackathon/datasets"
DATASETS_MICROLABS_USA = os.path.join(DATASETS_PATH, "microlabs_usa")

URLS = {
    "Acetazolamide Extended-Release Capsules": "https://www.microlabsusa.com/products/acetazolamide-extended-release-capsules/",
    "Amlodipine Besylate and Olmesartan Medoxomil Tablets": "https://www.microlabsusa.com/products/amlodipine-besylate-and-olmesartan-medoxomil-tablets/",
    "Amoxicillin and Clavulanate Potassium for Oral Suspension, USP": "https://www.microlabsusa.com/products/amoxicillin-and-clavulanate-potassium-for-oral-suspension-usp/",
    "Amoxicillin and Clavulanate Potassium Tablets, USP": "https://www.microlabsusa.com/products/amoxicillin-and-clavulanate-potassium-tablets-usp/",
    "Amoxicillin Capsules, USP": "https://www.microlabsusa.com/products/amoxicillin-capsules-usp/",
    "Aspirin and Extended-Release Dipyridamole Capsules": "https://www.microlabsusa.com/products/aspirin-and-extended-release-dipyridamole-capsules/",
    "Atorvastatin Calcium Tablets": "https://www.microlabsusa.com/products/atorvastatin-calcium-tablets/",
    "Bimatoprost Ophthalmic Solution": "https://www.microlabsusa.com/products/bimatoprost-ophthalmic-solution/",
    "Celecoxib capsules": "https://www.microlabsusa.com/products/celecoxib-capsules/",
    "Chlordiazepoxide Hydrochloride and Clidinium Bromide Capsules, USP": "https://www.microlabsusa.com/products/chlordiazepoxide-hydrochloride-and-clidinium-bromide-capsules-usp/",
    "Clindamycin Hydrochloride Capsules, USP " : "https://www.microlabsusa.com/products/clindamycin-hydrochloride-capsules-usp/",
    "Clobazam Tablets" : "https://www.microlabsusa.com/products/clobazam-tablets/",
    "Clobetasol Propionate Topical Solution, USP" : "https://www.microlabsusa.com/products/clobetasol-propionate-topical-solution-usp/",
    "Clomipramine Hydrochloride Capsules, USP" : "https://www.microlabsusa.com/products/clomipramine-hydrochloride-capsules-usp/",
    "Cromolyn Sodium Inhalation Solution, USP" : "https://www.microlabsusa.com/products/cromolyn-sodium-inhalation-solution-usp/",
    "Cromolyn Sodium Oral Solution" : "https://www.microlabsusa.com/products/cromolyn-sodium-oral-solution/",
    "Dalfampridine Extended-Release Tablets" : "https://www.microlabsusa.com/products/dalfampridine-extended-release-tablets/",
    "Diclofenac Sodium and Misoprostol Delayed-Release Tablets, USP" : "https://www.microlabsusa.com/products/diclofenac-sodium-and-misoprostol-delayed-release-tablets-usp/",
    "Dorzolamide HCl Ophthalmic Solution, USP" : "https://www.microlabsusa.com/products/dorzolamide-hcl-ophthalmic-solution-usp/",
    "Dorzolamide HCl and Timolol Maleate Ophthalmic Solution, USP" : "https://www.microlabsusa.com/products/dorzolamide-hcl-and-timolol-maleate-ophthalmic-solution-usp/",
    "Dorzolamide HCl and Timolol Maleate Ophthalmic Solution, USP (Preservative-Free)" : "https://www.microlabsusa.com/products/dorzolamide-hcl-and-timolol-maleate-ophthalmic-solution-usppreservative-free/",
    "Erythromycin Topical Solution, USP" : "https://www.microlabsusa.com/products/erythromycin-topical-solution-usp/",
    "Famotidine for Oral Suspension, USP" : "https://www.microlabsusa.com/products/famotidine-for-oral-suspension-usp/",
    "Fenofibric Acid Delayed-Release Capsules" : "https://www.microlabsusa.com/products/fenofibric-acid-delayed-release-capsules/",
    "Glimepiride Tablets, USP" : "https://www.microlabsusa.com/products/glimepiride-tablets-usp/",
    "Ketorolac Tromethamine Ophthalmic Solution" : "https://www.microlabsusa.com/products/ketorolac-tromethamine-ophthalmic-solution/",
    "Levocetirizine Dihydrochloride Tablets, USP" : "https://www.microlabsusa.com/products/levocetirizine-dihydrochloride-tablets-usp/",
    "Mefenamic Acid Capsules, USP" : "https://www.microlabsusa.com/products/mefenamic-acid-capsules-usp/",
    "Metformin Hydrochloride Extended-Release Tablets, USP" : "https://www.microlabsusa.com/products/metformin-hydrochloride-extended-release-tablets-usp/",
    "Metformin Hydrochloride Oral Solution" : "https://www.microlabsusa.com/products/metformin-hydrochloride-oral-solution/",
    "Methenamine Hippurate Tablets, USP" : "https://www.microlabsusa.com/products/methenamine-hippurate-tablets-usp/",
    "Olmesartan Medoxomil Tablets, USP" : "https://www.microlabsusa.com/products/olmesartan-medoxomil-tablets-usp/",
    "Piroxicam Capsules, USP" : "https://www.microlabsusa.com/products/piroxicam-capsules-usp/",
    "Potassium Chloride Oral Solution, USP" : "https://www.microlabsusa.com/products/potassium-chloride-oral-solution-usp/",
    "Ramelteon Tablets" : "https://www.microlabsusa.com/products/ramelteon-tablets/",
    "Ranolazine Extended-Release Tablets" : "https://www.microlabsusa.com/products/ranolazine-extended-release-tablets/",
    "Rasagiline Tablets" : "https://www.microlabsusa.com/products/rasagiline-tablets/",
    "Roflumilast Tablets" : "https://www.microlabsusa.com/products/roflumilast/",
    "Rufinamide Tablets, USP" : "https://www.microlabsusa.com/products/rufinamide-tablets-usp/",
    "Tafluprost Ophthalmic Solution" : "https://www.microlabsusa.com/products/tafluprost-opthalmic-solution/",
    "Telmisartan Tablets, USP" : "https://www.microlabsusa.com/products/telmisartan-tablets-usp/",
    "Timolol Maleate Ophthalmic Solution, USP (Preservative-Free)" : "https://www.microlabsusa.com/products/timolol-maleate-ophthalmic-solution-usp-preservative-free/",
    "Tobramycin Inhalation Solution, USP" : "https://www.microlabsusa.com/products/tobramycin-inhalation-solution-usp/",
    "Travoprost Ophthalmic Solution, USP" : "https://www.microlabsusa.com/products/travoprost-ophthalmic-solution-usp/",
    "Triamcinolone Acetonide Lotion, USP" : "https://www.microlabsusa.com/products/triamcinolone-acetonide-lotion-usp/"

}

def setup_prescribing_info_urls(urls_map):
    updated_urls = {}

    print("[INFO] Starting to process URLs...")
    for key, value in urls_map.items():
        print(f"[INFO] Processing product: {key}")
        updated_urls[key] = {"product_url": value}
        try:
            data = requests.get(value)
            if data.status_code != 200:
                print(f"[ERROR] Failed to fetch URL: {value} (Status Code: {data.status_code})")
                continue
            soup = BeautifulSoup(data.text, "html.parser")
            h2 = soup.findAll("h2")

            got = False
            for h2_item in h2:
                txt = h2_item.get_text()
                if txt and txt.strip().lower() == "Prescribing Information".lower():
                    child_url = h2_item.findAll("a")
                    if child_url:
                        href = child_url[0].get("href")
                        if not href.startswith("http"):
                            print(f"[ERROR] Invalid prescribing info URL: {href}")
                            continue
                        updated_urls[key]["prescribing_info_url"] = href
                        html = requests.get(href)
                        prescribing_soup = BeautifulSoup(html.text, "html.parser")
                        updated_urls[key]["prescribing_soup"] = prescribing_soup
                        print(f"[INFO] Found prescribing info for {key}")
                        got = True
                        break  # Move this break inside the if block
            if not got:
                print(f"[WARNING] No prescribing information found for {key}")

        except Exception as e:
            print(f"[ERROR] Exception occurred while processing {key}: {e}")
    print("[INFO] Finished processing URLs.")
    return updated_urls

def get_all_sections(soup):
    headers = soup.find_all(['h1', 'h2', 'h3'])
    info = {}
    for header in headers:
        section_title = header.get_text(strip=True)
        content = []
        for sibling in header.find_next_siblings():
            if sibling.name in ['h1', 'h2', 'h3']:
                break
            content.append(sibling.get_text(strip=True))
        info[section_title] = "\n".join(content)
    return info

def process_prescribing_soup(name, soup):
    print(f"[INFO] Parsing prescribing information for {name}")
    try:
        results = get_all_sections(soup)
        results["product_name"] = name
        print(f"[INFO] Successfully parsed prescribing info for {name}")
        return results
    except Exception as e:
        print(f"[ERROR] Failed to process prescribing soup for {name}: {e}")
        return {}

def create_dataset_file(pth, result):
    try:
        sanitized_name = re.sub(r'[<>:"/\\|?*]', '_', result["product_name"])
        fname = os.path.join(pth, sanitized_name + ".json")
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4)
        print(f"[INFO] Dataset file created at {fname}")
    except Exception as e:
        print(f"[ERROR] Failed to create dataset file for {result.get('product_name', 'unknown')}: {e}")

if __name__ == '__main__':
    print("[INFO] Script started.")
    os.makedirs(DATASETS_MICROLABS_USA, exist_ok=True)
    print(f"[INFO] Output directory ensured: {DATASETS_MICROLABS_USA}")

    modified_urls = setup_prescribing_info_urls(URLS)
    print("[INFO] Processing prescribing soups...")
    for k, v in modified_urls.items():
        if "prescribing_soup" not in v:
            print(f"[WARNING] Skipping {k} due to missing prescribing info.")
            continue
        results = process_prescribing_soup(k, v["prescribing_soup"])
        if results:
            create_dataset_file(DATASETS_MICROLABS_USA, results)
        else:
            print(f"[WARNING] No data extracted for {k}, skipping file creation.")

    print("[INFO] Script completed.")