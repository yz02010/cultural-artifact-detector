import json
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

# Output files for titles and mapping
TITLES_FILE = 'titles.txt'
JSON_MAPPING_FILE = 'image_title_mapping.json'
TXT_MAPPING_FILE = 'image_title_mapping.txt'

def setup_driver():
    chrome_options = Options()
    chrome_options.add_argument('--headless')  # Run in headless mode (no GUI)
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver

def scrape_titles(driver, page_num):
    url = f'https://digicol.dpm.org.cn/?page={page_num}'
    print(f"\nScraping page {page_num}: {url}")
    
    try:
        driver.get(url)
        # Wait for the content to load - now looking for divs with class "pic"
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "pic"))
        )
        
        # Let the page fully load
        time.sleep(3)
        
        # Get all divs with class "pic"
        pic_divs = driver.find_elements(By.CLASS_NAME, "pic")
        print(f"Found {len(pic_divs)} pic divs on page {page_num}")
        
        titles_mapping = []
        for index, pic_div in enumerate(pic_divs):
            try:
                image_id = f"image_{page_num}_{index}"
                
                # Initialize all fields
                title = ""
                cultural_number = ""
                cultural_category = ""
                cultural_dynasty = ""
                
                # Look for img_box2 element which contains all the cultural attributes
                img_box_elements = pic_div.find_elements(By.CLASS_NAME, "img_box2")
                
                if img_box_elements:
                    img_box = img_box_elements[0]
                    # Get all the cultural attributes
                    title = img_box.get_attribute("cultural-name") or ""
                    cultural_number = img_box.get_attribute("cultural-number") or ""
                    cultural_category = img_box.get_attribute("cultural-category") or ""
                    cultural_dynasty = img_box.get_attribute("cultural-dynasty") or ""
                
                # If no title found in img_box2, try aria-label from pic div
                if not title:
                    title = pic_div.get_attribute("aria-label") or ""
                
                # If still no title, try to find any text content that might be the title
                if not title:
                    try:
                        # Look for any text within the pic div or its children
                        all_text_elements = pic_div.find_elements(By.XPATH, ".//*[text()]")
                        for elem in all_text_elements:
                            text = elem.text.strip()
                            if text and len(text) > 3:  # Assuming titles are longer than 3 characters
                                title = text
                                break
                    except:
                        pass
                
                # Create a dictionary with all the information
                item_data = {
                    "title": title.strip() if title else "No title",
                    "cultural_number": cultural_number.strip() if cultural_number else "No number",
                    "cultural_category": cultural_category.strip() if cultural_category else "No category",
                    "cultural_dynasty": cultural_dynasty.strip() if cultural_dynasty else "No dynasty"
                }
                
                # Debug: Print the HTML structure of the first few elements
                if index < 3:
                    try:
                        print(f"Debug - HTML structure for {image_id}:")
                        print(pic_div.get_attribute('outerHTML')[:500] + "...")
                    except:
                        pass
                
                print(f"{image_id}: {item_data['title']}")
                print(f"  Number: {item_data['cultural_number']}")
                print(f"  Category: {item_data['cultural_category']}")
                print(f"  Dynasty: {item_data['cultural_dynasty']}")
                
                titles_mapping.append((image_id, item_data))
                    
            except Exception as e:
                print(f"Error processing pic div {index}: {str(e)}")
                continue
        
        return titles_mapping
    except Exception as e:
        print(f"Error scraping page {page_num}: {str(e)}")
        return []

def save_mapping(mapping):
    # Load existing JSON data if it exists
    existing_mapping = {}
    try:
        with open(JSON_MAPPING_FILE, 'r', encoding='utf-8') as f:
            existing_mapping = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        # File doesn't exist or is empty/corrupted, start fresh
        pass
    
    # Merge new mapping with existing data
    existing_mapping.update(mapping)
    
    # Save only valid titles to simple text file (append mode)
    with open(TITLES_FILE, 'a', encoding='utf-8') as f:
        for _, item_data in mapping.items():
            if item_data['title'] != 'No title':
                f.write(item_data['title'] + '\n')
    
    # Save complete mapping to JSON (overwrite with merged data)
    with open(JSON_MAPPING_FILE, 'w', encoding='utf-8') as f:
        json.dump(existing_mapping, f, ensure_ascii=False, indent=2)
    
    # Save mapping to readable text file (append mode)
    with open(TXT_MAPPING_FILE, 'a', encoding='utf-8') as f:
        for image_id, item_data in sorted(mapping.items()):
            f.write(f"{image_id}: {item_data['title']}\n")
            f.write(f"  Number: {item_data['cultural_number']}\n")
            f.write(f"  Category: {item_data['cultural_category']}\n")
            f.write(f"  Dynasty: {item_data['cultural_dynasty']}\n")
            f.write("\n")  # Add blank line between entries
    
    total_new_titles = sum(1 for item_data in mapping.values() if item_data['title'] != 'No title')
    total_all_titles = sum(1 for item_data in existing_mapping.values() if item_data['title'] != 'No title')
    print(f"\nAdded {total_new_titles} new titles to {TITLES_FILE}")
    print(f"Total titles now: {total_all_titles}")
    print(f"Updated image-title mapping with cultural data in {JSON_MAPPING_FILE} and {TXT_MAPPING_FILE}")

def main():
    driver = setup_driver()
    mapping = {}
    try:
        start_page = 50
        end_page = 99  # You can adjust this number to scrape more pages
        
        for page_num in range(start_page, end_page + 1):
            titles_mapping = scrape_titles(driver, page_num)
            for image_id, item_data in titles_mapping:
                mapping[image_id] = item_data
            time.sleep(2)
    finally:
        driver.quit()
    
    save_mapping(mapping)

if __name__ == "__main__":
    main()
