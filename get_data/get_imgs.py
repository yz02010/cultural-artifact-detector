import requests
from bs4 import BeautifulSoup
import os
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

# Create image directory if it doesn't exist
if not os.path.exists('image'):
    os.makedirs('image')

headers = {'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}

def setup_driver():
    chrome_options = Options()
    chrome_options.add_argument('--headless')  # Run in headless mode (no GUI)
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver

def download_image(url, title):
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            # Create a valid filename from the title
            filename = "".join(x for x in title if x.isalnum() or x in (' ', '-', '_')).rstrip()
            if not filename:  # If title is empty, use part of the URL
                filename = url.split('/')[-1].split('.')[0]
            filename = f"image/{filename}.jpg"
            
            with open(filename, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded: {filename}")
            return True
    except Exception as e:
        print(f"Error downloading {url}: {str(e)}")
    return False

def scrape_page(driver, page_num):
    url = f'https://digicol.dpm.org.cn/?page={page_num}'
    print(f"\nScraping page {page_num}: {url}")
    
    try:
        driver.get(url)
        # Wait for the content to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "img001"))
        )
        
        # Let the page fully load
        time.sleep(3)
        
        # Get all images with class img001
        images = driver.find_elements(By.CLASS_NAME, "img001")
        print(f"Found {len(images)} images on page {page_num}")
        
        for img in images:
            try:
                img_url = img.get_attribute('src')
                title = img.get_attribute('aria-label') or img.get_attribute('alt') or ''
                
                if not title:
                    # Try to get title from parent
                    parent = img.find_element(By.XPATH, "./..")
                    title = parent.get_attribute('aria-label') or parent.get_attribute('title') or ''
                
                if img_url:
                    print(f"Found image: {img_url}")
                    print(f"Title: {title}")
                    download_image(img_url, title or f"image_{page_num}_{images.index(img)}")
                    time.sleep(1)
            except Exception as e:
                print(f"Error processing image: {str(e)}")
                continue
        
        return True
    except Exception as e:
        print(f"Error scraping page {page_num}: {str(e)}")
        return False

def main():
    driver = setup_driver()
    try:
        start_page = 50
        end_page = 99  # You can adjust this number to scrape more pages
        
        for page_num in range(start_page, end_page + 1):
            if not scrape_page(driver, page_num):
                print(f"Stopping at page {page_num} due to error")
                break
            time.sleep(2)
    finally:
        driver.quit()

if __name__ == "__main__":
    main()
