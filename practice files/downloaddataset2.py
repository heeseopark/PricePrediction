from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver import ActionChains
from selenium.common.exceptions import NoSuchElementException, ElementNotInteractableException
import time

global browser

service = Service('.\chromedriver\chromedriver.exe')
options = webdriver.ChromeOptions()
options.add_experimental_option("excludeSwitches", ["enable-logging"])

# Add download preferences
prefs = {
    "download.default_directory": r"C:\Github\PricePrediction\btc_data",
    "download.prompt_for_download": False,
    "download.directory_upgrade": True,
}
options.add_experimental_option("prefs", prefs)

browser = webdriver.Chrome(service=service, options=options)

browser.get('https://data.binance.vision/?prefix=data/spot/daily/klines/BTCUSDT/1m/')

browser.implicitly_wait(10)

row = 3
while True:
    try:
        download_link = browser.find_element(By.XPATH, f'/html/body/div/table/tbody/tr[{row}]/td[1]/a')

        browser.execute_script("arguments[0].scrollIntoView({ block: 'center' });", download_link)
        
        download_link.click()
        print(f'row {row} downloaded')
        row += 2
        time.sleep(0.1)
    except (NoSuchElementException, ElementNotInteractableException):
        action = ActionChains(browser)
        action.move_to_element(download_link)
        browser.execute_script("arguments[0].scrollIntoView({ block: 'center' });", download_link)
        
        download_link.click()
        row += 2
        time.sleep(0.1)
        break

print('download all complete')
