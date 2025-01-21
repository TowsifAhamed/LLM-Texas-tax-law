import requests
from bs4 import BeautifulSoup
import os

# List of chapters
chapters = [
    "CHAPTER 1. GENERAL PROVISIONS",
    "CHAPTER 5. STATE ADMINISTRATION",
    "CHAPTER 6. LOCAL ADMINISTRATION",
    "CHAPTER 11. TAXABLE PROPERTY AND EXEMPTIONS",
    "CHAPTER 21. TAXABLE SITUS",
    "CHAPTER 22. RENDITIONS AND OTHER REPORTS",
    "CHAPTER 23. APPRAISAL METHODS AND PROCEDURES",
    "CHAPTER 24. CENTRAL APPRAISAL",
    "CHAPTER 25. LOCAL APPRAISAL",
    "CHAPTER 26. ASSESSMENT",
    "CHAPTER 31. COLLECTIONS",
    "CHAPTER 32. TAX LIENS AND PERSONAL LIABILITY",
    "CHAPTER 33. DELINQUENCY",
    "CHAPTER 34. TAX SALES AND REDEMPTION",
    "CHAPTER 41. LOCAL REVIEW",
    "CHAPTER 41A. APPEAL THROUGH BINDING ARBITRATION",
    "CHAPTER 42. JUDICIAL REVIEW",
    "CHAPTER 43. SUIT AGAINST APPRAISAL OFFICE",
    "CHAPTER 101. GENERAL PROVISIONS",
    "CHAPTER 111. COLLECTION PROCEDURES",
    "CHAPTER 112. TAXPAYERS' SUITS",
    "CHAPTER 113. TAX LIENS",
    "CHAPTER 141. MULTISTATE TAX COMPACT",
    "CHAPTER 142. SIMPLIFIED SALES AND USE TAX ADMINISTRATION ACT",
    "CHAPTER 151. LIMITED SALES, EXCISE, AND USE TAX",
    "CHAPTER 152. TAXES ON SALE, RENTAL, AND USE OF MOTOR VEHICLES",
    "CHAPTER 154. CIGARETTE TAX",
    "CHAPTER 155. CIGARS AND TOBACCO PRODUCTS TAX",
    "CHAPTER 156. HOTEL OCCUPANCY TAX",
    "CHAPTER 158. MANUFACTURED HOUSING SALES AND USE TAX",
    "CHAPTER 160. TAXES ON SALES AND USE OF BOATS AND BOAT MOTORS",
    "CHAPTER 162. MOTOR FUEL TAXES",
    "CHAPTER 163. SALES AND USE TAXATION OF AIRCRAFT",
    "CHAPTER 171. FRANCHISE TAX",
    "CHAPTER 172. TAX CREDIT FOR CERTIFIED REHABILITATION OF CERTIFIED HISTORIC STRUCTURES",
    "CHAPTER 181. CEMENT PRODUCTION TAX",
    "CHAPTER 182. MISCELLANEOUS GROSS RECEIPTS TAXES",
    "CHAPTER 183. MIXED BEVERAGE TAXES",
    "CHAPTER 191. MISCELLANEOUS OCCUPATION TAXES",
    "CHAPTER 201. GAS PRODUCTION TAX",
    "CHAPTER 202. OIL PRODUCTION TAX",
    "CHAPTER 204. TAX CREDIT FOR NEW FIELD DISCOVERIES",
    "CHAPTER 301. GENERAL PROVISIONS",
    "CHAPTER 302. TAXATION POWERS OF MUNICIPALITIES",
    "CHAPTER 311. TAX INCREMENT FINANCING ACT",
    "CHAPTER 312. PROPERTY REDEVELOPMENT AND TAX ABATEMENT ACT",
    "CHAPTER 313. TEXAS ECONOMIC DEVELOPMENT ACT",
    "CHAPTER 320. MISCELLANEOUS PROVISIONS",
    "CHAPTER 321. MUNICIPAL SALES AND USE TAX ACT",
    "CHAPTER 322. SALES AND USE TAXES FOR SPECIAL PURPOSE TAXING AUTHORITIES",
    "CHAPTER 323. COUNTY SALES AND USE TAX ACT",
    "CHAPTER 324. COUNTY HEALTH SERVICES SALES AND USE TAX",
    "CHAPTER 325. COUNTY SALES AND USE TAX FOR LANDFILL AND CRIMINAL DETENTION CENTER",
    "CHAPTER 327. MUNICIPAL SALES AND USE TAX FOR STREET MAINTENANCE",
    "CHAPTER 351. MUNICIPAL HOTEL OCCUPANCY TAXES",
    "CHAPTER 352. COUNTY HOTEL OCCUPANCY TAXES"
]

# Base URL for Texas statutes
base_url = "https://statutes.capitol.texas.gov/Docs/TX/htm/TX.{}.htm"

# Function to scrape a single chapter
def scrape_chapter(chapter_number):
    url = base_url.format(chapter_number)
    print(f"Attempting URL: {url}")
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            content = soup.get_text(separator="\n")
            return content
        else:
            print(f"Failed to fetch chapter {chapter_number}: HTTP {response.status_code}")
            return None
    except requests.RequestException as e:
        print(f"Error fetching chapter {chapter_number}: {e}")
        return None

# Function to extract chapter number
def extract_chapter_number(chapter_title):
    return chapter_title.split()[1].strip('.')

# Create a directory for the output files
output_dir = "doc"
os.makedirs(output_dir, exist_ok=True)

# Main scraping loop
for chapter in chapters:
    chapter_number = extract_chapter_number(chapter)
    chapter_filename = f"{chapter}.txt".replace(" ", "_")  # Replace spaces with underscores
    file_path = os.path.join(output_dir, chapter_filename)

    print(f"Scraping {chapter}...")
    content = scrape_chapter(chapter_number)
    if content:
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(content)
        print(f"Saved {chapter} to {file_path}")
    else:
        print(f"Skipping {chapter} due to fetch failure.")

print("Scraping completed. All chapters saved in the 'doc' folder.")
