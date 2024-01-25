# scraper.py
import requests
from bs4 import BeautifulSoup
import json
import markdown

def scrape_github_repo(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Initialize repository data dictionary
        repo = {}

        # Extracting repository name
        name_element = soup.select_one('[itemprop="name"]')
        if name_element:
            repo["name"] = name_element.text.strip()

        # Extracting latest commit date
        latest_commit_element = soup.select_one('relative-time')
        if latest_commit_element and 'datetime' in latest_commit_element.attrs:
            repo["latest_commit"] = latest_commit_element['datetime']

        # Extracting stars, watchers, and forks
        for icon, key in [('.octicon-star', 'stars'), 
                          ('.octicon-eye', 'watchers'), 
                          ('.octicon-repo-forked', 'forks')]:
            element = soup.select_one(icon)
            if element:
                sibling = element.find_next_sibling('strong')
                if sibling:
                    repo[key] = sibling.text.strip()

        # Extract README link if exists
        main_branch_element = soup.select_one('.octicon-git-branch')
        if main_branch_element:
            main_branch_span = main_branch_element.find_next_sibling('span')
            if main_branch_span:
                main_branch = main_branch_span.text.strip()
                readme_url = f'https://raw.githubusercontent.com/{url.split("/")[-1]}/{main_branch}/README.md'
                readme_response = requests.get(readme_url)
                if readme_response.status_code == 200:
                    repo['readme'] = readme_response.text

        return repo
    except Exception as e:
        print(f"An error occurred while scraping: {e}")
        return {}

def format_to_markdown(data):
    md = markdown.Markdown()
    return md.convert(json.dumps(data))

def format_to_json(data):
    return json.dumps(data, indent=4)

def export_data(data, format_type='json'):
    if format_type == 'markdown':
        formatted_data = format_to_markdown(data)
    else:
        formatted_data = format_to_json(data)

    with open(f'repo.{format_type}', 'w') as file:
        file.write(formatted_data)

def main():
    repo_url = input("Enter GitHub repository URL: ")
    format_type = input("Choose output format (json/markdown): ").lower()

    scraped_data = scrape_github_repo(repo_url)
    export_data(scraped_data, format_type)
    print(f"Data exported in {format_type} format.")

if __name__ == "__main__":
    main()
