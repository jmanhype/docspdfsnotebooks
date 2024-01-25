import openai
import httpx
from bs4 import BeautifulSoup

# Set your OpenAI API key here
openai.api_key = 'your-api-key'

class GenericLinkUnfurler:
    def __init__(self):
        # Initialize the httpx client here if you plan to use it throughout the class
        self.client = httpx.Client()

    def unfurl(self, url: str):
        # Check if the URL is a Twitter link and convert it
        if "twitter.com" in url:
            url = self.convert_to_vxtwitter(url)
        
        metadata = self.scrape_metadata(url)
        summary = self.summarize_content(metadata['content'])
        return {
            'url': url,
            'title': metadata.get('title', ''),
            'description': metadata.get('description', ''),
            'summary': summary
        }

    def convert_to_vxtwitter(self, twitter_url: str):
        # Convert Twitter URL to vxtwitter.com link
        parts = twitter_url.split('/')
        user = parts[3]
        status_id = parts[5]
        return f"https://vxtwitter.com/{user}/status/{status_id}"

    def scrape_metadata(self, url: str):
        response = self.client.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        title = soup.find('title').get_text() if soup.find('title') else 'No title'
        description = soup.find('meta', attrs={'name': 'description'})['content'] if soup.find('meta', attrs={'name': 'description'}) else 'No description'
        return {
            'title': title,
            'description': description,
            'content': response.text
        }

    def summarize_content(self, html_content: str):
        # Convert HTML content to text
        soup = BeautifulSoup(html_content, 'html.parser')
        text_content = soup.get_text()
        # Use GPT-3 to generate a summary of the text content
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f"Summarize the following content:\n\n{text_content}\n\n",
            max_tokens=150,
            temperature=0.7
        )
        return response.choices[0].text.strip()

# Example usage
def main():
    unfurler = GenericLinkUnfurler()
    url = 'https://twitter.com/SteveMills/status/1721454191905022104'  # Example Twitter URL
    result = unfurler.unfurl(url)
    print(result)

if __name__ == "__main__":
    main()
