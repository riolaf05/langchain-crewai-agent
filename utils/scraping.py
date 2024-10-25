from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from duckduckgo_search import DDGS

def get_page(urls):
    loader = AsyncChromiumLoader(urls)
    html = loader.load()

    bs_transformer = BeautifulSoupTransformer()
    docs_transformed = bs_transformer.transform_documents(html, tags_to_extract=["p"], remove_unwanted_tags=["a"])

    return docs_transformed

# helper function to reduce the amount of text
def truncate(text):
    words = text.split()
    truncated = " ".join(words[:400])

    return truncated


def ddg_search(query, max_results=5):
    '''
    Performs a DuckDuckGo search for the given query and returns a list of URLs.
    Truncates text to ensure they fit within the context window of the LLM.
    '''

    #queries DuckDuckGo
    results = DDGS().text(query, max_results)
    urls = []
    for result in results:
        url = result['href']
        urls.append(url)

    docs = get_page(urls)

    content = []
    for doc in docs:
        page_text = re.sub("\n\n+", "\n", doc.page_content)
        text = truncate(page_text)
        content.append(text)

    return content