import logging
import datetime

from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain

from langchain_community.document_loaders import YoutubeAudioLoader
from langchain_community.document_loaders.generic import GenericLoader


from langchain_community.document_loaders.parsers.audio import OpenAIWhisperParser

from langchain_community.document_loaders.parsers.audio import OpenAIWhisperParserLocal

from langchain.chains.question_answering import load_qa_chain

from langchain_community.document_loaders import RSSFeedLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
import sys

# LangChain
class LangChainAI:

    def __init__(self, 
                 model_name="gpt-3.5-turbo-16k", 
                 temperature=0.1
                 ):
        self.llm=None
        if model_name.startswith("gpt"):
            self.llm = ChatOpenAI(
                model_name=model_name, #e.g. gpt-4, gpt-3.5-turbo-16k
                temperature=temperature  # temperature dictates how whacky the output should be
                )  
        elif model_name.startswith('Llama3'):
            self.llm = ChatGroq(
                model_name=model_name, #e.g. Llama3-8b-8192
                temperature=temperature, 
                )

    def translate_text(self, text):
        prompt_template = PromptTemplate.from_template("traduci {text} in italiano.")
        prompt_template.format(text=text)
        llmchain = LLMChain(llm=self.llm, prompt=prompt_template)
        res = llmchain.run(text) + "\n\n"
        return res

    def stuff_chain(self, docs):
        """
        Making the text more understandable by clearing unreadeable stuff,
        using the chain StuffDocumentsChain:
        this chain will take a list of documents,
        inserts them all into a prompt, and passes that prompt to an LLM
        https://python.langchain.com/v0.1/docs/use_cases/summarization/#option-2-map-reduce
        """

        # Define prompt
        prompt_template = """
            Sei un agente deputato al riassumere testi.
            
            Questo è il testo fornito:
            "{text}"

            Il riassunto deve avere la seguente struttura:
            
            RIASSUNTO
            Riassunto..\n
            \n\n

            AGROMENTI TRATTATI
            1. argomento #1 \n
            2. argomento #2 \n
            ...
            \n\n
            """
        prompt = PromptTemplate.from_template(prompt_template)

        # Define LLM chain
        llm_chain = LLMChain(llm=self.llm, prompt=prompt)

        # Define StuffDocumentsChain
        stuff_chain = StuffDocumentsChain(
            llm_chain=llm_chain, document_variable_name="text"
        )
        res = stuff_chain.run(docs)
        return res

    def map_reduce_chain(self, docs):
        '''
        Map reduce chain
        https://python.langchain.com/v0.1/docs/use_cases/summarization/#option-2-map-reduce
        '''

        # map
        map_template = """Di seguito un testo lungo diviso in documenti:
        {docs}
        Basansoti su questa lista di documenti, per favore crea un riassunto per ciascuno di essi. 
        Se ci sono link che mostrano le referenze delle informazioni riportale insieme al riassunto . 
        Riassunto:"""
        map_prompt = PromptTemplate.from_template(map_template)
        map_chain = LLMChain(llm=self.llm, prompt=map_prompt)

        # Reduce
        reduce_template = """Di seguito una lista di riassunti:
        {docs}
        Prendi queste informazioni e sintetizzale in un elenco puntato finale di argomenti, che contiene i temi principali trattati.. 
        
        Riporta i punti con un formato uguale a questo: 
        
        - Argomento: <titolo>
          Link: il link con le referenze se presente nel testo (non ne inventare uno). 
          Descrizione: una breve descrizione 
        
        ....

        Elimina tutti gli argomenti non relativi all'argomento principale e che sono legati alla struttura della pagina web . 

        Risposta:"""
        reduce_prompt = PromptTemplate.from_template(reduce_template)
        reduce_chain = LLMChain(llm=self.llm, prompt=reduce_prompt)

        # Combines and iteratively reduces the mapped documents
        combine_documents_chain = StuffDocumentsChain(
            llm_chain=reduce_chain, 
            document_variable_name="docs"
        )
        reduce_documents_chain = ReduceDocumentsChain(
            # This is final chain that is called.
            combine_documents_chain=combine_documents_chain,
            # If documents exceed context for `StuffDocumentsChain`
            collapse_documents_chain=combine_documents_chain,
            # The maximum number of tokens to group documents into.
            token_max=4000,
        )

        # Combining documents by mapping a chain over them, then combining results
        map_reduce_chain = MapReduceDocumentsChain(
            # Map chain
            llm_chain=map_chain,
            # Reduce chain
            reduce_documents_chain=reduce_documents_chain,
            # The variable name in the llm_chain to put the documents in
            document_variable_name="docs",
            # Return the results of the map steps in the output
            return_intermediate_steps=False,
        )

        return map_reduce_chain.run(docs)

    def paraphrase_text(self, text):
        """
        Paraphrasing the text using the chain
        """
        prompt = PromptTemplate(
            input_variables=["long_text"],
            template="Puoi parafrasare questo testo (in italiano)? {long_text} \n\n",
        )
        llmchain = LLMChain(llm=self.llm, prompt=prompt)
        res = llmchain.run(text) + "\n\n"
        return res

    def expand_text(self, text):
        """
        Enhancing the text using the chain
        """
        prompt = PromptTemplate(
            input_variables=["long_text"],
            template="Puoi arricchiere l'esposizione di questo testo (in italiano)? {long_text} \n\n",
        )
        llmchain = LLMChain(llm=self.llm, prompt=prompt)
        res = llmchain.run(text) + "\n\n"
        return res

    def draft_text(self, text):
        """
        Makes a draft of the text using the chain
        """
        prompt = PromptTemplate(
            input_variables=["long_text"],
            template="Puoi fare una minuta della trascrizione di una riunione contenuta in questo testo (in italiano)? {long_text} \n\n",
        )
        llmchain = LLMChain(llm=self.llm, prompt=prompt)
        res = llmchain.run(text) + "\n\n"
        return res

    def chat_prompt(self, text):
        # TODO
        pass

    def extract_video(self, url):
        """
        Estrae il testo di un video da un url in ingresso
        """
        local = False
        text = ""
        save_dir = ""
        # Transcribe the videos to text
        if local:
            loader = GenericLoader(
                YoutubeAudioLoader([url], save_dir), OpenAIWhisperParserLocal()
            )
        else:
            loader = GenericLoader(
                YoutubeAudioLoader([url], save_dir), OpenAIWhisperParser()
            )
        docs = loader.load()
        for docs in docs:
            # write all the text into the var
            text += docs.page_content + "\n\n"
        return text

    def github_prompt(self, url):
        # TODO
        pass

    def summarize_repo(self, url):
        # TODO
        pass

    def generate_paragraph(self, text):
        # TODO
        pass

    def final_chain(self, user_questions):
        # Generating the final answer to the user's question using all the chains

        sentences = []

        for text in user_questions:
            # print(text)

            # Chains
            prompt = PromptTemplate(
                input_variables=["long_text"],
                template="Puoi rendere questo testo più comprensibile? {long_text} \n\n",
            )
            llmchain = LLMChain(llm=self.llm, prompt=prompt)
            res = llmchain.run(text) + "\n\n"
            print(res)
            sentences.append(res)

        print(sentences)

        # Chain 2
        template = """Puoi ordinare il testo di queste frasi secondo il significato? {sentences}\n\n"""
        prompt_template = PromptTemplate(
            input_variables=["sentences"], template=template
        )
        question_chain = LLMChain(llm=self.llm, prompt=prompt_template, verbose=True)

        # Final Chain
        template = """Puoi sintetizzare questo testo in una lista di bullet points utili per la comprensione rapida del testo? '{text}'"""
        prompt_template = PromptTemplate(input_variables=["text"], template=template)
        answer_chain = LLMChain(llm=self.llm, prompt=prompt_template)

        overall_chain = SimpleSequentialChain(
            chains=[question_chain, answer_chain],
            verbose=True,
        )

        res = overall_chain.run(sentences)

        return res

    def create_chatbot_chain(self):
        chain = load_qa_chain(self.llm, chain_type="stuff", verbose=False)
        return chain

    def filter_datetime_metadata(self, docs):
        """
        Takes a list of documents in input
        """
        for doc in docs:
            doc.metadata["source"] = "rss"
            if isinstance(doc.metadata["publish_date"], datetime.datetime):
                # print(doc.metadata['publish_date'])
                doc.metadata["publish_date"] = doc.metadata["publish_date"].strftime(
                    "%Y-%m-%d"
                )

    def filter_newline_content(self, docs):
        """
        Takes a list of documents in input
        """
        for doc in docs:
            doc.page_content = doc.page_content.replace("\n", " ")
            doc.metadata["source"] = "html"
        return docs

    def rss_loader(self, feed):
        splitted_docs = []
        urls = [feed]  # TODO: change for multiple?
        loader = RSSFeedLoader(urls=urls)
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, chunk_overlap=0
        )  # FIXME
        for doc in data:
            splitted_docs.append(text_splitter.split_documents(data))
        self.filter_datetime_metadata(splitted_docs[0])
        logging.info(
            "RSS scraping completed...scraped {} documents".format(
                len(splitted_docs[0])
            )
        )
        return splitted_docs[0]

    def webpage_loader(self, url):
        splitted_docs = []
        loader = WebBaseLoader(url)
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " "], chunk_size=2000, chunk_overlap=0
        )  # FIXME
        for doc in data:
            splitted_docs.append(text_splitter.split_documents(data))
        self.filter_newline_content(splitted_docs[0])
        logging.info(
            "Web pages scraping completed...scraped {} documents".format(
                len(splitted_docs[0])
            )
        )
        return data

    # parent document retriever
    # https://github.com/azharlabs/medium/blob/main/notebooks/LangChain_RAG_Parent_Document_Retriever.ipynb?source=post_page-----5bd5c3474a8a--------------------------------
    # def

