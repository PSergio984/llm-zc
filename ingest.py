import requests
from minsearch import Index


def load_faq_data():
    # URL pointing to the list of available courses
    docs_url = "https://datatalks.club/faq/json/courses.json"
    
    # Fetch the catalog of courses
    response = requests.get(docs_url)
    courses_raw = response.json()

    documents = []
    # Base URL for fetching individual course FAQ JSON files
    url_prefix = "https://datatalks.club/faq"

    # Fetch FAQ data for each course and combine them
    for course in courses_raw:
        # Construct url to individual course json file
        course_url = f"""{url_prefix}{course["path"]}"""
        
        # Download course FAQ document
        course_response = requests.get(course_url)
        course_response.raise_for_status()
        course_data = course_response.json()

        # Add course documents to the master collection
        documents.extend(course_data)

    return documents


def build_index(documents):
    # Initialize minsearch Index with searchable text fields and filterable keyword fields
    index = Index(
        text_fields=["question", "section", "answer"],
        keyword_fields=["course"]
    )
    # Fit/build the index with the loaded documents
    index.fit(documents)
    return index

