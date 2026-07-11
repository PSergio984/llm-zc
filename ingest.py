import requests

def fetch_datasets():
    """
    Downloads datasets from the DataTalksClub course FAQ JSON files.
    Returns a list of documents.
    """
    docs_url = "https://datatalks.club/faq/json/courses.json"
    response = requests.get(docs_url)
    response.raise_for_status()
    courses_raw = response.json()
    
    documents = []
    url_prefix = "https://datatalks.club/faq"

    for course in courses_raw:
        course_name = course.get("course")
        course_url = f"{url_prefix}{course['path']}"
        
        try:
            course_response = requests.get(course_url)
            course_response.raise_for_status()
            course_data = course_response.json()
            
            # Ensure each document has the course identifier
            for doc in course_data:
                if "course" not in doc:
                    doc["course"] = course_name
                    
            documents.extend(course_data)
        except Exception as e:
            print(f"Warning: Failed to fetch data for course {course_name} from {course_url}: {e}")

    return documents
