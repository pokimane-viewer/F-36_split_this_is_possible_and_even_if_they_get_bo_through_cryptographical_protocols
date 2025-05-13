"""
∀b∈B: (title(b)≠∅) ∧ (author(b)≠∅) ∧ (ddc(b)∈ℝ)  
∃b∈B: “Nonfiction”∈genre(b)  
|{b∈B | “dragon”∈tags(b)}| ≥ 1  

English — Propositional truths about the dataset  
• Every book record in the union of sample and fallback data has a non-empty title, author, and a numeric Dewey Decimal number.  
• At least one book in the combined collection is classified as Nonfiction.  
• There is at least one book whose tags include “dragon.”  

Real-world applicability  
Maintaining these invariants guarantees catalog completeness (patrons can always identify a work by title/author), supports nonfiction readers’ discovery needs, and preserves genre-specific access pathways (e.g., fantasy readers hunting for dragon stories).  

Sources  
Kristin Hannah’s *The Women* release 2024 :contentReference[oaicite:0]{index=0}; *Fourth Wing* dragon fantasy overview :contentReference[oaicite:1]{index=1}; *Atomic Habits* summary and framework :contentReference[oaicite:2]{index=2}; Osage murders narrative in *Killers of the Flower Moon* :contentReference[oaicite:3]{index=3}; Open Library Search API docs :contentReference[oaicite:4]{index=4}; DDC 813 American fiction scope :contentReference[oaicite:5]{index=5}; DDC 158.1 personal improvement scope :contentReference[oaicite:6]{index=6}; Rebecca Yarros publication update :contentReference[oaicite:7]{index=7}; Fiction about Vietnam War nurses list :contentReference[oaicite:8]{index=8}; Historical context of Osage murders :contentReference[oaicite:9]{index=9}
"""

import json
import os
from pathlib import Path
from typing import List, Dict
from difflib import get_close_matches
from collections import defaultdict, Counter
import requests  # External API for “useful and insightful” enrichment

# --------------------------- original code (unchanged) ---------------------------

# Define the path
base_path = Path("/mnt/data/library_search_system")
file_path = base_path / "library_data.json"

# Handle file loading with fallback
try:
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    print(f"Loading data from: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        library_data = json.load(f)
    print(f"Loaded {len(library_data)} books")

except FileNotFoundError as e:
    print(str(e))
    user_input = input("File not found. Do you want to continue with test data? (y/n): ").strip().lower()
    if user_input != 'y':
        print("Exiting program.")
        exit()
    else:
        print("Using fallback test data.")
        # Minimal fallback test data
        library_data = [
            {
                "Title": "Atomic Habits",
                "Author": "James Clear",
                "DDC Number": "158.1",
                "Notes": "A book on habit formation"
            },
            {
                "Title": "The Things We Leave Unfinished",
                "Author": "Rebecca Yarros",
                "DDC Number": "813.6",
                "Notes": "Contemporary romance"
            },
            {
                "Title": "Digital Minimalism",
                "Author": "Cal Newport",
                "DDC Number": "303.483",
                "Notes": "Nonfiction on technology and focus"
            }
        ]
        print(f"Loaded {len(library_data)} test books")

# Define the SmartLibrarySearch class
class SmartLibrarySearch:
    def __init__(self, data: List[Dict]):
        print("Initializing SmartLibrarySearch...")
        self.books = data
        self.index_by_ddc = self._index_by_ddc()
        self.index_by_author = self._index_by_author()
        print("Initialization complete.")

    def _index_by_ddc(self) -> Dict[str, List[Dict]]:
        print("Indexing by DDC...")
        ddc_index = defaultdict(list)
        for book in self.books:
            ddc = book.get("DDC Number")
            if ddc:
                ddc_index[ddc].append(book)
        print(f"DDC Index contains {len(ddc_index)} unique DDC numbers")
        return ddc_index

    def _index_by_author(self) -> Dict[str, List[Dict]]:
        print("Indexing by author...")
        author_index = defaultdict(list)
        for book in self.books:
            author = book.get("Author", "").lower()
            author_index[author].append(book)
        print(f"Author Index contains {len(author_index)} unique authors")
        return author_index

    def search_by_ddc(self, ddc_prefix: str) -> List[Dict]:
        print(f"Searching by DDC prefix: {ddc_prefix}")
        results = [book for ddc, books in self.index_by_ddc.items() if ddc.startswith(ddc_prefix) for book in books]
        print(f"Found {len(results)} books matching DDC prefix '{ddc_prefix}'")
        return results

    def search_by_author(self, author_name: str) -> List[Dict]:
        print(f"Searching by author name: {author_name}")
        matches = get_close_matches(author_name.lower(), self.index_by_author.keys(), n=3, cutoff=0.6)
        print(f"Found matches: {matches}")
        results = []
        for match in matches:
            results.extend(self.index_by_author[match])
        print(f"Total books found for author '{author_name}': {len(results)}")
        return results

    def smart_natural_search(self, query: str) -> List[Dict]:
        print(f"Performing natural language search for query: '{query}'")
        keywords = query.lower().split()
        results = []
        for book in self.books:
            text_blob = f"{book['Title']} {book['Author']} {book.get('Notes', '')}".lower()
            if all(kw in text_blob for kw in keywords):
                results.append(book)
        print(f"Natural search found {len(results)} books for query '{query}'")
        return results

# Initialize search system
search_system = SmartLibrarySearch(library_data)

# Test queries
test_ddc_results = search_system.search_by_ddc("813")
test_author_results = search_system.search_by_author("Rebecca Yarros")
test_smart_results = search_system.smart_natural_search("nonfiction habit building")

# Summarize results
summary = {
    "DDC Search (813)": [book["Title"] for book in test_ddc_results],
    "Author Search (Rebecca Yarros)": [book["Title"] for book in test_author_results],
    "Natural Language Search (nonfiction habit building)": [book["Title"] for book in test_smart_results]
}

print("\n--- Search Summaries ---")
for key, titles in summary.items():
    print(f"{key}:\n  {titles}\n")

# Final output
print(summary)

# --------------------------- enhancements for “useful & insightful” output ---------------------------

def _convert_sample_json(sample: List[Dict]) -> List[Dict]:
    """Convert lowercase-keyed sample JSON into the uppercase-keyed structure SmartLibrarySearch expects."""
    converted = []
    for b in sample:
        converted.append(
            {
                "Title": b["title"],
                "Author": b["author"],
                "DDC Number": b["ddc"],
                "Notes": "; ".join(
                    [
                        "Genres: " + ", ".join(b.get("genre", [])),
                        "Themes: " + ", ".join(b.get("themes", [])),
                        "Tags: " + ", ".join(b.get("tags", [])),
                    ]
                ),
            }
        )
    return converted

# Embed the richer sample JSON shipped with the prompt
sample_json = [
    {
        "title": "The Women",
        "author": "Kristin Hannah",
        "ddc": "813.54",
        "genre": ["Fiction", "Historical"],
        "themes": ["Women in War", "Friendship", "Resilience"],
        "tone": ["Emotional", "Heartfelt"],
        "time_period": "Vietnam War",
        "audience": "Adult",
        "setting": ["United States", "Vietnam"],
        "tags": ["historical fiction", "female protagonist", "nursing"],
    },
    {
        "title": "Fourth Wing",
        "author": "Rebecca Yarros",
        "ddc": "813.6",
        "genre": ["Fantasy", "Romance"],
        "themes": ["War", "Magic", "Love Triangle"],
        "tone": ["Adventurous", "Emotional"],
        "time_period": "Fantasy",
        "audience": "Adult",
        "setting": ["Academy", "Imaginary Kingdom"],
        "tags": ["dragon", "strong heroine", "battle school"],
    },
    {
        "title": "Atomic Habits",
        "author": "James Clear",
        "ddc": "158.1",
        "genre": ["Nonfiction", "Self-help"],
        "themes": ["Habit Building", "Self Improvement"],
        "tone": ["Practical", "Motivational"],
        "time_period": "Contemporary",
        "audience": "General",
        "setting": ["Everyday Life"],
        "tags": ["productivity", "behavior", "psychology"],
    },
    {
        "title": "Killers of the Flower Moon",
        "author": "David Grann",
        "ddc": "364.1523",
        "genre": ["Nonfiction", "True Crime"],
        "themes": ["Murder", "Corruption", "Justice"],
        "tone": ["Serious", "Investigative"],
        "time_period": "1920s",
        "audience": "Adult",
        "setting": ["Oklahoma"],
        "tags": ["FBI", "Osage murders", "history"],
    },
]

# Merge original data with converted sample
library_data_extended = library_data + _convert_sample_json(sample_json)

# Re-index with enriched data
search_system_full = SmartLibrarySearch(library_data_extended)

# Insight #1 – Most common genres
genre_counter = Counter()
for book in sample_json:
    genre_counter.update(book["genre"])

print("\nTop genres in the sample JSON:")
for genre, count in genre_counter.most_common(5):
    print(f"  • {genre}: {count}")

# Insight #2 – Fetch first-publish-year from Open Library for each title
def fetch_publish_year(title: str) -> str:
    """Return the first publish year from Open Library Search API (or 'N/A')."""
    try:
        resp = requests.get("https://openlibrary.org/search.json", params={"title": title, "limit": 1}, timeout=5)
        if resp.ok:
            docs = resp.json().get("docs", [])
            if docs:
                return str(docs[0].get("first_publish_year", "N/A"))
    except Exception:
        pass
    return "N/A"

print("\nFirst known publication year via Open Library:")
for b in sample_json:
    year = fetch_publish_year(b["title"])
    print(f"  • {b['title']}: {year}")

# Insight #3 – Count of Dewey classes represented
ddc_classes = {bk["ddc"].split(".")[0] for bk in sample_json}
print(f"\nNumber of distinct Dewey hundreds classes represented: {len(ddc_classes)} ({', '.join(sorted(ddc_classes))})")
