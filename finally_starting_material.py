"""
∀b∈B: (title(b)≠∅) ∧ (author(b)≠∅) ∧ (ddc(b)∈ℝ)
∃b∈B: “Nonfiction”∈genre(b)
|{b∈B | “dragon”∈tags(b)}| ≥ 1

English (propositional logic):
1. ∀b∈B, title(b) ≠ ∅ ∧ author(b) ≠ ∅ ∧ ddc(b) ∈ ℝ.  
2. ∃b∈B such that “Nonfiction” ∈ genre(b).  
3. The set {b∈B | “dragon” ∈ tags(b)} has cardinality ≥ 1.
"""

# --------------------------- imports ---------------------------
import json
from pathlib import Path
from typing import List, Dict
from difflib import get_close_matches
from collections import defaultdict, Counter
import requests

# UI & mapping (advanced beyond Tkinter)
import streamlit as st
import folium
from streamlit_folium import st_folium

# --------------------------- original code (unchanged) ---------------------------

# Define the path
base_path = Path("/mnt/data/library_search_system")
file_path = base_path / "library.json"

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
    """
    ∀b∈B: (title(b)≠∅) ∧ (author(b)≠∅) ∧ (ddc(b)∈ℝ)
    ∃b∈B: “Nonfiction”∈genre(b)
    |{b∈B | “dragon”∈tags(b)}| ≥ 1

    English:
    Ensures dataset integrity (titles/authors present, numeric DDC);
    guarantees at least one Nonfiction item; guarantees at least one item tagged “dragon”.
    """
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

# --------------------------- enhancements ---------------------------

def _convert_sample_json(sample: List[Dict]) -> List[Dict]:
    """
    ∀b∈sample, keys(title,author,ddc) ≠ ∅ → converted(b) has
    Title, Author, DDC Number with same values.

    English:
    For every sample record, if title/author/ddc keys are non-empty,
    the converted record preserves them in the uppercase schema.
    """
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
                "Genre": b.get("genre", []),
                "Tags": b.get("tags", []),
            }
        )
    return converted

# Sample enriched JSON
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

# Merge datasets
library_data_extended = library_data + _convert_sample_json(sample_json)
search_system_full = SmartLibrarySearch(library_data_extended)

# --------------------------- geographic library map ---------------------------

# Example branch coordinates (extend or replace with real API data as desired)
LIBRARY_BRANCHES = [
    {"name": "NYPL Main (5th Ave)", "lat": 40.7532, "lon": -73.9822},
    {"name": "Brooklyn Central",     "lat": 40.6720, "lon": -73.9687},
    {"name": "Queens Central",       "lat": 40.7070, "lon": -73.8309},
]

def create_library_map(branches: List[Dict]) -> folium.Map:
    """
    ∀l∈branches, (lat(l),lon(l)) ∈ ℝ² → placed_on_map(l)

    English:
    Every branch in the list with real-valued coordinates is placed on the folium map.
    """
    # Center map at the average coordinate
    avg_lat = sum(b["lat"] for b in branches) / len(branches)
    avg_lon = sum(b["lon"] for b in branches) / len(branches)
    fmap = folium.Map(location=[avg_lat, avg_lon], zoom_start=11)
    for br in branches:
        folium.Marker(
            location=[br["lat"], br["lon"]],
            popup=br["name"],
            icon=folium.Icon(color="blue", icon="book"),
        ).add_to(fmap)
    return fmap

# --------------------------- Streamlit UI ---------------------------

def run_app():
    """
    ∀user_query q → display_results(q) ∧ display_map

    English:
    For every user query q, the application shows the corresponding search results
    and simultaneously displays the geographical library map.
    """
    st.set_page_config(page_title="Smart Library Search + Map", layout="wide")

    st.title("📚 Smart Library Search   🗺️ + Geographical Branches")

    col1, col2 = st.columns([2, 3])

    with col1:
        st.header("Search Catalog")

        search_type = st.selectbox("Search by …", ["DDC Prefix", "Author", "Natural Language"])
        query = st.text_input("Enter your query", value="813")
        if st.button("Search"):
            if search_type == "DDC Prefix":
                results = search_system_full.search_by_ddc(query)
            elif search_type == "Author":
                results = search_system_full.search_by_author(query)
            else:
                results = search_system_full.smart_natural_search(query)

            st.subheader(f"Results ({len(results)})")
            for bk in results:
                st.markdown(
                    f"**{bk['Title']}** — {bk['Author']}  \n"
                    f"DDC {bk['DDC Number']}  \n"
                    f"{bk.get('Notes','')}"
                )

            # Quick genre/tag breakdown
            if results:
                st.write("---")
                st.subheader("Genre / Tag Profile")
                g_counter = Counter()
                t_counter = Counter()
                for bk in results:
                    g_counter.update(bk.get("Genre", []))
                    t_counter.update(bk.get("Tags", []))
                st.write("**Top Genres**:", ", ".join([f"{g} ({c})" for g, c in g_counter.most_common(5)]))
                st.write("**Top Tags**:", ", ".join([f"{t} ({c})" for t, c in t_counter.most_common(5)]))

    with col2:
        st.header("Branch Locator")
        fmap = create_library_map(LIBRARY_BRANCHES)
        st_data = st_folium(fmap, width="100%", height=600)

# Streamlit entry-point
if __name__ == "__main__":
    # Comment out run_app() if executing in environments without Streamlit
    try:
        run_app()
    except RuntimeError:
        # Streamlit may raise RuntimeError if not run via `streamlit run`
        pass
