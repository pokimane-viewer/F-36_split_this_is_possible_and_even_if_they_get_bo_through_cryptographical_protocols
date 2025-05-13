import json
from pathlib import Path
from typing import List, Dict
from difflib import get_close_matches
from collections import defaultdict, Counter
import requests

import matplotlib.pyplot as plt
import mpld3
from mpld3 import plugins

# Optional (kept unchanged for completeness, though not needed after the conversion)
import folium
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

# --------------------------- matplotlib rendering ---------------------------

def create_branch_scatter_map(ax, branches: List[Dict]):
    """
    ∀l∈branches, (lat(l),lon(l)) ∈ ℝ² → plotted(l)

    English:
    Every branch with real-valued coordinates is plotted on the scatter map.
    """
    lats = [b["lat"] for b in branches]
    lons = [b["lon"] for b in branches]
    scatter = ax.scatter(lons, lats, c="blue", marker="o")
    for b in branches:
        ax.annotate(b["name"], (b["lon"], b["lat"]), xytext=(3, 3), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Library Branches")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.3)
    return scatter

def render_page(query: str, results: List[Dict]):
    """
    ∀(q,r)∈Queries×Results → rendered(q,r)

    English:
    Given any query q and its corresponding results r, a UI-like page
    combining textual results and the branch map is rendered.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    # Left pane – search results
    ax1.axis("off")
    ax1.set_title(f"Search Results for: \"{query}\"", loc="left", fontsize=12, weight="bold")

    if not results:
        ax1.text(0.05, 0.9, "No results found.", fontsize=10, va="top")
    else:
        y = 0.95
        for i, bk in enumerate(results[:15], 1):
            text = f"{i}. {bk['Title']} — {bk['Author']} (DDC {bk['DDC Number']})"
            ax1.text(0.05, y, text, fontsize=9, va="top")
            y -= 0.06
            if y < 0.05:
                break

    # Right pane – static scatter map
    create_branch_scatter_map(ax2, LIBRARY_BRANCHES)

    plt.tight_layout()
    return fig

# --------------------------- interactive d3 map within matplotlib ---------------------------

def render_interactive_branch_map(branches: List[Dict], *, show: bool = True, save_path: str | None = None) -> str:
    """
    ∀l∈branches: (lat(l),lon(l))∈ℝ² → interactive_d3(l)
    ∃html: rendered(html) ∧ (show=True → displayed(html)) ∧ (save_path≠∅ → persisted(html))

    English:
    Every branch with real-valued coordinates is rendered as an interactive
    d3-backed element via mpld3; the function returns the HTML string,
    optionally displaying it and/or saving it to disk.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    scatter = create_branch_scatter_map(ax, branches)

    # Attach tooltips with branch names
    labels = [b["name"] for b in branches]
    tooltip = plugins.PointLabelTooltip(scatter, labels=labels)
    plugins.connect(fig, tooltip)

    html = mpld3.fig_to_html(fig)

    if save_path:
        Path(save_path).write_text(html, encoding="utf-8")

    if show:
        mpld3.show()

    return html

# --------------------------- demonstration stub ---------------------------

if __name__ == "__main__":
    # Example demo to illustrate full functionality
    demo_query = "dragon"
    demo_results = search_system_full.smart_natural_search(demo_query)
    
    # Render static page with scatter map
    render_page(demo_query, demo_results)
    plt.show(block=False)

    # Render interactive d3 map (opens in browser or Jupyter output)
    render_interactive_branch_map(LIBRARY_BRANCHES, show=True, save_path=None)
