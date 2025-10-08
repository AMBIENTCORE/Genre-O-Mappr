"""
Install deps:
    pip install mutagen matplotlib
"""

from __future__ import annotations
import os
import math
import threading
import queue
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set

# GUI
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# Tag parsing
from mutagen import File as MutagenFile
from mutagen.id3 import ID3
from mutagen.easymp4 import EasyMP4
from mutagen.flac import FLAC

# Viz
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Configure matplotlib for dark theme
import matplotlib.pyplot as plt
plt.style.use('dark_background')

AUDIO_EXTENSIONS: Set[str] = {
    ".mp3", ".flac", ".ogg", ".m4a", ".mp4", ".wav", ".aiff", ".aif", ".aifc", ".wv", ".ape"
}

# ---------------- Data model ----------------
@dataclass
class ParseResult:
    genre_counts: Dict[str, int] = field(default_factory=dict)
    cooccurrence: Dict[Tuple[str, str], int] = field(default_factory=dict)
    genre_files: Dict[str, List[str]] = field(default_factory=dict)  # NEW: genre -> file paths
    total_files: int = 0
    processed_files: int = 0

# ---------------- Helpers ----------------

def split_genres(raw: str) -> List[str]:
    """Split a raw genre string into individual genres.
    Primary delimiter is semicolon ';' (required). Also tolerate commas inside fragments.
    Return unique (per file), case‑insensitive, preserving original case of first occurrence.
    """
    if not raw:
        return []
    parts: List[str] = []
    for frag in raw.split(";"):
        frag = frag.strip()
        if not frag:
            continue
        if "," in frag:
            for sub in frag.split(","):
                sub = sub.strip()
                if sub:
                    parts.append(sub)
        else:
            parts.append(frag)
    seen = set()
    out: List[str] = []
    for p in parts:
        k = p.lower()
        if k not in seen:
            out.append(p)
            seen.add(k)
    return out


def extract_genres(path: str) -> List[str]:
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".mp3":
            audio = ID3(path)
            raw = ";".join([str(x) for x in audio.getall("TCON")]) if audio else ""
            return split_genres(raw)
        elif ext in {".m4a", ".mp4"}:
            audio = EasyMP4(path)
            raw = ";".join(audio.get("genre", []))
            return split_genres(raw)
        elif ext == ".flac":
            audio = FLAC(path)
            raw = ";".join(audio.get("genre", []))
            return split_genres(raw)
        else:
            audio = MutagenFile(path, easy=True)
            if audio is None:
                return []
            raw = ";".join(audio.get("genre", []))
            return split_genres(raw)
    except Exception:
        return []


def walk_audio_files(root: str) -> List[str]:
    files: List[str] = []
    for base, _, names in os.walk(root):
        for n in names:
            if os.path.splitext(n)[1].lower() in AUDIO_EXTENSIONS:
                files.append(os.path.join(base, n))
    return files

# ---------------- Worker ----------------
class ParseWorker(threading.Thread):
    def __init__(self, folders: List[str], outq: queue.Queue):
        super().__init__(daemon=True)
        self.folders = folders
        self.q = outq
        self.stop_flag = threading.Event()

    def run(self):
        all_files: List[str] = []
        for f in self.folders:
            if self.stop_flag.is_set():
                self.q.put(("stopped", None))
                return
            all_files.extend(walk_audio_files(f))
        total = len(all_files)
        result = ParseResult(total_files=total)
        self.q.put(("total", total))

        for i, path in enumerate(all_files, start=1):
            if self.stop_flag.is_set():
                self.q.put(("stopped", result))
                return
            genres = extract_genres(path)
            if genres:
                for g in genres:
                    result.genre_counts[g] = result.genre_counts.get(g, 0) + 1
                    result.genre_files.setdefault(g, []).append(path)  # collect file path
                # co‑occurrence for each unordered pair
                if len(genres) > 1:
                    for a_i in range(len(genres)):
                        for b_i in range(a_i + 1, len(genres)):
                            a, b = genres[a_i], genres[b_i]
                            key = tuple(sorted((a, b)))
                            result.cooccurrence[key] = result.cooccurrence.get(key, 0) + 1
            result.processed_files = i
            if i % 10 == 0 or i == total:
                self.q.put(("progress", i))
        self.q.put(("done", result))
    
    def stop(self):
        """Signal the worker to stop parsing."""
        self.stop_flag.set()

# ---------------- Layout helpers (map tab) ----------------

def get_genre_color(count: int, min_count: int, max_count: int) -> str:
    """Convert track count to RGB color using a 15-color green-to-red gradient.
    
    Uses logarithmic scaling to better distribute colors across the data range.
    
    Args:
        count: Number of tracks for this genre
        min_count: Minimum track count across all genres
        max_count: Maximum track count across all genres
    
    Returns:
        Hex color string from green (few tracks) to red (many tracks)
    """
    if max_count == min_count:
        # All genres have the same count, use middle color (yellow)
        return "#FFFF00"
    
    # Use logarithmic scaling to better distribute colors
    # This helps when there are a few very popular genres and many less popular ones
    import math
    
    # Add 1 to avoid log(0) and ensure we get a good distribution
    log_min = math.log(min_count + 1)
    log_max = math.log(max_count + 1)
    log_count = math.log(count + 1)
    
    # Normalize using logarithmic scale
    if log_max == log_min:
        normalized = 0.5  # Middle color if all counts are the same
    else:
        normalized = (log_count - log_min) / (log_max - log_min)
    
    # 15-color gradient from green to red with better color distribution
    colors = [
        "#00FF00",  # Pure Green
        "#22FF22",  # Green
        "#44FF44",  # Light Green
        "#66FF66",  # Bright Green
        "#88FF88",  # Lime Green
        "#AAFFAA",  # Light Lime
        "#CCFFCC",  # Very Light Green
        "#EEFFEE",  # Almost White Green
        "#FFFF00",  # Pure Yellow
        "#FFDD00",  # Gold
        "#FFBB00",  # Orange Yellow
        "#FF9900",  # Orange
        "#FF7700",  # Red Orange
        "#FF5500",  # Bright Red Orange
        "#FF0000"   # Pure Red
    ]
    
    # Map normalized value (0-1) to color index (0-14)
    color_index = int(normalized * (len(colors) - 1))
    # Ensure we don't go out of bounds
    color_index = min(max(color_index, 0), len(colors) - 1)
    
    return colors[color_index]

def force_directed_layout(genres: List[str], cooccurrence: Dict[Tuple[str, str], int], 
                         genre_counts: Dict[str, int], width: float = 50.0, height: float = 30.0) -> Dict[str, Tuple[float, float]]:
    """Use force-directed layout to position genres, clustering linked genres together.
    
    Similar to Every Noise at Once but with relationship-based clustering.
    Improved to better distribute smaller genres and space larger genres.
    """
    if not genres:
        return {}
    
    import random
    
    # Calculate genre properties for better layout
    max_count = max(genre_counts.values()) if genre_counts else 1
    genre_connections = {genre: 0 for genre in genres}
    for (genre_a, genre_b), co_count in cooccurrence.items():
        if genre_a in genre_connections:
            genre_connections[genre_a] += co_count
        if genre_b in genre_connections:
            genre_connections[genre_b] += co_count
    
    max_connections = max(genre_connections.values()) if genre_connections else 1
    
    # Initialize random positions with slight bias toward center for better initial distribution
    positions = {}
    for genre in genres:
        # Slightly more spread out initial positions
        x = random.uniform(-width/2.2, width/2.2)
        y = random.uniform(-height/2.2, height/2.2)
        positions[genre] = (x, y)
    
    # Force-directed algorithm parameters
    iterations = 250  # Balanced iterations for good convergence without being too slow
    cooling_factor = 0.96
    initial_temperature = 1.2
    
    temperature = initial_temperature
    
    for iteration in range(iterations):
        forces = {genre: (0.0, 0.0) for genre in genres}
        
        # Attractive forces between linked genres
        for (genre_a, genre_b), co_count in cooccurrence.items():
            if genre_a in positions and genre_b in positions:
                x1, y1 = positions[genre_a]
                x2, y2 = positions[genre_b]
                
                dx = x2 - x1
                dy = y2 - y1
                distance = math.hypot(dx, dy)
                
                if distance > 0.001:  # Avoid division by zero
                    # Moderate attractive force for connected genres
                    attraction_strength = (co_count / 2.5) * temperature  # Reduced from 1.2 to 2.5
                    # Target distance based on connection strength - larger target distances to prevent tight clustering
                    target_distance = 1.2 + (0.8 / (co_count + 1))  # Increased from 0.4+0.3 to 1.2+0.8
                    
                    if distance > target_distance:
                        force_magnitude = attraction_strength * (distance - target_distance) / distance
                        fx = force_magnitude * dx
                        fy = force_magnitude * dy
                        
                        forces[genre_a] = (forces[genre_a][0] + fx, forces[genre_a][1] + fy)
                        forces[genre_b] = (forces[genre_b][0] - fx, forces[genre_b][1] - fy)
        
        # Enhanced repulsive forces with size and connection-based scaling
        for i, genre_a in enumerate(genres):
            for j, genre_b in enumerate(genres[i+1:], i+1):
                x1, y1 = positions[genre_a]
                x2, y2 = positions[genre_b]
                
                dx = x2 - x1
                dy = y2 - y1
                distance = math.hypot(dx, dy)
                
                if distance > 0.001:  # Avoid division by zero
                    # Calculate size-based minimum distance with enhanced scaling
                    size_a = (genre_counts.get(genre_a, 1) / max_count) * 3.0 + 1.0  # Increased from 2.0+0.5 to 3.0+1.0
                    size_b = (genre_counts.get(genre_b, 1) / max_count) * 3.0 + 1.0
                    min_distance = size_a + size_b
                    
                    # Calculate connection-based repulsion strength - enhanced for highly connected genres
                    connections_a = genre_connections[genre_a]
                    connections_b = genre_connections[genre_b]
                    connection_factor = (connections_a + connections_b) / (max_connections * 2) + 1.0  # Increased from 0.5 to 1.0
                    
                    if distance < min_distance:
                        # Strong repulsion for overlapping genres
                        repulsion_strength = temperature * connection_factor * 1.2 * (min_distance - distance) / distance
                        fx = repulsion_strength * dx
                        fy = repulsion_strength * dy
                        
                        forces[genre_a] = (forces[genre_a][0] - fx, forces[genre_a][1] - fy)
                        forces[genre_b] = (forces[genre_b][0] + fx, forces[genre_b][1] + fy)
                    elif distance < min_distance * 4:  # Extended medium-range repulsion
                        # Enhanced repulsion for nearby genres
                        repulsion_strength = temperature * connection_factor * 0.8 * (min_distance * 4 - distance) / distance  # Increased from 0.5 to 0.8
                        fx = repulsion_strength * dx
                        fy = repulsion_strength * dy
                        
                        forces[genre_a] = (forces[genre_a][0] - fx, forces[genre_a][1] - fy)
                        forces[genre_b] = (forces[genre_b][0] + fx, forces[genre_b][1] + fy)
        
        # Enhanced spreading force for isolated genres and large genres
        # Skip if there's only one genre (can't calculate center of mass)
        if len(genres) > 1:
            for genre in genres:
                connections = genre_connections[genre]
                genre_size = genre_counts.get(genre, 1) / max_count
                
                # Apply spreading force to both isolated genres AND large genres
                if connections < max_connections * 0.15 or genre_size > 0.3:  # Increased threshold and added size condition
                    x, y = positions[genre]
                    
                    # Calculate center of mass of all other genres
                    center_x = sum(pos[0] for g, pos in positions.items() if g != genre) / (len(genres) - 1)
                    center_y = sum(pos[1] for g, pos in positions.items() if g != genre) / (len(genres) - 1)
                    
                    # Push away from center
                    dx = x - center_x
                    dy = y - center_y
                    distance_from_center = math.hypot(dx, dy)
                    
                    if distance_from_center > 0.001:
                        # Enhanced spreading force based on size and connections
                        base_spread = 0.3 + genre_size * 0.4  # Stronger spreading for larger genres
                        spread_strength = temperature * base_spread * (1.0 - connections / max_connections)
                        fx = spread_strength * dx / distance_from_center
                        fy = spread_strength * dy / distance_from_center
                        
                        forces[genre] = (forces[genre][0] + fx, forces[genre][1] + fy)
        
        # Apply forces with cooling
        for genre in genres:
            fx, fy = forces[genre]
            
            # Limit force magnitude to prevent instability
            force_magnitude = math.hypot(fx, fy)
            if force_magnitude > temperature:
                fx = fx / force_magnitude * temperature
                fy = fy / force_magnitude * temperature
            
            # Update position
            x, y = positions[genre]
            new_x = x + fx
            new_y = y + fy
            
            # Keep within bounds with some margin
            margin = 2.0
            new_x = max(-width/2 + margin, min(width/2 - margin, new_x))
            new_y = max(-height/2 + margin, min(height/2 - margin, new_y))
            
            positions[genre] = (new_x, new_y)
        
        # Cool down the temperature
        temperature *= cooling_factor
    
    return positions


def enforce_no_overlaps(positions: Dict[str, Tuple[float, float]], 
                       genre_counts: Dict[str, int], 
                       max_x: float = 48.0, max_y: float = 28.0) -> Dict[str, Tuple[float, float]]:
    """Post-process positions to guarantee no overlaps using iterative repulsion.
    
    Enhanced to work better with size-based spacing and connection-aware layout.
    """
    if not positions:
        return {}
    
    genres = list(positions.keys())
    final_positions = dict(positions)
    
    # Calculate enhanced label sizes based on genre properties
    max_count = max(genre_counts.values()) if genre_counts else 1
    label_sizes = {}
    for genre in genres:
        # Base size on text length and popularity
        name_length = len(genre)
        popularity = genre_counts.get(genre, 1)
        
        # Size factor based on popularity (larger genres need more space)
        popularity_factor = (popularity / max_count) * 2.0 + 0.8  # Increased from 1.5+0.5 to 2.0+0.8
        
        # Text length factor
        text_factor = name_length * 0.08  # Restored to 0.08 for better text spacing
        
        # Combined size with better scaling
        raw_size = text_factor + popularity_factor
        label_size = max(1.0, min(3.0, raw_size))  # Increased range: 1.0 to 3.0
        label_sizes[genre] = label_size
    
    # Iterative overlap resolution with improved algorithm
    max_iterations = 400  # Balanced iterations for good results without being too slow
    overlap_tolerance = 0.05  # Stricter tolerance to eliminate overlaps
    
    for iteration in range(max_iterations):
        overlaps_found = False
        
        for i, genre_a in enumerate(genres):
            for j, genre_b in enumerate(genres[i+1:], i+1):
                x1, y1 = final_positions[genre_a]
                x2, y2 = final_positions[genre_b]
                
                # Calculate required minimum distance with better padding
                size_a = label_sizes[genre_a]
                size_b = label_sizes[genre_b]
                min_distance = size_a + size_b + 1.0  # Even more aggressive padding to prevent overlaps
                
                # Calculate current distance
                dx = x2 - x1
                dy = y2 - y1
                current_distance = math.hypot(dx, dy)
                
                if current_distance < min_distance - overlap_tolerance:
                    overlaps_found = True
                    
                    # Calculate separation vector
                    if current_distance < 0.001:  # Avoid division by zero
                        # Random direction if too close
                        import random
                        angle = random.uniform(0, 2 * math.pi)
                        dx = math.cos(angle)
                        dy = math.sin(angle)
                        current_distance = 0.001
                    
                    # Normalize direction
                    dx /= current_distance
                    dy /= current_distance
                    
                    # Calculate how much each genre needs to move
                    separation_needed = min_distance - current_distance
                    move_distance = separation_needed / 2.0
                    
                    # Enhanced movement weighting
                    # Smaller genres move more, but also consider popularity
                    popularity_a = genre_counts.get(genre_a, 1)
                    popularity_b = genre_counts.get(genre_b, 1)
                    
                    # Weight by inverse size and popularity (smaller, less popular moves more)
                    weight_a = (size_b / size_a) * (popularity_b / max(popularity_a, 1))
                    weight_b = (size_a / size_b) * (popularity_a / max(popularity_b, 1))
                    
                    total_weight = weight_a + weight_b
                    if total_weight > 0:
                        move_a = move_distance * (weight_a / total_weight)
                        move_b = move_distance * (weight_b / total_weight)
                    else:
                        move_a = move_distance * 0.5
                        move_b = move_distance * 0.5
                    
                    # Update positions
                    new_x1 = x1 - dx * move_a
                    new_y1 = y1 - dy * move_a
                    new_x2 = x2 + dx * move_b
                    new_y2 = y2 + dy * move_b
                    
                    final_positions[genre_a] = (new_x1, new_y1)
                    final_positions[genre_b] = (new_x2, new_y2)
        
        if not overlaps_found:
            break
    
    # Final pass: ensure all genres are within reasonable bounds
    # Use the provided bounds with some margin
    
    for genre in genres:
        x, y = final_positions[genre]
        x = max(-max_x, min(max_x, x))
        y = max(-max_y, min(max_y, y))
        final_positions[genre] = (x, y)
    
    return final_positions

# ---------------- GUI ----------------
class GenreExplorer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Genre-o-Node")
        self.state('zoomed')  # Start maximized on Windows
        
        # Configure dark theme
        self.configure(bg='#2b2b2b')
        
        # Configure ttk style for dark theme
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Configure dark colors for ttk widgets
        self.style.configure('TFrame', background='#2b2b2b')
        self.style.configure('TLabel', background='#2b2b2b', foreground='#ffffff')
        self.style.configure('TButton', background='#404040', foreground='#ffffff')
        self.style.configure('TNotebook', background='#2b2b2b')
        self.style.configure('TNotebook.Tab', background='#404040', foreground='#ffffff', padding=[10, 5])
        self.style.configure('Treeview', background='#3c3c3c', foreground='#ffffff', fieldbackground='#3c3c3c')
        self.style.configure('Treeview.Heading', background='#404040', foreground='#ffffff')
        self.style.configure('TScrollbar', background='#404040', troughcolor='#2b2b2b')
        self.style.configure('TProgressbar', background='#00ff00', troughcolor='#2b2b2b')

        # State
        self.folders: List[str] = []
        self.result: ParseResult | None = None
        self.q: queue.Queue = queue.Queue()
        self.current_worker: ParseWorker | None = None

        # Top controls
        top = ttk.Frame(self)
        top.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        ttk.Button(top, text="Import Folder", command=self.on_add_folder).pack(side=tk.LEFT)
        ttk.Button(top, text="Clear Folders", command=self.on_clear_folders).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(top, text="Parse Data", command=self.on_parse).pack(side=tk.LEFT, padx=(8, 0))
        self.stop_button = ttk.Button(top, text="Stop Parsing", command=self.on_stop_parse, state="disabled")
        self.stop_button.pack(side=tk.LEFT, padx=(8, 0))

        self.folders_var = tk.StringVar(value="No folders selected.")
        ttk.Label(self, textvariable=self.folders_var, anchor="w", justify="left").pack(side=tk.TOP, fill=tk.X, padx=10)

        prog = ttk.Frame(self)
        prog.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(0, 0))
        self.progress = ttk.Progressbar(prog, mode="determinate")
        self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.status_var = tk.StringVar(value="Idle")
        ttk.Label(prog, textvariable=self.status_var, width=32).pack(side=tk.LEFT, padx=(10, 0))

        # Tabs
        self.tabs = ttk.Notebook(self)
        self.tabs.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.info_tab = ttk.Frame(self.tabs)
        self.map_tab = ttk.Frame(self.tabs)
        self.tabs.add(self.map_tab, text="Visual Map")
        self.tabs.add(self.info_tab, text="Information")

        self._build_info_tab()
        self._build_map_tab()

        # Queue polling
        self.after(100, self._poll_queue)

    # ----- Top controls -----
    def on_add_folder(self):
        path = filedialog.askdirectory(title="Choose a music folder")
        if path:
            # Normalize path for consistent comparison
            path = os.path.normpath(path)
            
            # Check for exact duplicates
            if path in self.folders:
                messagebox.showwarning("Duplicate Folder", f"This folder has already been added:\n{path}")
                return
            
            # Check if any existing folder is a parent of the new path
            for existing in self.folders:
                if path.startswith(existing + os.sep) or path == existing:
                    messagebox.showwarning("Duplicate Folder", 
                                         f"A parent folder is already added:\n{existing}\n\n"
                                         f"This includes the folder you're trying to add:\n{path}")
                    return
            
            # Check if the new path is a parent of any existing folder
            folders_to_remove = []
            for existing in self.folders:
                if existing.startswith(path + os.sep):
                    folders_to_remove.append(existing)
            
            if folders_to_remove:
                msg = f"The folder you're adding is a parent of {len(folders_to_remove)} existing folder(s):\n\n"
                msg += "\n".join(folders_to_remove[:5])  # Show first 5
                if len(folders_to_remove) > 5:
                    msg += f"\n... and {len(folders_to_remove) - 5} more"
                msg += "\n\nDo you want to remove the child folder(s) and add the parent folder instead?"
                
                if messagebox.askyesno("Parent Folder Detected", msg):
                    for folder in folders_to_remove:
                        self.folders.remove(folder)
                else:
                    return
            
            self.folders.append(path)
            self.folders_var.set("; ".join(self.folders))

    def on_clear_folders(self):
        self.folders.clear()
        self.folders_var.set("No folders selected.")
        self.result = None
        self._reset_progress()
        self._clear_info()
        self._clear_map()

    def on_parse(self):
        if not self.folders:
            messagebox.showinfo("No folders", "Please import at least one folder.")
            return
        self._reset_progress()
        self.status_var.set("Indexing files…")
        self.current_worker = ParseWorker(self.folders.copy(), self.q)
        self.current_worker.start()
        self.stop_button.configure(state="normal")

    def on_stop_parse(self):
        """Stop the current parsing operation."""
        if self.current_worker:
            self.current_worker.stop()
            self.status_var.set("Stopping...")

    def _reset_progress(self):
        self.progress.configure(value=0, maximum=100)
        self.status_var.set("Idle")
        self.current_worker = None
        self.stop_button.configure(state="disabled")

    def _poll_queue(self):
        try:
            while True:
                msg, payload = self.q.get_nowait()
                if msg == "total":
                    total = max(1, int(payload))
                    self.progress.configure(maximum=total)
                    self.status_var.set(f"Found {total} audio files…")
                elif msg == "progress":
                    processed = int(payload)
                    self.progress.configure(value=processed)
                    self.status_var.set(f"Parsing… {processed}/{int(self.progress['maximum'])}")
                elif msg == "done":
                    self.result = payload
                    self.status_var.set("Parsing complete.")
                    self._refresh_info()
                    self._refresh_map()
                    self.current_worker = None
                    self.stop_button.configure(state="disabled")
                elif msg == "stopped":
                    if payload:  # Partial results available
                        self.result = payload
                        self.status_var.set("Parsing stopped.")
                        self._refresh_info()
                        self._refresh_map()
                    else:
                        self.status_var.set("Parsing stopped.")
                    self.current_worker = None
                    self.stop_button.configure(state="disabled")
                self.q.task_done()
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)

    # ----- Information tab -----
    def _build_info_tab(self):
        left = ttk.Frame(self.info_tab)
        right = ttk.Frame(self.info_tab)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Table
        table_container = ttk.Frame(left)
        table_container.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.genre_tree = ttk.Treeview(table_container, columns=("genre", "count"), show="headings", height=26)
        self.genre_tree.heading("genre", text="Genre")
        self.genre_tree.heading("count", text="# Tracks")
        self.genre_tree.column("genre", width=400)
        self.genre_tree.column("count", width=90, anchor="e")
        yscroll = ttk.Scrollbar(table_container, orient="vertical", command=self.genre_tree.yview)
        self.genre_tree.configure(yscrollcommand=yscroll.set)
        self.genre_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        yscroll.pack(side=tk.LEFT, fill=tk.Y)

        actions = ttk.Frame(left)
        actions.pack(side=tk.TOP, fill=tk.X, padx=6, pady=(0,6))
        ttk.Button(actions, text="Show Files for Selected", command=self._show_files_for_selected).pack(side=tk.LEFT)

        # Pie chart
        self.info_fig = Figure(figsize=(5.4, 4.6), dpi=100, facecolor='#2b2b2b')
        self.info_ax = self.info_fig.add_subplot(111)
        self.info_ax.set_facecolor('#2b2b2b')
        # Hide axes initially to prevent placeholder display
        self.info_ax.axis('off')
        # Show "Pending data" message initially
        self.info_ax.text(0.5, 0.5, "Pending data", ha="center", va="center", color='white', fontsize=14)
        self.info_canvas = FigureCanvasTkAgg(self.info_fig, master=right)
        self.info_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=6, pady=6)

        self.genre_tree.bind("<Double-1>", lambda e: self._show_files_for_selected())

    def _clear_info(self):
        if hasattr(self, 'genre_tree'):
            for row in self.genre_tree.get_children():
                self.genre_tree.delete(row)
        if hasattr(self, 'info_ax'):
            self.info_ax.clear()
            self.info_ax.set_facecolor('#2b2b2b')
            self.info_ax.axis('off')
            self.info_ax.text(0.5, 0.5, "Pending data", ha="center", va="center", color='white', fontsize=14)
            self.info_canvas.draw()

    def _refresh_info(self):
        if not self.result:
            self._clear_info()
            return
        items = sorted(self.result.genre_counts.items(), key=lambda x: x[1], reverse=True)
        self._clear_info()
        for g, c in items:
            self.genre_tree.insert("", tk.END, values=(g, c))
        if items:
            self.info_ax.clear()
            self.info_ax.set_facecolor('#2b2b2b')
            topN = 15
            labels = [g for g, _ in items[:topN]]
            sizes = [c for _, c in items[:topN]]
            other = sum(c for _, c in items[topN:])
            if other:
                labels.append("Other")
                sizes.append(other)
            self.info_ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, textprops={'color': 'white'})
            self.info_ax.axis('equal')
            self.info_canvas.draw()
        else:
            self.info_ax.clear()
            self.info_ax.set_facecolor('#2b2b2b')
            self.info_ax.axis('off')
            self.info_ax.text(0.5, 0.5, "Pending data", ha="center", va="center", color='white', fontsize=14)
            self.info_canvas.draw()

    def _show_files_for_selected(self):
        if not self.result:
            return
        sel = self.genre_tree.selection()
        if not sel:
            return
        genre = self.genre_tree.item(sel[0], 'values')[0]
        paths = self.result.genre_files.get(genre, [])
        self._create_files_popup(f"Files for {genre} ({len(paths)})", paths)

    # ----- Visual map tab -----
    def _build_map_tab(self):
        # Add tip text at the top
        tip_frame = ttk.Frame(self.map_tab)
        tip_frame.pack(side=tk.TOP, fill=tk.X, padx=6, pady=(6, 0))
        tip_text = "💡 Tip: Double-click on genre labels to see their tracks, or on links to see tracks with both genres"
        tip_label = ttk.Label(tip_frame, text=tip_text, font=('Arial', 9), foreground='#cccccc')
        tip_label.pack(side=tk.LEFT)
        
        self.map_fig = Figure(figsize=(7.6, 6.6), dpi=100, facecolor='#2b2b2b')
        self.map_ax = self.map_fig.add_subplot(111)
        self.map_ax.set_facecolor('#1e1e1e')
        # Hide axes initially to prevent placeholder display
        self.map_ax.axis('off')
        # Show "Pending data" message initially
        self.map_ax.text(0.5, 0.5, "Pending data", ha="center", va="center", color='white', fontsize=14)
        self.map_canvas = FigureCanvasTkAgg(self.map_fig, master=self.map_tab)
        self.map_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=6, pady=6)
        
        # Zoom and pan state
        self.zoom_level = 1.0
        self.base_xlim = (-25.0, 25.0)
        self.base_ylim = (-15.0, 15.0)
        self.pan_offset_x = 0.0
        self.pan_offset_y = 0.0
        
        # Mouse state for panning
        self.pan_start_x = 0
        self.pan_start_y = 0
        self.is_panning = False
        
        # Bind mouse events
        self.map_canvas.get_tk_widget().bind("<MouseWheel>", self._on_mouse_wheel)
        self.map_canvas.get_tk_widget().bind("<Button-4>", self._on_mouse_wheel)  # Linux
        self.map_canvas.get_tk_widget().bind("<Button-5>", self._on_mouse_wheel)  # Linux
        
        # Pan events
        self.map_canvas.get_tk_widget().bind("<Button-1>", self._on_mouse_press)
        self.map_canvas.get_tk_widget().bind("<B1-Motion>", self._on_mouse_drag)
        self.map_canvas.get_tk_widget().bind("<ButtonRelease-1>", self._on_mouse_release)

    def _clear_map(self):
        # Clear the stored object references to prevent stale object access
        self.text_objects = {}
        self.line_objects = {}
        
        self.map_ax.clear()
        self.map_ax.set_facecolor('#1e1e1e')
        self.map_ax.axis('off')
        self.map_ax.text(0.5, 0.5, "Pending data", ha="center", va="center", color='white', fontsize=14)
        self.map_canvas.draw()

    def _refresh_map(self):
        if not self.result or not self.result.genre_counts:
            self._clear_map()
            return
        
        # Clear the map without showing "Pending data" since we have valid data
        self.map_ax.clear()
        self.map_ax.set_facecolor('#1e1e1e')
        self.map_ax.axis('off')
        
        # Get all genres (no particular order needed for Every Noise style)
        genres = list(self.result.genre_counts.keys())
        
        # Use a wider grid to better utilize horizontal space
        # Fixed dimensions that work well for most screen sizes
        base_width = 70.0   # Increased from 50.0 for better horizontal utilization
        base_height = 30.0  # Keep height reasonable
        
        # Use force-directed layout to cluster linked genres
        positions = force_directed_layout(genres, self.result.cooccurrence, self.result.genre_counts, width=base_width, height=base_height)
        
        # Enforce no overlaps as a post-processing step
        positions = enforce_no_overlaps(positions, self.result.genre_counts, base_width * 0.96, base_height * 0.93)
            
        
        # Draw links between co-occurring genres first (so they appear behind labels)
        if hasattr(self, 'result') and self.result.cooccurrence:
            max_cooccurrence = max(self.result.cooccurrence.values()) if self.result.cooccurrence else 1
            
            # Store line objects for hover effects
            self.line_objects = {}
            
            for (genre_a, genre_b), co_count in self.result.cooccurrence.items():
                if genre_a in positions and genre_b in positions:
                    x1, y1 = positions[genre_a]
                    x2, y2 = positions[genre_b]
                    
                    # Line thickness based on co-occurrence frequency
                    line_width = 0.5 + (co_count / max_cooccurrence) * 2.5  # 0.5 to 3.0
                    alpha = 0.3 + (co_count / max_cooccurrence) * 0.4  # 0.3 to 0.7
                    
                    # Draw the connection line with clipping disabled
                    line = self.map_ax.plot([x1, x2], [y1, y2], 
                                          color='gray', 
                                          linewidth=line_width, 
                                          alpha=alpha,
                                          zorder=1)[0]  # Behind labels
                    # Disable clipping so lines draw across the entire grid
                    line.set_clip_on(False)
                    
                    # Store reference for hover effects
                    self.line_objects[(genre_a, genre_b)] = line
        
        # Draw floating labels (like Every Noise at Once)
        # Calculate min and max counts for color mapping
        counts = list(self.result.genre_counts.values())
        min_count = min(counts) if counts else 1
        max_count = max(counts) if counts else 1
        
        # Store text objects for hover effects
        self.text_objects = {}
        
        for genre in genres:
            x, y = positions[genre]
            count = self.result.genre_counts[genre]
            
            # Size and opacity based on popularity
            size_factor = 8 + (count / max_count) * 6  # Font size 8-14
            alpha_factor = 0.7 + (count / max_count) * 0.3  # Alpha 0.7-1.0
            
            # Get color based on track count (green = few tracks, red = many tracks)
            genre_color = get_genre_color(count, min_count, max_count)
            
            # Draw the genre label (just the name, no count)
            text_artist = self.map_ax.text(x, y, genre, 
                                         fontsize=size_factor, 
                                         ha='center', va='center',
                                         color=genre_color,
                                         alpha=alpha_factor,
                                         fontweight='normal',
                                         fontfamily='sans-serif',
                                         zorder=10)  # In front of links
            
            # Store reference for hover effects
            self.text_objects[genre] = text_artist
        

        # Set initial bounds to show the full dynamic grid
        self.map_ax.set_xlim(-base_width/2, base_width/2)
        self.map_ax.set_ylim(-base_height/2, base_height/2)
        
        # Update the base bounds to match the actual grid
        self.base_xlim = (-base_width/2, base_width/2)
        self.base_ylim = (-base_height/2, base_height/2)
        
        # Clean, minimal styling like Every Noise
        self.map_ax.set_aspect('equal')
        self.map_ax.set_facecolor('#1e1e1e')  # Dark background
        
        # Remove all axes, ticks, and grid for clean look
        self.map_ax.set_xticks([])
        self.map_ax.set_yticks([])
        self.map_ax.axis('off')
        
        # Add hover and click event handlers
        self.map_canvas.mpl_connect("motion_notify_event", self._on_hover)
        self.map_canvas.mpl_connect("button_press_event", self._on_click)
        
        self.map_canvas.draw()
    
    def _zoom_in(self):
        """Zoom in by increasing zoom level."""
        self.zoom_level *= 1.2
        self._apply_zoom()
        self.map_canvas.draw()
    
    def _zoom_out(self):
        """Zoom out by decreasing zoom level."""
        self.zoom_level /= 1.2
        self._apply_zoom()
        self.map_canvas.draw()
    
    def _reset_zoom(self):
        """Reset zoom to original level."""
        self.zoom_level = 1.0
        self.pan_offset_x = 0.0
        self.pan_offset_y = 0.0
        self._apply_zoom()
        self.map_canvas.draw()
    
    def _apply_zoom(self):
        """Apply current zoom level and pan offset to the map bounds."""
        if not hasattr(self, 'zoom_level'):
            return
        
        # Calculate zoomed bounds
        x_range = self.base_xlim[1] - self.base_xlim[0]
        y_range = self.base_ylim[1] - self.base_ylim[0]
        
        x_center = (self.base_xlim[0] + self.base_xlim[1]) / 2 + self.pan_offset_x
        y_center = (self.base_ylim[0] + self.base_ylim[1]) / 2 + self.pan_offset_y
        
        new_x_range = x_range / self.zoom_level
        new_y_range = y_range / self.zoom_level
        
        new_xlim = (x_center - new_x_range/2, x_center + new_x_range/2)
        new_ylim = (y_center - new_y_range/2, y_center + new_y_range/2)
        
        self.map_ax.set_xlim(new_xlim)
        self.map_ax.set_ylim(new_ylim)
    
    def _on_mouse_wheel(self, event):
        """Handle mouse wheel zoom events."""
        # Determine zoom direction
        if event.delta > 0 or event.num == 4:  # Scroll up or Linux scroll up
            self._zoom_in()
        elif event.delta < 0 or event.num == 5:  # Scroll down or Linux scroll down
            self._zoom_out()
    
    def _on_mouse_press(self, event):
        """Handle mouse button press for panning."""
        self.pan_start_x = event.x
        self.pan_start_y = event.y
        self.is_panning = True
    
    def _on_mouse_drag(self, event):
        """Handle mouse drag for panning."""
        if not self.is_panning:
            return
        
        # Calculate pan distance in data coordinates
        dx = event.x - self.pan_start_x
        dy = event.y - self.pan_start_y
        
        # Convert screen coordinates to data coordinates
        x_range = self.base_xlim[1] - self.base_xlim[0]
        y_range = self.base_ylim[1] - self.base_ylim[0]
        
        # Get current canvas size
        canvas_width = self.map_canvas.get_width_height()[0]
        canvas_height = self.map_canvas.get_width_height()[1]
        
        # Calculate pan offset in data coordinates
        pan_dx = -(dx / canvas_width) * (x_range / self.zoom_level)
        pan_dy = (dy / canvas_height) * (y_range / self.zoom_level)  # Flip Y axis
        
        # Update pan offset
        self.pan_offset_x += pan_dx
        self.pan_offset_y += pan_dy
        
        # Apply the pan
        self._apply_zoom()
        self.map_canvas.draw()
        
        # Update start position for next drag
        self.pan_start_x = event.x
        self.pan_start_y = event.y
    
    def _on_mouse_release(self, event):
        """Handle mouse button release."""
        self.is_panning = False
    
    def _on_hover(self, event):
        """Handle mouse hover over text labels and links."""
        if not hasattr(self, 'text_objects') or not self.text_objects:
            return
        
        # First, reset all text labels to normal
        for genre, text_obj in self.text_objects.items():
            # Skip if text object has no valid figure
            if text_obj.figure is None:
                continue
            if text_obj.get_fontweight() == 'bold':
                text_obj.set_fontweight('normal')
        
        # Reset all lines to normal
        if hasattr(self, 'line_objects') and self.line_objects:
            for (genre_a, genre_b), line_obj in self.line_objects.items():
                # Skip if line object has no valid figure
                if line_obj.figure is None:
                    continue
                if line_obj.get_color() == 'white':  # Reset if it was highlighted
                    if hasattr(line_obj, '_original_width'):
                        line_obj.set_linewidth(line_obj._original_width)
                    line_obj.set_color('gray')
        
        # Check if mouse is over any text label (priority over lines)
        text_hovered = False
        for genre, text_obj in self.text_objects.items():
            # Skip if text object has no valid figure
            if text_obj.figure is None:
                continue
            # Check if mouse is within text bounds (approximate)
            contains, _ = text_obj.contains(event)
            if contains:
                text_obj.set_fontweight('bold')
                text_hovered = True
                break  # Only highlight one text label at a time
        
        # Only check lines if no text is being hovered
        if not text_hovered and hasattr(self, 'line_objects') and self.line_objects:
            for (genre_a, genre_b), line_obj in self.line_objects.items():
                # Skip if line object has no valid figure
                if line_obj.figure is None:
                    continue
                # Check if mouse is within line bounds (approximate)
                contains, _ = line_obj.contains(event)
                if contains:
                    # Store original width for restoration
                    if not hasattr(line_obj, '_original_width'):
                        line_obj._original_width = line_obj.get_linewidth()
                    line_obj.set_linewidth(line_obj._original_width * 2)
                    line_obj.set_color('white')
                    break  # Only highlight one line at a time
        
        # Always redraw to ensure proper reset of all elements
        self.map_canvas.draw()
    
    def _on_click(self, event):
        """Handle mouse click on text labels and links."""
        if not hasattr(self, 'text_objects') or not self.text_objects or not self.result:
            return
        
        # Check if mouse is over any text label (priority over lines)
        for genre, text_obj in self.text_objects.items():
            # Skip if text object has no valid figure
            if text_obj.figure is None:
                continue
            contains, _ = text_obj.contains(event)
            if contains:
                self._show_files_for_genre(genre)
                return
        
        # Check if mouse is over any line
        if hasattr(self, 'line_objects') and self.line_objects:
            for (genre_a, genre_b), line_obj in self.line_objects.items():
                # Skip if line object has no valid figure
                if line_obj.figure is None:
                    continue
                contains, _ = line_obj.contains(event)
                if contains:
                    self._show_files_for_genre_combination(genre_a, genre_b)
                    return
    
    def _show_files_for_genre(self, genre):
        """Show files for a single genre (similar to existing functionality)."""
        paths = self.result.genre_files.get(genre, [])
        self._create_files_popup(f"Files for {genre} ({len(paths)})", paths)
    
    def _show_files_for_genre_combination(self, genre_a, genre_b):
        """Show files that have both genres (combination)."""
        # Find files that have both genres
        files_a = set(self.result.genre_files.get(genre_a, []))
        files_b = set(self.result.genre_files.get(genre_b, []))
        combined_files = list(files_a.intersection(files_b))
        
        self._create_files_popup(f"Files with both '{genre_a}' and '{genre_b}' ({len(combined_files)})", combined_files)
    
    def _create_files_popup(self, title, paths):
        """Create a popup window showing file information in a sortable table."""
        win = tk.Toplevel(self)
        win.title(title)
        win.geometry("1200x600")
        win.configure(bg='#2b2b2b')

        top = ttk.Frame(win)
        top.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Search box
        search_frame = ttk.Frame(top)
        search_frame.pack(fill=tk.X, pady=(0,8))
        tk.Label(search_frame, text="Filter:", bg='#2b2b2b', fg='#ffffff').pack(side=tk.LEFT)
        query_var = tk.StringVar()
        ent = ttk.Entry(search_frame, textvariable=query_var)
        ent.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6,6))
        
        # Parse file information
        file_data = []
        for path in paths:
            try:
                # Extract metadata using the same method as the main parsing
                ext = os.path.splitext(path)[1].lower()
                artist = ""
                title = ""
                
                if ext == ".mp3":
                    audio = ID3(path)
                    if audio:
                        artist = str(audio.get("TPE1", [""])[0]) if audio.get("TPE1") else ""
                        title = str(audio.get("TIT2", [""])[0]) if audio.get("TIT2") else ""
                        album = str(audio.get("TALB", [""])[0]) if audio.get("TALB") else ""
                elif ext in {".m4a", ".mp4"}:
                    audio = EasyMP4(path)
                    if audio:
                        artist = audio.get("artist", [""])[0] if audio.get("artist") else ""
                        title = audio.get("title", [""])[0] if audio.get("title") else ""
                        album = audio.get("album", [""])[0] if audio.get("album") else ""
                elif ext == ".flac":
                    audio = FLAC(path)
                    if audio:
                        artist = audio.get("artist", [""])[0] if audio.get("artist") else ""
                        title = audio.get("title", [""])[0] if audio.get("title") else ""
                        album = audio.get("album", [""])[0] if audio.get("album") else ""
                else:
                    audio = MutagenFile(path, easy=True)
                    if audio:
                        artist = audio.get("artist", [""])[0] if audio.get("artist") else ""
                        title = audio.get("title", [""])[0] if audio.get("title") else ""
                        album = audio.get("album", [""])[0] if audio.get("album") else ""
                
                # Fallback to filename if no metadata
                if not title:
                    title = os.path.basename(path)
                if not artist:
                    artist = "Unknown Artist"
                if not album:
                    album = "Unknown Album"
                    
                file_data.append((artist, title, album, path))
            except Exception:
                # If metadata extraction fails, use filename
                filename = os.path.basename(path)
                file_data.append(("Unknown Artist", filename, "Unknown Album", path))
        
        # Sort by artist by default
        file_data.sort(key=lambda x: x[0].lower())
        
        # Create Treeview table
        table_frame = ttk.Frame(top)
        table_frame.pack(fill=tk.BOTH, expand=True)
        
        tree = ttk.Treeview(table_frame, columns=("artist", "title", "album", "path"), show="headings", height=20)
        tree.heading("artist", text="Artist", command=lambda: self._sort_table(tree, "artist", file_data, query_var))
        tree.heading("title", text="Title", command=lambda: self._sort_table(tree, "title", file_data, query_var))
        tree.heading("album", text="Album", command=lambda: self._sort_table(tree, "album", file_data, query_var))
        tree.heading("path", text="Path", command=lambda: self._sort_table(tree, "path", file_data, query_var))
        
        tree.column("artist", width=160, anchor="w")
        tree.column("title", width=220, anchor="w")
        tree.column("album", width=180, anchor="w")
        tree.column("path", width=400, anchor="w")
        
        # Scrollbars
        v_scroll = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=v_scroll.set)
        
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Store data for sorting and filtering
        tree.file_data = file_data
        tree.query_var = query_var
        
        def refresh_table():
            q = query_var.get().lower().strip()
            # Clear existing items
            for item in tree.get_children():
                tree.delete(item)
            # Add filtered items
            for artist, title, album, path in file_data:
                if not q or q in artist.lower() or q in title.lower() or q in album.lower() or q in path.lower():
                    tree.insert("", tk.END, values=(artist, title, album, path))
        
        refresh_table()
        ent.bind('<KeyRelease>', lambda e: refresh_table())
    
    def _sort_table(self, tree, column, file_data, query_var):
        """Sort the table by the specified column."""
        # Get current sort order
        current_items = [(tree.set(item, column), item) for item in tree.get_children()]
        is_reverse = current_items == sorted(current_items, key=lambda x: x[0].lower())
        
        # Sort the data
        if column == "artist":
            file_data.sort(key=lambda x: x[0].lower(), reverse=is_reverse)
        elif column == "title":
            file_data.sort(key=lambda x: x[1].lower(), reverse=is_reverse)
        elif column == "album":
            file_data.sort(key=lambda x: x[2].lower(), reverse=is_reverse)
        elif column == "path":
            file_data.sort(key=lambda x: x[3].lower(), reverse=is_reverse)
        
        # Refresh the table
        q = query_var.get().lower().strip()
        for item in tree.get_children():
            tree.delete(item)
        for artist, title, album, path in file_data:
            if not q or q in artist.lower() or q in title.lower() or q in album.lower() or q in path.lower():
                tree.insert("", tk.END, values=(artist, title, album, path))
    


if __name__ == "__main__":
    app = GenreExplorer()
    app.mainloop()
