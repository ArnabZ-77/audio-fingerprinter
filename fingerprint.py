import numpy as np
import librosa
from scipy.ndimage import maximum_filter
import pickle
import os
import time

def load_audio(filepath, sr=22050):
    y, _ = librosa.load(filepath, sr=sr)
    return y, sr

def compute_spectrogram(y, sr, n_fft=2048, hop_length=512):
    stft_matrix = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    stft_db = librosa.amplitude_to_db(np.abs(stft_matrix), ref=np.max)
    return stft_db

def get_constellation_map(stft_db, threshold_db=-45, neighborhood_size=15):
    local_max = maximum_filter(stft_db, size=neighborhood_size) == stft_db
    peaks = local_max & (stft_db > threshold_db)
    freq_indices, time_indices = np.where(peaks)
    return list(zip(time_indices, freq_indices))

def generate_hashes(peaks, target_zone_dt=(1, 50), target_zone_df=(-30, 30)):
    hashes = {}
    peaks = sorted(peaks, key=lambda x: x[0])
    
    for i, (t1, f1) in enumerate(peaks):
        for j in range(i + 1, len(peaks)):
            t2, f2 = peaks[j]
            dt = t2 - t1
            df = f2 - f1
            
            if target_zone_dt[0] <= dt <= target_zone_dt[1] and target_zone_df[0] <= df <= target_zone_df[1]:
                hash_key = (f1, f2, dt)
                if hash_key not in hashes:
                    hashes[hash_key] = []
                hashes[hash_key].append(t1)
    return hashes

def build_database(song_library_dir, output_pkl="database_index.pkl"):
    """Saves structured map containing inverted hash lookups along with visualization metadata."""
    database = {
        "index": {},  # { hash_key: [(song_name, t1), ...] }
        "songs": {}   # { song_name: { "hash_count": X, "peaks": [...] } }
    }
    if not os.path.exists(song_library_dir):
        print(f"Directory '{song_library_dir}' not found.")
        return
        
    files = [f for f in os.listdir(song_library_dir) if f.endswith(('.mp3', '.wav'))]
    if not files:
        print(f"No audio files found in '{song_library_dir}' folder!")
        return

    for filename in files:
        song_name = os.path.splitext(filename)[0]
        filepath = os.path.join(song_library_dir, filename)
        print(f"Indexing track: {song_name}...")
        
        try:
            y, sr = load_audio(filepath)
            stft_db = compute_spectrogram(y, sr)
            peaks = get_constellation_map(stft_db)
            song_hashes = generate_hashes(peaks)
            
            total_hashes = 0
            for hash_key, t1_list in song_hashes.items():
                if hash_key not in database["index"]:
                    database["index"][hash_key] = []
                for t1 in t1_list:
                    database["index"][hash_key].append((song_name, t1))
                    total_hashes += 1
            
            # Keep downsampled sample peaks list to display library thumbnail maps instantly
            database["songs"][song_name] = {
                "hash_count": total_hashes,
                "peaks": peaks[:1200]  # Cap storage limit size to keep binary lightweight
            }
        except Exception as e:
            print(f"Error parsing {filename}: {e}")
                    
    with open(output_pkl, "wb") as f:
        pickle.dump(database, f)
    print(f"\nSuccess! Saved database index file to '{output_pkl}' containing {len(database['songs'])} songs.")

def match_clip(query_filepath, database):
    """Executes full diagnostic pipeline while logging timeline profiles matching the demo specs."""
    metrics = {}
    
    # 1. Spectrogram
    t_start = time.time()
    y, sr = load_audio(query_filepath)
    stft_db = compute_spectrogram(y, sr)
    metrics["spectrogram_time"] = int((time.time() - t_start) * 1000)
    metrics["spectrogram_shape"] = f"{stft_db.shape[0]}x{stft_db.shape[1]}"
    
    # 2. Constellation
    t_start = time.time()
    peaks = get_constellation_map(stft_db)
    metrics["constellation_time"] = int((time.time() - t_start) * 1000)
    metrics["peak_count"] = len(peaks)
    
    # 3. Hashes
    t_start = time.time()
    query_hashes = generate_hashes(peaks)
    total_q_hashes = sum(len(v) for v in query_hashes.values())
    metrics["hashes_time"] = int((time.time() - t_start) * 1000)
    metrics["hash_count"] = total_q_hashes
    
    # 4. DB Lookup
    t_start = time.time()
    matches = {}
    for hash_key, q_t1_list in query_hashes.items():
        if hash_key in database["index"]:
            for song_name, db_t1 in database["index"][hash_key]:
                for q_t1 in q_t1_list:
                    offset = db_t1 - q_t1
                    if song_name not in matches:
                        matches[song_name] = []
                    matches[song_name].append(offset)
    metrics["db_time"] = int((time.time() - t_start) * 1000)
    metrics["tracks_checked"] = len(database["songs"])
    
    # 5. Scoring
    t_start = time.time()
    best_song = "Unknown / No Match"
    highest_vote_count = 0
    best_offsets_list = []
    scores_chart = {}
    
    for song_name, offsets in matches.items():
        if len(offsets) == 0:
            continue
        counts, bins = np.histogram(offsets, bins=np.arange(min(offsets)-1, max(offsets)+2))
        max_votes = np.max(counts)
        scores_chart[song_name] = int(max_votes)
        
        if max_votes > highest_vote_count:
            highest_vote_count = max_votes
            best_song = song_name
            best_offsets_list = offsets
            
    metrics["scoring_time"] = int((time.time() - t_start) * 1000)
    metrics["best_offset"] = int(np.median(best_offsets_list)) if best_offsets_list else 0
    
    # Sort candidate chart descending
    sorted_scores = sorted(scores_chart.items(), key=lambda x: x[1], reverse=True)
    
    return best_song, highest_vote_count, sorted_scores, best_offsets_list, stft_db, peaks, metrics