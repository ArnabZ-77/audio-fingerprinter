#!/usr/bin/env python3
"""
Create sample query clips for testing by extracting sections from library songs
"""
import os
import librosa
import soundfile as sf
import numpy as np

def create_sample_queries():
    """Extract 5-second clips from different songs for testing"""
    os.makedirs("query_clips", exist_ok=True)
    
    song_dir = "song_library"
    songs = sorted([f for f in os.listdir(song_dir) if f.endswith(('.mp3', '.wav'))])[:5]
    
    if not songs:
        print("No songs found in song_library")
        return
    
    for idx, song_file in enumerate(songs):
        try:
            filepath = os.path.join(song_dir, song_file)
            y, sr = librosa.load(filepath, sr=22050)
            
            # Extract a 5-second clip starting at 10 seconds
            start_sample = int(10 * sr)
            end_sample = int(15 * sr)
            
            clip = y[start_sample:end_sample]
            
            # Save as query clip
            query_name = f"query_{idx+1}_{os.path.splitext(song_file)[0]}.wav"
            query_path = os.path.join("query_clips", query_name)
            sf.write(query_path, clip, sr)
            
            print(f"✓ Created: {query_name}")
        except Exception as e:
            print(f"✗ Error with {song_file}: {e}")

if __name__ == "__main__":
    create_sample_queries()
    print("\nSample query clips created in query_clips/ folder")
