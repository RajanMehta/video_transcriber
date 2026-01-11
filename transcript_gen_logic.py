"""
Final Transcript Generation Logic
=================================

This script generates a speaker-attributed transcript from ASR (Automatic Speech Recognition) 
and Diarization outputs. It employs a multi-stage process to ensure accurate and 
smooth speaker assignment:

1. Data Loading:
   - Loads 'transcription.json' (ASR words with accurate timestamps).
   - Loads 'diarization.json' (Speaker segments with start/end times).

2. Segment Filtering:
   - Removes diarization segments shorter than 0.3 seconds.
   - Purpose: Eliminates noisy, ultra-short speaker detections that often cause rapid 
     speaker flipping artifacts.

3. Initial Speaker Assignment (Word-Level):
   - Iterates through each word from the ASR output.
   - Assigns a speaker using 'Nearest Neighbor' logic:
     a. Exact Match: If the word's midpoint falls strictly within a speaker segment.
     b. Nearest Neighbor: If the word falls in a gap (silence/noise), assigns it to 
        the temporally closest speaker segment to prevent "[Unknown]" labels.

4. Speaker Smoothing:
   - Iterates through the list of assigned speakers.
   - Checks the gap between the current word and the previous word.
   - If the gap is less than 0.25 seconds, the current word is forced to match the 
     previous speaker.
   - Purpose: Prevents mid-sentence flipping (e.g., "Wow. [change] Okay") by enforcing 
     continuity during natural speech flow.

5. Transcript Construction:
   - Groups consecutive words spoken by the same speaker into sentences.
   - formats the output as "[SPEAKER_XX]: Text..."
   - Saves to 'final_transcript.txt'.
"""

import json

def load_data():
    with open("transcription.json", "r") as f:
        transcription = json.load(f)
    with open("diarization.json", "r") as f:
        diarization = json.load(f)
    return transcription, diarization

def filter_short_segments(segments, threshold=0.3):
    return [s for s in segments if (s['end'] - s['start']) >= threshold]

def get_speaker_improved(word_start, word_end, segments):
    """
    Finds the speaker for the word.
    1. Checks if the word's midpoint is within a segment.
    2. If not, finds the closest segment in time.
    """
    word_mid = (word_start + word_end) / 2
    
    # 1. Exact match (midpoint inside segment)
    for seg in segments:
        if seg['start'] <= word_mid <= seg['end']:
            return seg['speaker']
            
    # 2. Closest match
    closest_speaker = "Unknown"
    min_dist = float('inf')
    
    for seg in segments:
        # Calculate distance from word interval to segment interval
        # if word is before segment: dist = seg.start - word.end
        # if word is after segment: dist = word.start - seg.end
        
        dist = 0
        if word_end < seg['start']:
            dist = seg['start'] - word_end
        elif word_start > seg['end']:
            dist = word_start - seg['end']
        else:
            # Overlap case (should have been caught above by midpoint, but logic might vary)
            dist = 0
            
        if dist < min_dist:
            min_dist = dist
            closest_speaker = seg['speaker']
            
    return closest_speaker

def generate_transcript():
    transcription, speaker_segments = load_data()
    speaker_segments = filter_short_segments(speaker_segments)
    word_timestamps = transcription.get('timestamps', [])
    
    # Step 1: Assign initial speakers
    word_speakers = []
    for word_data in word_timestamps:
        w_start = word_data.get('start', word_data.get('start_offset', 0))
        w_end = word_data.get('end', word_data.get('end_offset', 0))
        speaker = get_speaker_improved(w_start, w_end, speaker_segments)
        word_speakers.append({
            "word": word_data['word'],
            "start": w_start,
            "end": w_end,
            "speaker": speaker
        })
        
    # Step 2: Smooth speakers (Gap < 0.1s => conform to previous)
    for i in range(1, len(word_speakers)):
        prev = word_speakers[i-1]
        curr = word_speakers[i]
        
        # Gap calculation
        gap = curr['start'] - prev['end']
        
        if gap < 0.25:
            curr['speaker'] = prev['speaker']
            
    # Step 3: Build final transcript
    final_transcript = []
    current_speaker = None
    current_sentence = []

    for item in word_speakers:
        speaker = item['speaker']
        word = item['word']
        
        if speaker != current_speaker:
            if current_speaker is not None:
                final_transcript.append({
                    "speaker": current_speaker,
                    "text": " ".join(current_sentence)
                })
            current_speaker = speaker
            current_sentence = [word]
        else:
            current_sentence.append(word)

    if current_sentence:
        final_transcript.append({
            "speaker": current_speaker,
            "text": " ".join(current_sentence)
        })

    # Save Final Transcript
    with open("final_transcript.txt", "w") as f:
        for line in final_transcript:
            f.write(f"[{line['speaker']}]: {line['text']}\n")
    print("\nSaved to final_transcript.txt")
        
if __name__ == "__main__":
    generate_transcript()
