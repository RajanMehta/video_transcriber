
import os
import json
import torch
import nemo.collections.asr as nemo_asr
from pyannote.audio import Pipeline
from pydub import AudioSegment
from huggingface_hub import login

HF_TOKEN = "hf_..." 

# Login to Hugging Face
login(token=HF_TOKEN)

# Check GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


video_filename = "video.mp4" 
audio_filename = "./artifacts/extracted_audio.wav"

if os.path.exists(video_filename):
    # Extract audio using pydub/ffmpeg
    print(f"Extracting audio from {video_filename}...")
    audio = AudioSegment.from_file(video_filename)
    # Convert to mono 16kHz wav (ideal for ASR/Diarization)
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio.export(audio_filename, format="wav")
    print(f"Audio saved to {audio_filename}")
else:
    print(f"Error: {video_filename} not found. Please upload it to the Files tab.")


print("Loading ASR Model...")
asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v3")
asr_model = asr_model.to(device)

print("Transcribing...")
# Transcribe with timestamps=True to get word-level timing
hypotheses = asr_model.transcribe([audio_filename], timestamps=True)

# Extract text and timestamps
transcript_text = hypotheses[0].text
timestamps = hypotheses[0].timestamp

word_timestamps = timestamps.get('word', [])
print(f"Transcription Complete. Found {len(word_timestamps)} words.")
if word_timestamps:
    print(f"Sample: {word_timestamps[:3]}")


# Save Transcription Output
transcription_output = {
    "text": transcript_text,
    "timestamps": word_timestamps
}
with open("./artifacts/transcription.json", "w") as f:
    json.dump(transcription_output, f, indent=4)
print("Transcription output saved to transcription.json")


print("Loading Diarization Pipeline...")
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1"
).to(device)

print("Diarizing (this may take a moment)...")
diarization_result = pipeline(audio_filename)

# Store segments in a list for easy processing
# Format: {'start': float, 'end': float, 'speaker': str}
speaker_segments = []
for turn, _, speaker in diarization_result.itertracks(yield_label=True):
    if turn.end - turn.start < 0.3:
         continue
    speaker_segments.append({
        "start": turn.start,
        "end": turn.end,
        "speaker": speaker
    })

# Save Diarization Output
with open("./artifacts/diarization.json", "w") as f:
    json.dump(speaker_segments, f, indent=4)
print("Diarization output saved to diarization.json")
print(f"Diarization Complete. Found {len(speaker_segments)} segments.")


def get_speaker_for_word(word_start, word_end, segments):
    """Finds the speaker active during the word's timeframe."""
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
        dist = 0
        if word_end < seg['start']:
            dist = seg['start'] - word_end
        elif word_start > seg['end']:
            dist = word_start - seg['end']
        
        if dist < min_dist:
            min_dist = dist
            closest_speaker = seg['speaker']
            
    return closest_speaker

final_transcript = []
current_speaker = None
current_sentence = []

# Loop through all words from ASR
word_speakers = []
for word_data in word_timestamps:
    word = word_data['word']
    w_start = word_data['start']
    w_end = word_data['end']
    
    # Identify speaker
    speaker = get_speaker_for_word(w_start, w_end, speaker_segments)
    word_speakers.append({
        "word": word,
        "start": w_start,
        "end": w_end,
        "speaker": speaker
    })

# Smooth speakers (Gap < 0.25s => conform to previous)
for i in range(1, len(word_speakers)):
    prev = word_speakers[i-1]
    curr = word_speakers[i]
    
    # Gap calculation
    gap = curr['start'] - prev['end']
    
    if gap < 0.25:
        curr['speaker'] = prev['speaker']

# Build Final Transcript
for item in word_speakers:
    speaker = item['speaker']
    word = item['word']
    
    # If speaker changes, push the previous sentence and start a new one
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

# Append the last sentence
if current_sentence:
    final_transcript.append({
        "speaker": current_speaker,
        "text": " ".join(current_sentence)
    })


print("\n=== FINAL TRANSCRIPT ===\n")
for line in final_transcript:
    print(f"[{line['speaker']}]: {line['text']}")

# Save Final Transcript
with open("./artifacts/final_transcript.txt", "w") as f:
    for line in final_transcript:
        f.write(f"[{line['speaker']}]: {line['text']}\n")
print("\nSaved to final_transcript.txt")
