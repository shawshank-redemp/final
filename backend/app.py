from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import os
import base64
import json
import numpy as np
import librosa
import subprocess
import anthropic

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = '../uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

video_database = []

# ── Initialize Anthropic client (reads ANTHROPIC_API_KEY from env) ──────────
claude_client = anthropic.Anthropic()


# ── Feature extraction ───────────────────────────────────────────────────────

def extract_keyframes(video_path, num_frames=6):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in range(num_frames):
        frame_idx = int((i / num_frames) * total_frames)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (640, 360))
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            b64 = base64.b64encode(buffer).decode('utf-8')
            frames.append(b64)
    cap.release()
    return frames


def extract_motion_profile(video_path, max_frames=60):
    cap = cv2.VideoCapture(video_path)
    ret, prev = cap.read()
    if not ret:
        return None
    prev_gray = cv2.cvtColor(cv2.resize(prev, (320, 180)), cv2.COLOR_BGR2GRAY)
    magnitudes = []
    count = 0
    while count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(cv2.resize(frame, (320, 180)), cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        magnitudes.append(float(np.mean(mag)))
        prev_gray = gray
        count += 1
    cap.release()
    if not magnitudes:
        return None
    return {
        'profile': magnitudes,
        'mean': float(np.mean(magnitudes)),
        'std': float(np.std(magnitudes)),
        'max': float(np.max(magnitudes))
    }


def extract_audio_profile(video_path):
    try:
        audio_path = video_path + '_audio.wav'
        subprocess.run(
            f'ffmpeg -i "{video_path}" -ac 1 -ar 22050 "{audio_path}" -y -loglevel quiet',
            shell=True
        )
        if not os.path.exists(audio_path):
            return None
        y, sr = librosa.load(audio_path, duration=30)
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20), axis=1)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
        tempo_arr, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo = float(np.asarray(tempo_arr).flatten()[0])
        os.remove(audio_path)
        return {
            'mfcc': mfcc.tolist(),
            'chroma': chroma.tolist(),
            'tempo': tempo
        }
    except Exception as e:
        print(f"Audio error: {e}")
        return None


# ── Similarity computation ───────────────────────────────────────────────────

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def compare_videos(new_motion, new_audio, existing):
    motion_sim = 0.0
    audio_sim = 0.0

    if new_motion and existing.get('motion'):
        p1 = new_motion['profile']
        p2 = existing['motion']['profile']
        min_len = min(len(p1), len(p2))
        if min_len > 0:
            motion_sim = cosine_similarity(p1[:min_len], p2[:min_len])
            motion_sim = max(0.0, motion_sim)

    if new_audio and existing.get('audio'):
        mfcc_sim = cosine_similarity(new_audio['mfcc'], existing['audio']['mfcc'])
        chroma_sim = cosine_similarity(new_audio['chroma'], existing['audio']['chroma'])
        t1 = new_audio['tempo']
        t2 = existing['audio']['tempo']
        tempo_sim = 1 - abs(t1 - t2) / max(t1, t2, 1)
        audio_sim = max(0.0, mfcc_sim * 0.5 + chroma_sim * 0.3 + tempo_sim * 0.2)

    if new_motion and new_audio:
        overall = (motion_sim * 0.5 + audio_sim * 0.5) * 100
    elif new_motion:
        overall = motion_sim * 100
    elif new_audio:
        overall = audio_sim * 100
    else:
        overall = 0.0

    return {
        'filename': existing['filename'],
        'motion_similarity': round(motion_sim * 100, 1),
        'audio_similarity': round(audio_sim * 100, 1),
        'overall_similarity': round(overall, 1)
    }


def compute_final_score(comparisons):
    if not comparisons:
        return 100, "none", []

    most_similar = max(comparisons, key=lambda c: c['overall_similarity'])
    most = most_similar

    both_high = most['audio_similarity'] > 92 and most['motion_similarity'] > 92
    audio_high = most['audio_similarity'] > 92
    motion_high = most['motion_similarity'] > 95

    if both_high:
        score = int(10 + (100 - most['overall_similarity']) * 0.3)
    elif audio_high and not motion_high:
        score = int(40 + (100 - most['audio_similarity']) * 0.8)
    elif motion_high and not audio_high:
        score = int(45 + (100 - most['motion_similarity']) * 0.8)
    else:
        score = 100

    score = max(0, min(100, score))
    return score, most_similar['filename'], comparisons


# ── Claude Agentic Reasoning (replaces LLaVA) ───────────────────────────────

def ask_claude_agent(filename, comparisons, score, frames):
    """
    Claude acts as a synthetic content detection agent.
    It receives computed similarity scores + keyframes of the new video,
    then reasons about them to produce:
      - A short visual description (from frames)
      - A verdict + reason grounded in the actual scores
    """
    top = comparisons[0] if comparisons else None

    # Build the score context string
    if top:
        score_context = f"""
Similarity analysis results (computed via optical flow + audio fingerprinting):
- Closest matching video in database: '{top['filename']}'
- Motion similarity: {top['motion_similarity']}%
- Audio similarity:  {top['audio_similarity']}%
- Overall similarity: {top['overall_similarity']}%
- Authenticity score assigned: {score}/100
"""
    else:
        score_context = "No existing videos to compare against. This is the first entry."

    # Build message content — text + up to 3 keyframes from the new video
    content = []

    # Add keyframes as images so Claude can visually describe the video
    for b64_frame in frames[:3]:
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": b64_frame
            }
        })

    content.append({
        "type": "text",
        "text": f"""You are a synthetic content detection agent analyzing a short-form video called '{filename}'.

{score_context}

Your tasks:
1. Look at the keyframes above and write a 1-sentence visual description of what is happening in the video (person, action, setting).
2. Based ONLY on the similarity scores above, write a 1-sentence verdict explaining why this video received a score of {score}/100.

Rules:
- Do NOT say "no visuals provided" — you have the frames above.
- Do NOT hallucinate. Base the verdict strictly on the numbers.
- Keep both sentences short and factual.

Respond in this exact JSON format:
{{
  "description": "<1 sentence: what is visually happening in the video>",
  "verdict": "<1 sentence: why this score was given based on motion/audio similarity>"
}}"""
    })

    try:
        message = claude_client.messages.create(
            model="claude-opus-4-6",
            max_tokens=300,
            messages=[{"role": "user", "content": content}]
        )
        raw = message.content[0].text.strip()
        print(f"Claude agent response: {raw}")

        # Parse JSON from response
        start = raw.find('{')
        end = raw.rfind('}') + 1
        if start != -1 and end > start:
            parsed = json.loads(raw[start:end])
            return {
                "description": parsed.get("description", "Visual description unavailable."),
                "verdict": parsed.get("verdict", "Verdict unavailable.")
            }
    except Exception as e:
        print(f"Claude agent error: {e}")

    # Fallback to rule-based if Claude fails
    return {
        "description": "Visual description unavailable.",
        "verdict": _fallback_verdict(comparisons, score)
    }


def _fallback_verdict(comparisons, score):
    """Rule-based fallback if Claude API call fails."""
    if not comparisons:
        return "First video in database — authenticity score is 100."
    top = comparisons[0]
    if top["audio_similarity"] > 92 and top["motion_similarity"] > 92:
        return f"Both audio ({top['audio_similarity']}%) and motion ({top['motion_similarity']}%) are highly similar to '{top['filename']}' — flagged as repetitive synthetic content."
    elif top["audio_similarity"] > 92:
        return f"Audio fingerprint is {top['audio_similarity']}% similar to '{top['filename']}' — likely reuses the same soundtrack."
    elif top["motion_similarity"] > 95:
        return f"Motion pattern is {top['motion_similarity']}% similar to '{top['filename']}' — repeated movement detected."
    else:
        return f"Low similarity (motion: {top['motion_similarity']}%, audio: {top['audio_similarity']}%) — content appears original."


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video uploaded'}), 400

    video = request.files['video']
    filename = video.filename
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    video.save(filepath)

    print(f"\nProcessing: {filename}")

    frames = extract_keyframes(filepath)
    motion = extract_motion_profile(filepath)
    audio = extract_audio_profile(filepath)

    # ── First video — no comparison possible ────────────────────────────────
    if not video_database:
        print("First video — score: 100")

        # Still ask Claude to visually describe it
        agent_result = ask_claude_agent(filename, [], 100, frames)

        entry = {
            'filename': filename,
            'score': 100,
            'description': agent_result['description'],
            'reasoning': 'First video in the database — no existing content to compare against. Authenticity score is 100.',
            'similarity_to': 'none',
            'comparisons': [],
            'motion': motion,
            'audio': audio,
            'frames': frames
        }
        video_database.append(entry)
        return jsonify({
            'filename': filename,
            'score': 100,
            'description': entry['description'],
            'reasoning': entry['reasoning'],
            'similarity_to': 'none',
            'comparisons': []
        })

    # ── Subsequent videos — compare and reason ───────────────────────────────
    comparisons = [compare_videos(motion, audio, v) for v in video_database]
    comparisons.sort(key=lambda c: c['overall_similarity'], reverse=True)
    print(f"Comparisons: {comparisons}")

    score, similar_to, comparisons = compute_final_score(comparisons)
    print(f"Computed score: {score}")

    print("Asking Claude agent for verdict...")
    agent_result = ask_claude_agent(filename, comparisons, score, frames)

    # Claude provides both description and verdict; combine into reasoning
    reasoning = agent_result['verdict']
    description = agent_result['description']

    entry = {
        'filename': filename,
        'score': score,
        'description': description,
        'reasoning': reasoning,
        'similarity_to': similar_to,
        'comparisons': comparisons,
        'motion': motion,
        'audio': audio,
        'frames': frames
    }
    video_database.append(entry)

    return jsonify({
        'filename': filename,
        'score': score,
        'description': description,
        'reasoning': reasoning,
        'similarity_to': similar_to,
        'comparisons': comparisons
    })


@app.route('/videos', methods=['GET'])
def get_videos():
    return jsonify([{
        'filename': v['filename'],
        'score': v['score'],
        'description': v['description']
    } for v in video_database])


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'running', 'videos_in_db': len(video_database)})


if __name__ == '__main__':
    app.run(debug=True, port=5000)