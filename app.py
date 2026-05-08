from flask import Flask, render_template, request, jsonify, send_file
from faster_whisper import WhisperModel
import imageio_ffmpeg
import subprocess
import os
import tempfile
import uuid

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

model = None
model_size = "base"

def get_model():
    global model
    if model is None:
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
    return model

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'audio' not in request.files:
        return jsonify({'error': '오디오 파일이 없습니다'}), 400

    file = request.files['audio']
    if file.filename == '':
        return jsonify({'error': '파일이 선택되지 않았습니다'}), 400

    language = request.form.get('language', 'ko')
    selected_model = request.form.get('model_size', 'base')

    global model_size, model
    if selected_model != model_size:
        model_size = selected_model
        model = None

    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4()}_{file.filename}")
    file.save(temp_path)

    audio_path = temp_path
    is_video = file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm'))

    try:
        if is_video:
            print(f"비디오에서 오디오 추출 중: {file.filename}")
            audio_path = os.path.splitext(temp_path)[0] + "_audio.mp3"
            try:
                ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
                subprocess.run([ffmpeg_exe, '-i', temp_path, '-q:a', '0', '-map', 'a', audio_path, '-y'],
                              capture_output=True, check=True)
            except subprocess.CalledProcessError as e:
                return jsonify({'error': f'오디오 추출 실패: {e.stderr.decode()}'}), 500
            except Exception as e:
                return jsonify({'error': f'ffmpeg 실행 오류: {str(e)}'}), 500

        whisper_model = get_model()
        segments, info = whisper_model.transcribe(audio_path, language=language, beam_size=5, vad_filter=True)

        result = {
            'language': info.language,
            'language_probability': round(info.language_probability, 2),
            'segments': []
        }

        full_text = []
        for segment in segments:
            result['segments'].append({
                'start': round(segment.start, 2),
                'end': round(segment.end, 2),
                'text': segment.text.strip()
            })
            full_text.append(segment.text.strip())

        result['full_text'] = ' '.join(full_text)
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    finally:
        if is_video and os.path.exists(audio_path):
            os.remove(audio_path)
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
