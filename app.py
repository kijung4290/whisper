from flask import Flask, render_template, request, jsonify
from faster_whisper import WhisperModel
import imageio_ffmpeg
import subprocess
import os
import tempfile
import uuid
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

model = None
model_size = "tiny"
model_ready = False

ALLOWED_LANGUAGES = {'ko', 'en', 'ja', 'zh', 'de', 'fr', 'es', 'ru'}

def get_model():
    global model, model_ready
    if model is None:
        logger.info(f"모델 로딩 중: {model_size}")
        try:
            model = WhisperModel(model_size, device="cpu", compute_type="int8")
            model_ready = True
            logger.info("모델 로딩 완료")
        except Exception as e:
            logger.error(f"모델 로딩 실패: {e}")
            model_ready = False
            raise
    return model

try:
    get_model()
except Exception as e:
    logger.error(f"앱 시작 중 모델 로딩 실패: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'model_ready': model_ready, 'model_size': model_size})

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'audio' not in request.files:
        return jsonify({'error': '오디오 파일이 없습니다'}), 400

    file = request.files['audio']
    if file.filename == '':
        return jsonify({'error': '파일이 선택되지 않았습니다'}), 400

    language = request.form.get('language', 'ko')
    if language == 'auto':
        language = None

    selected_model = request.form.get('model_size', 'tiny')

    global model_size, model
    if selected_model != model_size:
        model_size = selected_model
        model = None

    temp_filename = f"{uuid.uuid4()}_{uuid.uuid4().hex[:8]}"
    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
    file.save(temp_path)

    audio_path = temp_path
    orig_ext = os.path.splitext(file.filename)[1].lower()
    is_video = orig_ext in ('.mp4', '.avi', '.mov', '.mkv', '.webm')

    try:
        if not model_ready:
            logger.error("모델이 준비되지 않음")
            return jsonify({'error': '모델이 아직 로딩되지 않았습니다. 잠시 후 다시 시도해주세요.'}), 503

        if is_video:
            logger.info(f"비디오에서 오디오 추출 중")
            audio_path = temp_path + "_audio.mp3"
            try:
                ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
                result = subprocess.run([ffmpeg_exe, '-i', temp_path, '-q:a', '0', '-map', 'a', audio_path, '-y'],
                              capture_output=True, check=True)
            except subprocess.CalledProcessError as e:
                err = e.stderr.decode('utf-8', errors='replace')
                logger.error(f"오디오 추출 실패: {err}")
                return jsonify({'error': f'오디오 추출 실패'}), 500
            except Exception as e:
                logger.error(f"ffmpeg 실행 오류: {e}")
                return jsonify({'error': f'ffmpeg 실행 오류: {str(e)}'}), 500

        if not os.path.exists(audio_path):
            return jsonify({'error': '오디오 파일을 찾을 수 없습니다'}), 400

        whisper_model = get_model()
        transcribe_kwargs = {}
        if language:
            transcribe_kwargs['language'] = language
        logger.info(f"전사 시작: language={language}, model={model_size}")
        segments, info = whisper_model.transcribe(audio_path, beam_size=5, vad_filter=True, **transcribe_kwargs)

        result = {
            'language': info.language,
            'language_probability': round(getattr(info, 'language_probability', 0), 2),
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
