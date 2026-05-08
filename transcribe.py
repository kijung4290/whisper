import sys
import os
from faster_whisper import WhisperModel

def transcribe_audio(audio_path, model_size="base", language="ko", device="cpu"):
    """
    음성 파일을 전사하는 함수

    Args:
        audio_path: 오디오 파일 경로
        model_size: 모델 크기 (tiny, base, small, medium, large-v3)
        language: 언어 코드 (ko, en, ja 등)
        device: cpu 또는 cuda
    """
    if not os.path.exists(audio_path):
        print(f"Error: 파일을 찾을 수 없습니다 - {audio_path}")
        return

    print(f"모델 로딩 중: {model_size}")
    model = WhisperModel(model_size, device=device, compute_type="int8")

    print(f"음성 파일 전사 중: {audio_path}")
    segments, info = model.transcribe(
        audio_path,
        language=language,
        beam_size=5,
        vad_filter=True
    )

    print(f"\n감지된 언어: {info.language} (확률: {info.language_probability:.2f})")
    print("\n" + "="*60)
    print("전사 결과")
    print("="*60 + "\n")

    full_text = []
    for segment in segments:
        text = segment.text.strip()
        print(f"[{segment.start:6.2f}s -> {segment.end:6.2f}s] {text}")
        full_text.append(text)

    # 전체 텍스트 저장
    output_file = os.path.splitext(audio_path)[0] + "_transcript.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(full_text))

    print(f"\n전사 결과가 저장되었습니다: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python transcribe.py <오디오파일> [모델크기] [언어]")
        print("\n모델 크기 옵션:")
        print("  tiny   - 가장 빠름, 낮은 정확도")
        print("  base   - 빠름, 보통 정확도 (기본값)")
        print("  small  - 보통 속도, 좋은 정확도")
        print("  medium - 느림, 높은 정확도")
        print("  large-v3 - 가장 느림, 최고 정확도")
        print("\n예시:")
        print("  python transcribe.py audio.mp3")
        print("  python transcribe.py audio.mp3 medium ko")
        sys.exit(1)

    audio_file = sys.argv[1]
    model_size = sys.argv[2] if len(sys.argv) > 2 else "base"
    language = sys.argv[3] if len(sys.argv) > 3 else "ko"

    transcribe_audio(audio_file, model_size, language)
