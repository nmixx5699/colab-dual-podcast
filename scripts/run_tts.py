import argparse
import json
from pathlib import Path
from src.tts import VoiceEngine
from src.dialogue import prepare_dialogue
from src.audio_timeline import build_timeline, save_timeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('settings')
    args = parser.parse_args()
    settings = json.loads(Path(args.settings).read_text(encoding='utf-8'))
    engine = VoiceEngine(Path(settings['cache']) / 'tts', str(Path(settings['cache']) / 'hf'))
    try:
        utterances = engine.synthesize(prepare_dialogue(settings['dialogue']), settings['voices'], settings['config']['seed'])
        tracks, mix, events = build_timeline(utterances, engine.sample_rate, settings['config']['pause_sec'])
        duration = len(mix) / engine.sample_rate
        save_timeline(Path(settings['run_dir']) / 'audio', tracks, mix, events, engine.sample_rate)
        print(f'Actual dialogue duration: {duration:.3f}s. Stage limit: {settings["config"]["max_duration_sec"]}s.')
        if duration > settings['config']['max_duration_sec']:
            print('Stage duration exceeded. Preview audio, then shorten dialogue or explicitly choose the next stage before video generation.')
    finally:
        engine.close()


if __name__ == '__main__':
    main()
