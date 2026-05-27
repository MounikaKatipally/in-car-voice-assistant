from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / 'data' / 'metadata' / 'intent_annotations_200.csv'

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


INTENT_EXAMPLES = {
    'navigation': [
        'navigate to the nearest gas station',
        'take me home',
        'start navigation to work',
        'find the closest coffee shop',
        'directions to the airport',
        'route me to downtown',
        'show me the way to the grocery store',
        'navigate to the nearest hospital',
        'find parking near me',
        'take me to the closest charging station',
        'start route guidance',
        'show directions to my office',
        'navigate to the hotel',
        'find the nearest restaurant',
        'take me to the mall',
        'route to the train station',
        'directions to the nearest pharmacy',
        'navigate to my saved home address',
        'find a gas station on this route',
        'take me to the nearest bank'
    ],
    'play_music': [
        'play some music',
        'play my driving playlist',
        'start the next song',
        'play relaxing music',
        'resume music',
        'play my favorite songs',
        'start spotify',
        'play something upbeat',
        'play the previous track',
        'skip this song',
        'pause the music',
        'turn up the music',
        'play rock music',
        'play pop songs',
        'shuffle my playlist',
        'play songs by taylor swift',
        'start apple music',
        'play workout music',
        'play the latest album',
        'continue the song'
    ],
    'call_contact': [
        'call mom',
        'call dad',
        'call john',
        'dial my manager',
        'call my office',
        'phone sarah',
        'call my brother',
        'call my sister',
        'dial the last number',
        'call emergency contact',
        'call home',
        'call alex on mobile',
        'make a call to david',
        'call my wife',
        'call my husband',
        'dial customer service',
        'call the doctor',
        'call my friend priya',
        'call my boss',
        'return the missed call'
    ],
    'weather': [
        'what is the weather today',
        'will it rain today',
        'show me the weather forecast',
        'what is the temperature outside',
        'is it cold outside',
        'do I need an umbrella',
        'weather for this evening',
        'how hot is it today',
        'is there snow in the forecast',
        'check the weather near me',
        'what is tomorrow weather',
        'tell me the weekly forecast',
        'is it windy outside',
        'what is the humidity',
        'will there be storms today',
        'weather at my destination',
        'current weather please',
        'show me weather updates',
        'is it sunny today',
        'forecast for my route'
    ],
    'climate_control': [
        'turn on the air conditioning',
        'increase the temperature',
        'decrease the temperature',
        'set temperature to seventy two',
        'turn on the heater',
        'turn off the AC',
        'make it cooler',
        'make it warmer',
        'turn on seat heating',
        'turn on defrost',
        'set fan speed to high',
        'lower the fan speed',
        'turn on climate control',
        'sync the cabin temperature',
        'turn on rear defrost',
        'set driver temperature to seventy',
        'set passenger temperature to seventy two',
        'turn on air circulation',
        'cool down the car',
        'warm up the cabin'
    ],
    'radio': [
        'turn on the radio',
        'play fm radio',
        'switch to am radio',
        'change the radio station',
        'tune to ninety eight point seven',
        'scan radio stations',
        'save this radio station',
        'play the next radio station',
        'go back to the previous station',
        'increase radio volume',
        'mute the radio',
        'turn off the radio',
        'open radio',
        'play news radio',
        'play sports radio',
        'tune to my favorite station',
        'switch radio band',
        'find local radio stations',
        'play classical radio',
        'resume radio'
    ],
    'settings': [
        'open settings',
        'change display brightness',
        'turn on bluetooth',
        'turn off bluetooth',
        'connect my phone',
        'open vehicle settings',
        'change language settings',
        'adjust screen brightness',
        'turn on dark mode',
        'reset system settings',
        'open sound settings',
        'pair a new device',
        'show system information',
        'update preferences',
        'open driver profile settings',
        'change units to miles',
        'change units to kilometers',
        'turn off notifications',
        'open privacy settings',
        'adjust voice assistant settings'
    ],
    'traffic': [
        'show traffic ahead',
        'is there traffic on my route',
        'check traffic conditions',
        'avoid traffic',
        'find a faster route',
        'show road delays',
        'traffic update please',
        'are there accidents nearby',
        'show congestion on the highway',
        'reroute around traffic',
        'how long is the delay',
        'traffic near downtown',
        'check commute traffic',
        'show traffic map',
        'avoid tolls and traffic',
        'is the highway clear',
        'find alternate route',
        'show current road conditions',
        'any traffic alerts',
        'traffic status to work'
    ],
    'cancel': [
        'cancel',
        'stop',
        'never mind',
        'cancel that',
        'stop navigation',
        'cancel the request',
        'forget it',
        'stop the assistant',
        'dismiss',
        'exit',
        'go back',
        'stop current task',
        'cancel route',
        'do not do that',
        'ignore that',
        'close this',
        'stop listening',
        'cancel command',
        'end this',
        'clear request'
    ],
    'confirm': [
        'yes',
        'confirm',
        'that is correct',
        'go ahead',
        'do it',
        'yes please',
        'confirm that',
        'sounds good',
        'proceed',
        'okay',
        'sure',
        'accept',
        'continue',
        'that works',
        'correct',
        'approve',
        'start it',
        'yes continue',
        'please continue',
        'confirm the selection'
    ],
}


def main():
    records = []

    for intent_label, examples in INTENT_EXAMPLES.items():
        for command_text in examples:
            records.append({
                'command_text': command_text,
                'intent_label': intent_label,
                'source': 'synthetic_in_car_commands',
                'annotation_method': 'manual_rule_based_seed',
            })

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f'Created {len(df)} intent annotations.')
    print(f'Output file: {OUTPUT_PATH}')
    print()
    print(df.head())


if __name__ == '__main__':
    main()
