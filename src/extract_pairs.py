#!/usr/bin/env python3
"""
TRS Conversation Extractor for TCOF Corpus
Simplified version for BARTHEZ + QWEN training data extraction
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path
import re
import argparse
import json
from typing import List, Dict, Tuple, Optional


class TRSExtractor:
    def __init__(self):
        # Define noise patterns to remove based on TCOF conventions
        self.noise_patterns = {
            'events': ['shh', 'mic', 'pap', 'rire', 'b', 'grincement de porte'],
            'pause_markers': ['+', '///', '###', '$$$'],
            'pronunciation_markers': ['[pron=', '[lang=', '{sic}', '{interprétation='],
            'incomprehensible': ['*', '***'],
            'liaison_markers': ['='],
            'hesitation_markers': ['/'],
            'word_fragments': ['-']
        }
        
        # Define minimal responses to filter out 
        self.minimal_responses = {
            'oui', 'ouais', 'non', 'ah', 'oh', 'hum', 'hum hum', 'ah oui', 'ah non',
            'ben oui', 'ben non', 'si', 'voilà', 'bon', 'bien', 'ok', 'd\'accord',
            'exactement', 'c\'est ça', 'mmh', 'mhm', 'euh', 'ben', 'alors', 'donc',
            'bah', 'hein', 'quoi', 'pardon', 'hm', 'ah bon', 'c\'est vrai',
            'mmh mmh', 'ah ah', 'oh oh', 'eh ben', 'eh bien', 'dis donc', 'bah oui',
            'ben dis donc', 'ah bah', 'ben non', 'ben si', 'wouah', 'woh', 'je sais', 'pas forcement',
            'je sais pas', 'aussi'
        }
        
        # Child-specific vocabulary to filter for L2 context
        self.child_specific_terms = {
            'maman', 'papa', 'mamie', 'papi', 'doudou', 'pipi', 'caca', 'bêtise',
            'poupée', 'jouet', 'récré', 'maîtresse'
        }
        
        # Useful question patterns for L2 learners
        self.l2_useful_patterns = {
            'qu\'est-ce que', 'comment', 'pourquoi', 'où', 'quand', 'qui', 
            'combien', 'quel', 'quelle', 'est-ce que', 'qu\'est-ce qui',
            'tu peux', 'tu sais', 'tu aimes', 'tu fais', 'tu as'
        }
        
        # Minimum word count for meaningful exchanges
        self.min_words_child = 2  # Lower threshold for child utterances
        self.min_words_adult = 4  # Adult should provide substantive response
    
    def clean_text(self, text: str) -> str:
        """
        Clean text according to TCOF conventions
        """
        if not text:
            return ""
        
        # Remove XML entities
        text = text.replace('&nbsp;', ' ')
        
        # Remove pause markers
        for marker in self.noise_patterns['pause_markers']:
            text = text.replace(marker, '')
        
        # Remove incomprehensible markers
        for marker in self.noise_patterns['incomprehensible']:
            text = text.replace(marker, '')
        
        # Remove liaison markers 
        text = re.sub(r'=([^=]+)=', r'\1', text)
        
        # Remove hesitation markers 
        text = re.sub(r'/([^/]+)/', '', text)
        
        # Remove word fragments 
        text = re.sub(r'\b\w+-(?=\s|$)', '', text)
        
        # Remove {} and other comment markers
        text = re.sub(r'\{[^}]+\}', '', text)
        
        # Remove pronunciation markers 
        text = re.sub(r'\[pron=[^\]]+\]', '', text)
        
        # Remove language markers 
        text = re.sub(r'\[lang=[^\]]+\]', '', text)
        
        # Remove other square bracket markers
        text = re.sub(r'\[[^\]]+\]', '', text)
        
        # Remove parentheses for morphological variants
        text = re.sub(r'\(([^)]*)\)', r'\1', text)
        
        # Clean up multiple spaces and trim
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'\b(\w+)\s+\1\b', r'\1', text)
        text = text.replace('<', '').replace('>', '')
        text = text.replace('/', '').replace('\\', '')


    
    # Remove filler words 
        fillers = {'euh', 'hum', 'ben', 'ah', 'mmh', 'mhm'}
        tokens = text.split()
        tokens = [t for t in tokens if t.lower() not in fillers]
        text = ' '.join(tokens)
        
        # Final cleanup
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_speaker_text(self, turn_element) -> Tuple[str, str]:
        """Extract speaker ID and clean text from a Turn element"""
        speaker_id = turn_element.get('speaker', '')
        
        # Handle overlapping speakers 
        if '+' in speaker_id:
            speaker_id = speaker_id.split('+')[0].strip()
        
        # Extract text content, excluding Event and Comment elements
        text_parts = []
        
        def extract_text_recursive(element):
            if element.tag in ['Event', 'Comment']:
                return
            
            if element.text:
                text_parts.append(element.text)
            
            for child in element:
                extract_text_recursive(child)
                if child.tail:
                    text_parts.append(child.tail)
        
        extract_text_recursive(turn_element)
        
        raw_text = ''.join(text_parts)
        cleaned_text = self.clean_text(raw_text)
        
        return speaker_id, cleaned_text
    
    def identify_speakers(self, speakers_element) -> Dict[str, str]:
        """Identify adult and child speakers from the Speakers section"""
        speaker_map = {}
        
        for speaker in speakers_element.findall('Speaker'):
            speaker_id = speaker.get('id', '')
            speaker_name = speaker.get('name', '')
            
            # Determine if speaker is adult or child based on conventions
            if speaker_name.lower() in ['adulte', 'adult']:
                speaker_map[speaker_id] = 'Adult'
            else:
                # Assume other speakers are children (named by first name)
                speaker_map[speaker_id] = 'Child'
        
        return speaker_map
    
    def extract_conversations(self, trs_file: Path) -> List[Dict]:
        """Extract conversation pairs from a single TRS file"""
        try:
            # Parse XML with proper encoding
            with open(trs_file, 'r', encoding='iso-8859-1') as f:
                content = f.read()
            
            # Parse XML
            root = ET.fromstring(content)
            
            # Get speaker information
            speakers_element = root.find('Speakers')
            if speakers_element is None:
                print(f"Warning: No Speakers section found in {trs_file}")
                return []
            
            speaker_map = self.identify_speakers(speakers_element)
            
            # Extract conversations
            conversations = []
            turns = root.findall('.//Turn')
            
            for turn in turns:
                speaker_id, text = self.extract_speaker_text(turn)
                
                if not text or not speaker_id:
                    continue
                
                # Map speaker ID to role
                speaker_role = speaker_map.get(speaker_id, 'Unknown')
                
                conversation_entry = {
                    'file': trs_file.name,
                    'speaker_id': speaker_id,
                    'speaker_role': speaker_role,
                    'text': text,
                    'start_time': turn.get('startTime', ''),
                    'end_time': turn.get('endTime', '')
                }
                
                conversations.append(conversation_entry)
            
            return conversations
            
        except ET.ParseError as e:
            print(f"Error parsing {trs_file}: {e}")
            return []
        except Exception as e:
            print(f"Error processing {trs_file}: {e}")
            return []
    
    def is_valid_pair(self, child_text: str, adult_text: str) -> bool:
        """
        Basic validation for conversation pairs
        """
        import string

        child_clean = child_text.strip().lower()
        adult_clean = adult_text.strip().lower()

        # Remove punctuation
        child_clean = child_clean.translate(str.maketrans('', '', string.punctuation))
        adult_clean = adult_clean.translate(str.maketrans('', '', string.punctuation))

        child_words = child_clean.split()
        adult_words = adult_clean.split()

        # Remove if too short
        if len(child_words) < self.min_words_child or len(adult_words) < self.min_words_adult:
            return False

        # Remove minimal responses
        if all(word in self.minimal_responses for word in adult_words):
            return False

        # remove 1-1 echos
        if child_clean == adult_clean:
            return False

        return True


    def is_valid_pair_for_l2(self, child_text, adult_text, filter_child_content=False):
        if not self.is_valid_pair(child_text, adult_text):
            return False

        child_clean = self.clean_text(child_text)
        adult_clean = self.clean_text(adult_text)

        child_lower = child_clean.lower().strip()

        # Normalize words by splitting and stripping
        child_words = re.split(r'\s+', child_lower)

        # Filter out if more than 50% of child words are minimal responses
        minimal_count = sum(1 for word in child_words if word in self.minimal_responses)
        if len(child_words) == 0 or (minimal_count / len(child_words)) > 0.5:
            return False

        # Additional L2 filters
        if filter_child_content:
            # Filter short utterances 
            if len(child_clean) < 5:
                return False
            if len(child_words) < 2:
                return False
            if any(word in self.minimal_responses for word in child_words) and len(set(child_words)) <= 2:
                return False
            if re.fullmatch(r"(euh|hum|mmh|ah|ben)+", child_lower):
                return False

        return True





    
    def create_conversation_pairs(self, conversations: List[Dict], l2_mode: bool = False) -> List[Dict]:
        """
        Create simple child-adult conversation pairs with optional L2 filtering
        """
        pairs = []
        
        for i in range(len(conversations) - 1):
            current = conversations[i]
            next_turn = conversations[i + 1]
            
            # Only process Child -> Adult pairs
            if (current['speaker_role'] == 'Child' and 
                next_turn['speaker_role'] == 'Adult'):
                
                child_text = current['text']
                adult_text = next_turn['text']
                
                # Use appropriate validation
                if l2_mode:
                    if self.is_valid_pair_for_l2(child_text, adult_text, filter_child_content=True):
                        pair = {
                            'child': child_text,
                            'adult': adult_text
                        }
                        pairs.append(pair)
                else:
                    if self.is_valid_pair(child_text, adult_text):
                        pair = {
                            'child': child_text,
                            'adult': adult_text
                        }
                        pairs.append(pair)
        
        return pairs
    
    def process_folder(self, input_folder: Path, output_file: Path = None, 
                      format: str = 'json', max_pairs: Optional[int] = None,
                      l2_mode: bool = False) -> Dict:
        """Process all TRS files in a folder"""
        if not input_folder.exists():
            raise FileNotFoundError(f"Input folder not found: {input_folder}")
        
        trs_files = list(input_folder.glob("*.trs"))
        
        if not trs_files:
            print(f"No TRS files found in {input_folder}")
            return {"files_processed": 0, "total_pairs": 0}
        
        all_pairs = []
        
        print(f"Processing {len(trs_files)} TRS files...")
        if l2_mode:
            print("L2 mode enabled: Filtering for adult language learners")
        
        for trs_file in trs_files:
            print(f"Processing: {trs_file.name}")
            
            conversations = self.extract_conversations(trs_file)
            pairs = self.create_conversation_pairs(conversations, l2_mode=l2_mode)
            
            all_pairs.extend(pairs)
            
            print(f"  - Extracted {len(pairs)} valid pairs")
            
            # Stop if we've reached max_pairs
            if max_pairs and len(all_pairs) >= max_pairs:
                all_pairs = all_pairs[:max_pairs]
                print(f"Reached maximum of {max_pairs} pairs")
                break
        
        # Save results
        if output_file:
            if format == 'json':
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(all_pairs, f, indent=2, ensure_ascii=False)
            
            elif format == 'jsonl':
                with open(output_file, 'w', encoding='utf-8') as f:
                    for pair in all_pairs:
                        f.write(json.dumps(pair, ensure_ascii=False) + '\n')
            
            elif format == 'txt':
                with open(output_file, 'w', encoding='utf-8') as f:
                    for pair in all_pairs:
                        f.write(f"Child: {pair['child']}\n")
                        f.write(f"Adult: {pair['adult']}\n")
                        f.write("-" * 50 + "\n")
            
            print(f"\nData saved to: {output_file}")
            print(f"Format: {format}")
            print(f"Total pairs: {len(all_pairs)}")
        
        # Print statistics
        print(f"\nStatistics:")
        print(f"Files processed: {len(trs_files)}")
        print(f"Total valid pairs: {len(all_pairs)}")
        
        if all_pairs:
            child_lengths = [len(p['child'].split()) for p in all_pairs]
            adult_lengths = [len(p['adult'].split()) for p in all_pairs]
            print(f"Average child utterance: {sum(child_lengths)/len(child_lengths):.1f} words")
            print(f"Average adult response: {sum(adult_lengths)/len(adult_lengths):.1f} words")
        
        return {
            "files_processed": len(trs_files),
            "total_pairs": len(all_pairs),
            "pairs": all_pairs
        }


def main():
    parser = argparse.ArgumentParser(
        description='Extract conversation pairs from TRS files for BARTHEZ + QWEN training'
    )
    parser.add_argument('input_folder', help='Folder containing TRS files')
    parser.add_argument('-o', '--output', help='Output file', 
                       default='conversation_pairs.json')
    parser.add_argument('--format', choices=['json', 'jsonl', 'txt'], 
                       default='json', help='Output format')
    parser.add_argument('--min-child-words', type=int, default=2, 
                       help='Minimum words in child utterance')
    parser.add_argument('--min-adult-words', type=int, default=4, 
                       help='Minimum words in adult response')
    parser.add_argument('--max-pairs', type=int, help='Maximum number of pairs to extract')
    parser.add_argument('--l2-mode', action='store_true', 
                       help='Enable L2 learner mode: filters child-specific content and prioritizes clear educational exchanges')
    
    args = parser.parse_args()
    
    extractor = TRSExtractor()
    extractor.min_words_child = args.min_child_words
    extractor.min_words_adult = args.min_adult_words
    
    input_folder = Path(args.input_folder)
    output_file = Path(args.output)
    
    try:
        results = extractor.process_folder(
            input_folder, 
            output_file, 
            format=args.format,
            max_pairs=args.max_pairs,
            l2_mode=args.l2_mode
        )
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())