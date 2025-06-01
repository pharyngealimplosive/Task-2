import pywikibot
from mwparserfromhell import parse, nodes
import urllib.parse
import re
import unicodedata
from dataclasses import dataclass
from typing import Set, List, Tuple, Optional, Dict, Any, Union
from functools import lru_cache
import yaml
import sys
from pathlib import Path
import traceback

try:
    import panphon
except ImportError:
    print("Error: panphon is required but not installed.")
    print("Please install it with: !pip install panphon")
    sys.exit(127) # Exit code 127: command/module not found

# Pre-intern common strings for performance
_STRINGS = {s: sys.intern(s) for s in ['ipa', 'separator', 'IPA link']}

@dataclass
class ProcessingStats:
    __slots__ = ['changes', 'processed_count', 'modified_count', 'skipped_count']
    
    def __init__(self):
        self.changes = self.processed_count = self.modified_count = self.skipped_count = 0

class IPAProcessor:
    """Optimized IPA processor with panphon integration."""
    
    def __init__(self, config_path: str = "ipa_config.yaml"):
        self.config_path = config_path
        self._load_config()
        self._compile_patterns()
        self._init_panphon()
        self.stats = ProcessingStats()
        print(f"IPA Processor initialized with config: {config_path}")
        print("Panphon integration enabled for enhanced IPA detection")

        # Retry configuration
        self.max_retries = 3
        self.base_delay = 1.0  # Base delay for exponential backoff
        self.max_delay = 30.0  # Maximum delay cap
    
    def _init_panphon(self):
        """Initialize panphon for IPA symbol validation."""
        try:
            self.ft = panphon.FeatureTable()
            print("Panphon FeatureTable loaded successfully")
        except Exception as e:
            print(f"Error: Could not initialize panphon FeatureTable: {e}")
            sys.exit(127)
    
    def _load_config(self):
        """Load and cache configuration with consolidated bracket logic."""
        config_file = Path(self.config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(config_file, 'r', encoding='utf-8') as f:
            c = yaml.safe_load(f)
        
        # Store essential config values directly as attributes for faster access
        self.max_ipa_length = c.get('max_ipa_length', 1)
        
        # Consolidated bracket processing
        self.brackets = {}
        self.bracket_chars = set()
        self.special_brackets = {}
        
        # Process IPA brackets with special templates first
        ipa_brackets_raw = c.get('ipa_brackets', {})
        for open_b, bracket_data in ipa_brackets_raw.items():
            close_b, template_name = bracket_data
            self.special_brackets[sys.intern(open_b)] = (sys.intern(close_b), sys.intern(template_name))
            self.brackets[open_b] = close_b
            self.bracket_chars.update([open_b, close_b])
        
        # Add regular brackets
        regular_brackets = c.get('brackets', {})
        for open_b, close_b in regular_brackets.items():
            if open_b not in self.brackets:  # Don't override special brackets
                self.brackets[open_b] = close_b
                self.bracket_chars.update([open_b, close_b])
        
        # Convert to frozensets for faster lookups
        self.bracket_chars = frozenset(self.bracket_chars)
        self.tone_symbols = frozenset(c.get('tone_symbols', []))
        self.multi_char_exceptions = frozenset(c.get('multi_char_exceptions', []))
        self.non_ipa_diacritics = frozenset(c.get('non_ipa_diacritics', []))
        self.separator_symbols = frozenset(c.get('separator_symbols', []))
        self.phonetic_terms = frozenset(term.lower() for term in c.get('phonetic_terms', []))
        self.valid_ipa_diacritics = frozenset(c.get('superscripts', []))
    
    def _compile_patterns(self):
        """Compile all regex patterns once."""
        separator_pattern = '|'.join(re.escape(sep) for sep in self.separator_symbols)
        self.separator_split_pattern = re.compile(rf'(\s*)({separator_pattern})(\s*)')
        self.table_pattern = re.compile(r'(?:^|\n)\s*:{0,4}\s*\{\|.*?\n\|\}', re.MULTILINE | re.DOTALL)
        self.word_pattern = re.compile(r'\b\w+\b')
        self.space_pattern = re.compile(r'(\s+)')
    
    @lru_cache(maxsize=2000)
    def is_valid_ipa_symbol(self, char: str) -> bool:
        """Check if a character is a valid IPA symbol using panphon."""
        if not char or char.isspace():
            return False
        
        # Check if it's in non-IPA diacritics (should be excluded)
        if char in self.non_ipa_diacritics:
            return False
        
        # Check if it's a valid IPA diacritic/superscript
        if char in self.valid_ipa_diacritics:
            return True
        
        # Tie bars are valid IPA structural elements
        if char in ['͡', '͜']:  # U+0361 and U+035C - combining double inverted breve (tie bars)
            return True
        
        # Ejective marker is a valid IPA symbol
        if char == 'ʼ':  # U+02BC MODIFIER LETTER APOSTROPHE
            return True
        
        # Length markers are valid IPA symbols
        if char in ['ː', 'ˑ']:  # U+02D0 and U+02D1 - length markers
            return True
    
        # Use panphon for IPA symbol validation
        if self.ft.fts(char):
            return True
        
        # Check normalized form
        normalized = unicodedata.normalize('NFD', char)
        return any(self.ft.seg_known(c) for c in normalized)
    
    @lru_cache(maxsize=1000)
    def _get_phonetic_features(self, char: str) -> Tuple[bool, bool, bool]:
        """Get phonetic features for a character: (is_syllabic, is_sonorant, is_consonantal)"""
        if not char or char.isspace():
            return False, False, False
        
        # Don't treat non-IPA diacritics as having phonetic features
        if char in self.non_ipa_diacritics:
            return False, False, False
        
        # Ejective marker itself doesn't have primary phonetic features
        # but shouldn't block analysis of the base sound
        if char == 'ʼ':
            return False, False, False
        
        # Length markers don't have primary phonetic features but are valid IPA
        if char == 'ː':
            return False, False, False

        if char == 'ˑ':
            return False, False, False
        
        # Valid IPA diacritics should not interfere with phonetic analysis
        if char in self.valid_ipa_diacritics:
            return False, False, False  # Diacritics don't have primary phonetic features
        
        features = self.ft.fts(char)
        if not features:
            # Try normalized form
            normalized = unicodedata.normalize('NFD', char)
            for c in normalized:
                if c not in ['ʼ', 'ː', 'ˑ']:  # Skip ejective marker and length markers in normalized analysis
                    features = self.ft.fts(c)
                    if features:
                        break
        
        if not features:
            return False, False, False
        
        return (features.get('syl', 0) > 0, 
                features.get('son', 0) > 0, 
                features.get('cons', 0) > 0)
    
    @lru_cache(maxsize=1000)
    def _count_vowels_and_get_type(self, segment: str) -> Tuple[int, str]:
        """Count vowels and determine sound type using panphon features."""
        vowel_count = 0
        sound_type = 'other'
        
        for char in segment:
            if char.isspace():
                continue
            
            is_syllabic, is_sonorant, is_consonantal = self._get_phonetic_features(char)
            
            # Count vowels and determine primary sound type
            if is_syllabic or (is_sonorant and not is_consonantal):
                vowel_count += 1
                if sound_type == 'other':
                    sound_type = 'vowel'
            elif is_consonantal or not is_sonorant:
                if sound_type == 'other':
                    sound_type = 'consonant'
        
        return vowel_count, sound_type
    
    @lru_cache(maxsize=500)
    def _is_valid_consonant_cluster(self, cluster: str) -> bool:
        """Check if a consonant cluster is valid based on multi_char_exceptions."""
        if cluster == 'ɖ𝼅' or cluster == 'ʈꞎ':  # problematic segments
            return True
        
        # First check the cluster exactly as provided
        if cluster in self.multi_char_exceptions:
            return True
        
        normalized = unicodedata.normalize('NFD', cluster)
        
        # Check normalized form
        if normalized in self.multi_char_exceptions:
            return True
        
        # For clusters with tie bars, check if individual components are valid IPA
        if '͡' in normalized or '͜' in normalized:
            # Split by tie bars and check each component
            components = []
            current_component = ''
            
            for char in normalized:
                if char in ['͡', '͜']:  # Tie bar characters
                    if current_component:
                        components.append(current_component)
                        current_component = ''
                else:
                    current_component += char
            
            if current_component:  # Add the last component
                components.append(current_component)
            
            # Check if all components are valid IPA sounds
            if len(components) >= 2:
                all_valid = True
                for component in components:
                    # Check if component is valid IPA using panphon
                    has_valid_ipa = False
                    for char in component:
                        if self.is_valid_ipa_symbol(char):
                            has_valid_ipa = True
                            break
                    
                    if not has_valid_ipa:
                        all_valid = False
                        break
                
                if all_valid:
                    return True
        
        # For clusters with ejectives, also check base form
        base_cluster = ''
        for c in normalized:
            if c == 'ʼ':  # Keep ejective markers
                base_cluster += c
            elif c in ['͡', '͜']:  # Keep tie bars for affricates like ɡ͡b
                base_cluster += c
            elif not unicodedata.category(c).startswith('M'):  # Keep non-combining marks
                base_cluster += c
            # Skip other combining diacritics but preserve structural elements
        
        # Check the base cluster with structural elements preserved
        if base_cluster in self.multi_char_exceptions:
            return True
        
        # As a fallback, check variants with and without ejective for flexibility
        if 'ʼ' in base_cluster:
            no_ejective = base_cluster.replace('ʼ', '')
            if no_ejective in self.multi_char_exceptions:
                return True
        
        # Only accept clusters that are explicitly in the multi_char_exceptions list
        return False
    
    def _analyze_sound_sequence(self, segment: str) -> Tuple[bool, List[str]]:
        """Analyze sound sequences for invalid patterns."""
        # Remove brackets, spaces, and valid diacritics for analysis
        # But preserve ejective markers and length markers as they're part of the sound
        clean_segment = ''
        i = 0
        while i < len(segment):
            char = segment[i]
            if char in self.bracket_chars or char.isspace():
                pass  # Skip brackets and spaces
            elif char in self.valid_ipa_diacritics and char not in ['ʼ', 'ː']:
                pass  # Skip regular diacritics but keep ejective markers and length markers
            else:
                clean_segment += char
            i += 1
        
        if not clean_segment:
            return False, []
    
        # FIRST: Check if the entire segment is a known valid exception
        if clean_segment in self.multi_char_exceptions:
            return False, []  # No invalid sequences if it's a known exception
        
        # Also check normalized form
        normalized_segment = unicodedata.normalize('NFD', clean_segment)
        if normalized_segment in self.multi_char_exceptions:
            return False, []
        
        normalized = unicodedata.normalize('NFD', clean_segment)
        sound_data = []
        
        i = 0
        while i < len(normalized):
            char = normalized[i]
            
            # Handle ejective sequences as single units
            if i < len(normalized) - 1 and normalized[i + 1] == 'ʼ':
                # Treat base + ejective as single sound unit
                base_char = normalized[i]
                if not unicodedata.category(base_char).startswith('M'):
                    _, sound_type = self._count_vowels_and_get_type(base_char)
                    if sound_type != 'other':
                        sound_data.append((base_char + 'ʼ', sound_type))
                i += 2  # Skip both base and ejective
            elif not unicodedata.category(char).startswith('M') and char not in ['ʼ', 'ː', 'ˑ']:
                # Regular character (not combining mark, not standalone ejective/length marker)
                _, sound_type = self._count_vowels_and_get_type(char)
                if sound_type != 'other':
                    sound_data.append((char, sound_type))
                i += 1
            else:
                i += 1
        
        if len(sound_data) < 2:
            return False, []
        
        invalid_sequences = []
        i = 0
        
        while i < len(sound_data) - 1:
            current_char, current_type = sound_data[i]
            next_char, next_type = sound_data[i + 1]
            
            # Check for invalid CV/VC patterns
            if ((current_type == 'consonant' and next_type == 'vowel') or
                (current_type == 'vowel' and next_type == 'consonant')):
                invalid_sequences.append(f"{current_type[0].upper()}{next_type[0].upper()}: {current_char}{next_char}")
                i += 1
            # Check consonant clusters - ONLY flag as invalid if not in exceptions
            elif current_type == 'consonant' and next_type == 'consonant':
                cluster_end = i + 1
                while cluster_end < len(sound_data) and sound_data[cluster_end][1] == 'consonant':
                    cluster_end += 1
                
                cluster = ''.join(sound_data[j][0] for j in range(i, cluster_end))
                
                # FIXED: Only validate consonant clusters if the whole segment is NOT already valid
                if not self._is_valid_consonant_cluster(cluster):
                    invalid_sequences.append(f"CC: {cluster}")
                
                i = cluster_end - 1
            else:
                i += 1
        
        return len(invalid_sequences) > 0, invalid_sequences
    
    @lru_cache(maxsize=2000)
    def analyze_segment(self, segment: str) -> Dict[str, Any]:
        """Single-pass segment analysis with caching and panphon integration."""
        
        seg_clean = ''.join(c for c in segment if c not in self.bracket_chars and not c.isspace())
        
        # Enhanced IPA validation with panphon
        contains_valid_ipa = False
        if seg_clean.strip():
            # FIRST: Check if it's in multi_char_exceptions (this should override other checks)
            if seg_clean in self.multi_char_exceptions:
                contains_valid_ipa = True
            elif unicodedata.normalize('NFD', seg_clean) in self.multi_char_exceptions:
                contains_valid_ipa = True
            else:
                # Check for tie bar constructions first
                if '͡' in seg_clean or '͜' in seg_clean:
                    # For tie bar constructions, check if components are valid IPA
                    components = []
                    current_component = ''
                    
                    for char in seg_clean:
                        if char in ['͡', '͜']:  # Tie bar characters
                            if current_component:
                                components.append(current_component)
                                current_component = ''
                        else:
                            current_component += char
                    
                    if current_component:  # Add the last component
                        components.append(current_component)
                    
                    # Check if any component contains valid IPA
                    for component in components:
                        if any(self.is_valid_ipa_symbol(c) for c in component if c not in ['ʼ', 'ː', 'ˑ']):
                            contains_valid_ipa = True
                            break
                
                # Regular IPA validation if not already found
                if not contains_valid_ipa:
                    if any(self.is_valid_ipa_symbol(c) for c in seg_clean if c not in ['ʼ', 'ː', 'ˑ']):
                        contains_valid_ipa = True
                
                # Special handling for ejective sequences (including complex ones like t͡ʃʼ)
                if not contains_valid_ipa and 'ʼ' in seg_clean:
                    # Find ejective marker positions
                    ejective_pos = seg_clean.find('ʼ')
                    while ejective_pos != -1:
                        # Check the sequence before the ejective marker
                        if ejective_pos > 0:
                            # For affricates like t͡ʃʼ, check the whole sequence before ʼ
                            base_sequence = seg_clean[:ejective_pos]
                            
                            # Check if base sequence contains valid IPA characters or tie bar constructions
                            has_valid_base = False
                            if '͡' in base_sequence or '͜' in base_sequence:
                                # Handle tie bar constructions
                                for char in base_sequence:
                                    if char not in ['͡', '͜'] and self.is_valid_ipa_symbol(char):
                                        has_valid_base = True
                                        break
                            else:
                                # Regular check
                                for char in base_sequence:
                                    if self.is_valid_ipa_symbol(char):
                                        has_valid_base = True
                                        break
                            
                            # Also check if it's a known multi-character exception
                            if has_valid_base or base_sequence in self.multi_char_exceptions:
                                contains_valid_ipa = True
                                break
                        
                        # Find next ejective marker
                        ejective_pos = seg_clean.find('ʼ', ejective_pos + 1)
                
                # Special handling for length markers - check if they follow valid IPA sounds
                if not contains_valid_ipa and ('ː' in seg_clean or 'ˑ' in seg_clean):
                    for length_marker in ['ː', 'ˑ']:
                        length_pos = seg_clean.find(length_marker)
                        while length_pos != -1:
                            # Check the sequence before the length marker
                            if length_pos > 0:
                                # Check if there's a valid IPA sound before the length marker
                                preceding_sequence = seg_clean[:length_pos]
                                
                                # Handle tie bar constructions before length markers
                                if '͡' in preceding_sequence or '͜' in preceding_sequence:
                                    for char in preceding_sequence:
                                        if char not in ['͡', '͜'] and self.is_valid_ipa_symbol(char):
                                            contains_valid_ipa = True
                                            break
                                else:
                                    # Check last character before length marker
                                    preceding_char = seg_clean[length_pos - 1]
                                    if self.is_valid_ipa_symbol(preceding_char):
                                        contains_valid_ipa = True
                                        break
                            
                            # Find next length marker
                            length_pos = seg_clean.find(length_marker, length_pos + 1)
                        
                        if contains_valid_ipa:
                            break
                
                # If segment contains only length markers, ejectives, or tie bars with valid IPA, it's valid
                if not contains_valid_ipa:
                    # Check if any character is valid IPA (including structural elements)
                    for char in seg_clean:
                        if self.is_valid_ipa_symbol(char):
                            contains_valid_ipa = True
                            break
        
        has_tone = any(c in self.tone_symbols for c in segment) or segment in self.tone_symbols
        
        # Check for non-IPA diacritics (but preserve ejectives, length markers, and tie bars)
        has_non_ipa = False
        normalized_seg = unicodedata.normalize('NFD', seg_clean)
        for char in normalized_seg:
            # Don't treat ejective markers, length markers, or tie bars as non-IPA
            if char in ['ʼ', 'ː', 'ˑ', '͡', '͜']:
                continue
            if char in self.non_ipa_diacritics:
                has_non_ipa = True
                break
        
        # Count vowels for diphthong detection (preserve structural elements)
        clean_for_vowel_count = ''
        i = 0
        while i < len(seg_clean):
            char = seg_clean[i]
            # Keep ejective markers, length markers, and tie bars in vowel counting, skip other diacritics
            if char in self.valid_ipa_diacritics and char not in ['ʼ', 'ː', 'ˑ', '͡', '͜']:
                pass  # Skip regular diacritics but keep structural elements
            else:
                clean_for_vowel_count += char
            i += 1
        
        vowel_count, _ = self._count_vowels_and_get_type(clean_for_vowel_count)
        is_diphthong = vowel_count >= 2
        
        # Check for invalid sequences - but override if segment is in multi_char_exceptions
        has_invalid = False
        invalid_list = []
        if seg_clean not in self.multi_char_exceptions and unicodedata.normalize('NFD', seg_clean) not in self.multi_char_exceptions:
            has_invalid, invalid_list = self._analyze_sound_sequence(segment)
        
        # Should link determination - FIXED: be more restrictive for consonant clusters
        should_link = (not has_tone and not is_diphthong and not has_non_ipa and
                      not has_invalid and seg_clean.strip() and contains_valid_ipa)
        
        return {
            'has_tone': has_tone,
            'is_diphthong': is_diphthong,
            'has_non_ipa': has_non_ipa,
            'has_invalid_sequences': has_invalid,
            'invalid_sequences': invalid_list,
            'should_link': should_link,
            'clean': seg_clean,
            'contains_valid_ipa': contains_valid_ipa
        }
    
    def detect_ipa_brackets(self, segment: str) -> Tuple[Optional[str], Optional[str], str, Optional[str]]:
        """Fast bracket detection with consolidated logic and special slash handling."""
        segment = segment.strip()
        
        # Special handling for slashes - check if they form phonemic brackets
        if segment.startswith('/') and segment.endswith('/') and segment.count('/') == 2:
            # This is a phonemic transcription like /x~y/ - treat as IPA brackets
            content = segment[1:-1].strip()
            template_name = self.special_brackets.get('/', (None, None))[1] if '/' in self.special_brackets else None
            return '/', '/', content, template_name
        
        # Check all other brackets in order of priority (excluding slashes for now)
        for open_b in self.brackets:
            if open_b == '/':  # Skip slashes, handled above
                continue
            close_b = self.brackets[open_b]
            if segment.startswith(open_b) and segment.endswith(close_b):
                content = segment[len(open_b):-len(close_b)].strip()
                template_name = self.special_brackets.get(open_b, (None, None))[1]
                return open_b, close_b, content, template_name
        
        return None, None, segment, None

    def _should_treat_slash_as_separator(self, content: str) -> bool:
        """Determine if slashes in content should be treated as allophone separators."""
        slash_count = content.count('/')
        
        # If no slashes, not relevant
        if slash_count == 0:
            return False
        
        # If content is already bracketed with slashes and only has 2 slashes total, 
        # those are the bracket slashes, not separators
        if content.startswith('/') and content.endswith('/') and slash_count == 2:
            return False
        
        # If more than 2 slashes, or slashes inside other brackets, treat as separators
        if slash_count > 2:
            return True
        
        # If content has slashes but isn't bracketed with them, treat as separators
        if slash_count >= 1 and not (content.startswith('/') and content.endswith('/')):
            return True
        
        # Check if slashes appear alongside other separators (strong indicator of separator usage)
        other_separators = [sep for sep in self.separator_symbols if sep != '/']
        if any(sep in content for sep in other_separators):
            return True
        
        return False
    
    def tokenize_content(self, content: str) -> List[Union[str, Tuple[str, str, str]]]:
        """Fast tokenization with single regex split and special slash handling."""
        result = []
        
        # Check if slashes should be treated as separators in this content
        treat_slash_as_separator = self._should_treat_slash_as_separator(content)
        
        # Create dynamic separator pattern based on slash treatment
        separator_symbols_to_use = set(self.separator_symbols)
        if treat_slash_as_separator:
            separator_symbols_to_use.add('/')
        
        # Handle HTML entities and tags - normalize &nbsp; before processing
        normalized_content = content.replace('&nbsp;', ' ')
        
        # Create pattern for this specific tokenization - handle HTML tags specially
        html_tags = ['<br/>', '<br />', '<br>']
        regular_separators = [sep for sep in separator_symbols_to_use if sep not in html_tags and sep != '&nbsp;']
        
        # Build pattern with HTML tags first (longer matches), then regular separators
        all_separators = html_tags + regular_separators
        separator_pattern = '|'.join(re.escape(sep) for sep in all_separators)
        
        if separator_pattern:
            dynamic_separator_pattern = re.compile(rf'(\s*)({separator_pattern})(\s*)')
            parts = dynamic_separator_pattern.split(normalized_content)
        else:
            parts = [normalized_content]  # No separators to split on
        
        for part in parts:
            if not part:
                continue
            # Check if this part is a separator (including original &nbsp; check)
            if (part.strip() in separator_symbols_to_use or 
                part.strip() in html_tags or 
                (part.strip() == ' ' and '&nbsp;' in separator_symbols_to_use)):
                result.append((_STRINGS['separator'], part.strip(), ''))
            elif not part.isspace() and part.strip():
                space_parts = self.space_pattern.split(part)
                for space_part in space_parts:
                    if space_part.strip():
                        result.append(space_part.strip())
                    elif space_part.isspace():
                        result.append(space_part)
        return result
    
    def contains_phonetic_terms(self, text: str, min_terms: int = 3) -> Tuple[bool, List[str]]:
        """Fast phonetic term detection."""
        words = set(self.word_pattern.findall(text.lower()))
        matched = [w for w in words if w in self.phonetic_terms]
        return len(matched) >= min_terms, matched[:min_terms] if matched else []
    
    def find_tables(self, text: str) -> List[Tuple[int, int, str]]:
        """Find table boundaries."""
        return [(m.start(), m.end(), m.group()) for m in self.table_pattern.finditer(text)]
    
    def is_in_table(self, pos: int, tables: List[Tuple[int, int, str]]) -> Tuple[bool, Optional[str]]:
        """Check if position is in any table."""
        for start, end, content in tables:
            if start <= pos <= end:
                return True, content
        return False, None
    
    def _create_template_node(self, template_name: str, content: str) -> nodes.Template:
        """Create a template node with given name and content."""
        template = nodes.Template(name=template_name)
        template.add("1", content.strip())
        return template
    
    def _process_segments_to_nodes(self, segments: List, template_name: str = None) -> Tuple[List[nodes.Node], bool]:
        """Convert segments to nodes with validation and proper bracket preservation."""
        new_nodes = []
        has_valid_conversions = False
        default_template = template_name or _STRINGS['IPA link']
        
        for segment in segments:
            if isinstance(segment, tuple) and len(segment) == 3:
                new_nodes.append(nodes.Text(segment[1]))
                if segment[2]:
                    new_nodes.append(nodes.Text(segment[2]))
                continue
            
            if isinstance(segment, str):
                if segment.isspace():
                    new_nodes.append(nodes.Text(segment))
                    continue
                
                if not segment.strip():
                    continue
                
                # Handle bracketed content
                open_b, close_b, core, bracket_template = self.detect_ipa_brackets(segment)
                if not core.strip():
                    # If no core content, add the original segment as text
                    new_nodes.append(nodes.Text(segment))
                    continue
                
                analysis = self.analyze_segment(core)
                
                if not analysis['should_link'] or not analysis['contains_valid_ipa']:
                    if not template_name:  # Only return empty for full replacement
                        return [], False
                    # Preserve original segment including brackets
                    new_nodes.append(nodes.Text(segment))
                    continue
                
                has_valid_conversions = True
                
                # FIXED: For special brackets, DON'T add brackets separately - the template handles them
                if bracket_template:
                    # Special brackets: template includes the bracket styling, don't add brackets
                    new_nodes.append(self._create_template_node(bracket_template, core))
                else:
                    # Regular brackets: add brackets and use default template
                    if open_b:
                        new_nodes.append(nodes.Text(open_b))
                    
                    used_template = default_template
                    new_nodes.append(self._create_template_node(used_template, core))
                    
                    if close_b:
                        new_nodes.append(nodes.Text(close_b))
        
        return new_nodes, has_valid_conversions
    
    def process_ipa_template(self, node: nodes.Template, parent_list: List, index: int) -> None:
        """Process IPA template with optimized logic, panphon validation, and improved bracket preservation."""
        if node.name.strip().lower() != _STRINGS['ipa']:
            return
        
        raw_content = str(node.params[0].value).strip()
        
        # First check if this looks like a simple phonemic transcription
        if raw_content.startswith('/') and raw_content.endswith('/') and raw_content.count('/') == 2:
            inner_content = raw_content[1:-1].strip()
            # Check if it contains separators that would make it allophone
            has_other_separators = any(sep in inner_content for sep in self.separator_symbols if sep != '/')
            if not has_other_separators:
                # This is a simple phonemic transcription, convert to IPA link
                analysis = self.analyze_segment(inner_content)
                if analysis['should_link'] and analysis['contains_valid_ipa']:
                    template_name = self.special_brackets.get('/', (None, None))[1]
                    if template_name:
                        # FIXED: Special brackets - template handles brackets, don't add extra
                        new_template = self._create_template_node(template_name, inner_content)
                        parent_list[index:index+1] = [new_template]
                        self.stats.changes += 1
                        print(f"Converted phonemic: {raw_content} -> {{{{{template_name}|{inner_content}}}}}")
                    return
        
        # Handle bracketed content with potential separators
        open_b, close_b, inner_content, template_name = self.detect_ipa_brackets(raw_content)
        
        # Determine if we should look for separators (including slash logic)
        should_check_separators = (
            any(sep in inner_content for sep in self.separator_symbols) or
            self._should_treat_slash_as_separator(raw_content)
        )
        
        if should_check_separators:
            segments = self.tokenize_content(inner_content)
            new_nodes, has_valid_conversions = self._process_segments_to_nodes(segments, template_name)
            
            if new_nodes and has_valid_conversions:
                # FIXED: For special brackets, don't add brackets - templates handle them
                final_nodes = []
                if open_b and not template_name:  # Only add brackets if NOT using special template
                    final_nodes.append(nodes.Text(open_b))
                final_nodes.extend(new_nodes)
                if close_b and not template_name:  # Only add brackets if NOT using special template
                    final_nodes.append(nodes.Text(close_b))
                
                parent_list[index:index+1] = final_nodes
                self.stats.changes += 1
                print(f"Converted allophone template with separators: {raw_content}")
            return
        
        # Handle simple special brackets without separators
        if template_name and inner_content.strip():
            analysis = self.analyze_segment(inner_content)
            if analysis['should_link'] and analysis['contains_valid_ipa']:
                # FIXED: Special brackets - template handles brackets, don't add extra
                new_template = self._create_template_node(template_name, inner_content)
                parent_list[index:index+1] = [new_template]
                self.stats.changes += 1
                print(f"Converted: {raw_content} -> {{{{{template_name}|{inner_content.strip()}}}}}")
            return
        
        # Handle simple unbracketed content
        if not template_name and inner_content.strip() and not open_b and not close_b:
            analysis = self.analyze_segment(inner_content)
            if analysis['should_link'] and analysis['contains_valid_ipa']:
                # Convert simple IPA content to IPA link template
                new_template = self._create_template_node(_STRINGS['IPA link'], inner_content)
                parent_list[index:index+1] = [new_template]
                self.stats.changes += 1
                print(f"Converted simple IPA: {raw_content} -> {{{{{_STRINGS['IPA link']}|{inner_content.strip()}}}}}")
            return
        
        # Handle cases where we have brackets but no special template (like parentheses)
        if open_b and close_b and not template_name and inner_content.strip():
            analysis = self.analyze_segment(inner_content)
            if analysis['should_link'] and analysis['contains_valid_ipa']:
                # Preserve brackets and convert content to IPA link
                new_nodes = [
                    nodes.Text(open_b),
                    self._create_template_node(_STRINGS['IPA link'], inner_content),
                    nodes.Text(close_b)
                ]
                parent_list[index:index+1] = new_nodes
                self.stats.changes += 1
                print(f"Converted bracketed IPA: {raw_content} -> {open_b}{{{{{_STRINGS['IPA link']}|{inner_content.strip()}}}}}{close_b}")
            return
        
        # Fallback processing for complex unbracketed content with separators
        segments = self.tokenize_content(raw_content)
        new_nodes, has_valid_conversions = self._process_segments_to_nodes(segments)
        
        if new_nodes and has_valid_conversions:
            parent_list[index:index+1] = new_nodes
            self.stats.changes += 1
            print(f"Converted IPA template: {raw_content}")
    
    def _exponential_backoff_delay(self, attempt: int, base_delay: float = None) -> float:
        """Calculate exponential backoff delay with jitter."""
        if base_delay is None:
            base_delay = self.base_delay
        
        # Calculate exponential delay: base_delay * (2 ^ attempt)
        delay = base_delay * (2 ** attempt)
        
        # Cap the delay at max_delay
        delay = min(delay, self.max_delay)
        
        # Add jitter (±20% randomization) to avoid thundering herd
        jitter = delay * 0.2 * random.uniform(-1, 1)
        final_delay = delay + jitter
        
        return max(0.1, final_delay)  # Ensure minimum 0.1s delay
        
    def _handle_page_save_with_retry(self, page: pywikibot.Page, new_text: str, 
                                   summary: str) -> bool:
        """Save page with exponential backoff retry logic and comprehensive error handling."""
        for attempt in range(self.max_retries):
            try:
                page.text = new_text
                page.save(summary=summary, bot=True)
                print(f"Page saved successfully after {attempt + 1} attempt(s)")
                return True
                
            except pywikibot.exceptions.EditConflictError as e:
                print(f"Edit conflict on attempt {attempt + 1}: {e}")
                if attempt < self.max_retries - 1:
                    delay = self._exponential_backoff_delay(attempt)
                    print(f"Retrying in {delay:.1f} seconds after refreshing page content...")
                    time.sleep(delay)
                    
                    try:
                        # Refresh page content and reapply changes
                        print("Refreshing page content to resolve edit conflict...")
                        refreshed_text = page.get(force=True)  # Force refresh from server
                        
                        # Reprocess the refreshed content
                        wikicode = parse(refreshed_text)
                        old_changes = self.stats.changes
                        self.stats.changes = 0
                        tables = self.find_tables(refreshed_text)
                        self._process_nodes_with_context(wikicode.nodes, 0, tables)
                        
                        if self.stats.changes > 0:
                            new_text = str(wikicode)
                            print(f"Reprocessed content: found {self.stats.changes} changes")
                        else:
                            print("No changes found in refreshed content")
                            return False
                            
                    except Exception as refresh_error:
                        print(f"Error refreshing page content: {refresh_error}")
                        if attempt == self.max_retries - 1:
                            return False
                        continue
                else:
                    print("Max retries exceeded for edit conflict")
                    return False
                    
            except pywikibot.exceptions.OtherPageSaveError as e:
                error_msg = str(e).lower()
                
                if "badtoken" in error_msg:
                    print(f"Bad token error on attempt {attempt + 1}: {e}")
                    if attempt < self.max_retries - 1:
                        delay = self._exponential_backoff_delay(attempt, 0.5)
                        print(f"Refreshing tokens and retrying in {delay:.1f} seconds...")
                        time.sleep(delay)
                        self.refresh_tokens()
                    else:
                        print("Max retries exceeded for token refresh")
                        return False
                        
                elif "readonly" in error_msg or "database" in error_msg:
                    print(f"Database/readonly error on attempt {attempt + 1}: {e}")
                    if attempt < self.max_retries - 1:
                        delay = self._exponential_backoff_delay(attempt, 2.0)  # Longer delay for DB issues
                        print(f"Waiting for database availability, retrying in {delay:.1f} seconds...")
                        time.sleep(delay)
                    else:
                        print("Max retries exceeded for database error")
                        return False
                        
                elif "rate" in error_msg or "throttl" in error_msg:
                    print(f"Rate limiting on attempt {attempt + 1}: {e}")
                    if attempt < self.max_retries - 1:
                        delay = self._exponential_backoff_delay(attempt, 3.0)  # Even longer for rate limits
                        print(f"Rate limited, waiting {delay:.1f} seconds...")
                        time.sleep(delay)
                    else:
                        print("Max retries exceeded for rate limiting")
                        return False
                        
                else:
                    print(f"Other save error on attempt {attempt + 1}: {e}")
                    if attempt < self.max_retries - 1:
                        delay = self._exponential_backoff_delay(attempt)
                        print(f"Retrying in {delay:.1f} seconds...")
                        time.sleep(delay)
                    else:
                        print("Max retries exceeded for other save error")
                        return False
                        
            except pywikibot.exceptions.LockedPageError as e:
                print(f"Page is locked: {e}")
                return False  # Don't retry for locked pages
                
            except pywikibot.exceptions.SpamblacklistError as e:
                print(f"Spam blacklist error: {e}")
                return False  # Don't retry for spam blacklist
                
            except pywikibot.exceptions.TitleblacklistError as e:
                print(f"Title blacklist error: {e}")
                return False  # Don't retry for title blacklist
                
            except pywikibot.exceptions.AbuseFilterDisallowedError as e:
                print(f"Abuse filter disallowed the edit: {e}")
                return False # Don't retry for abuse filter
            
            except Exception as e:
                print(f"Unexpected error on attempt {attempt + 1}: {e}")
                if attempt < self.max_retries - 1:
                    delay = self._exponential_backoff_delay(attempt)
                    print(f"Retrying in {delay:.1f} seconds...")
                    time.sleep(delay)
                else:
                    print("Max retries exceeded for unexpected error")
                    traceback.print_exc()
                    return False
        
        return False
    
    def process_page(self, page: pywikibot.Page) -> bool:
        """Process single page efficiently with redirect handling and improved error handling."""
        # First resolve any redirects
        page = self._resolve_redirect(page)
        
        print(f"\nProcessing: {page.title()}")
        
        try:
            text = page.get()
        except pywikibot.exceptions.NoPageError:
            print("Page doesn't exist!")
            return False
        except pywikibot.exceptions.IsRedirectPageError:
            print("Unexpected redirect encountered during page fetch")
            return False
        except Exception as e:
            print(f"Error fetching page content: {e}")
            return False
        
        try:
            wikicode = parse(text)
        except Exception as e:
            print(f"Error parsing page wikicode: {e}")
            return False
        
        # Reset stats and process
        self.stats.changes = 0
        
        try:
            tables = self.find_tables(text)
            self._process_nodes_with_context(wikicode.nodes, 0, tables)
        except Exception as e:
            print(f"Error processing page content: {e}")
            return False
        
        if self.stats.changes:
            new_text = str(wikicode)
            print(f"\nFound {self.stats.changes} IPA conversion(s)")
            
            try:
                pywikibot.showDiff(text, new_text)
            except Exception as e:
                print(f"Error showing diff: {e}")
                print("Proceeding without diff display...")
            
            if input("Save changes? (y/n): ").lower() == 'y':
                summary = f"IPA conversion in phonetic tables ({self.stats.changes} templates)"
                return self._handle_page_save_with_retry(page, new_text, summary)
        else:
            print("No IPA conversions found.")
        
        return False

    def _process_single_page_interactive():
        """Interactive single page processing with redirect handling."""
        site = pywikibot.Site('en', 'wikipedia')
        
        page_title = input("Enter page title: ").strip()
        if not page_title:
            print("No page title provided.")
            return
        
        try:
            page = pywikibot.Page(site, page_title)
            
            # Check if page exists before processing
            try:
                if not page.exists():
                    print(f"Page '{page_title}' does not exist.")
                    return
            except Exception as e:
                print(f"Error checking if page exists: {e}")
                return
            
            # Get processor instance (this would need to be passed in or accessed globally)
            config_path = getattr(_process_single_page_interactive, 'config_path', 'ipa_config.yaml')
            processor = IPAProcessor(config_path)
            
            success = processor.process_page(page)
            if success:
                print("Page processed and saved successfully!")
            else:
                print("Page processing completed (no changes saved).")
                
        except Exception as e:
            print(f"Error processing page '{page_title}': {e}")
            traceback.print_exc()
    
    def _process_nodes_with_context(self, node_list: List[nodes.Node], text_offset: int, tables: List):
        """Process nodes with table context."""
        i = current_offset = 0
        
        while i < len(node_list):
            node = node_list[i]
            node_str = str(node)
            
            if isinstance(node, nodes.Template) and node.name.strip().lower() == _STRINGS['ipa']:
                in_table, table_content = self.is_in_table(current_offset + text_offset, tables)
                if in_table:
                    is_relevant, _ = self.contains_phonetic_terms(table_content, 3)
                    if is_relevant:
                        self.process_ipa_template(node, node_list, i)
            elif isinstance(node, nodes.Tag) and hasattr(node, 'contents') and hasattr(node.contents, 'nodes'):
                tag_start_len = len(f"<{node.tag}>")
                self._process_nodes_with_context(node.contents.nodes, current_offset + text_offset + tag_start_len, tables)
            
            current_offset += len(node_str)
            i += 1
    
    def _resolve_redirect(self, page: pywikibot.Page) -> pywikibot.Page:
        """Follow redirects to get the actual target page."""
        original_title = page.title()
        
        try:
            if page.isRedirectPage():
                print(f"'{original_title}' is a redirect page.")
                target = page.getRedirectTarget()
                print(f"Following redirect to: '{target.title()}'")
                
                # Check if the target is also a redirect (handle redirect chains)
                redirect_chain = [original_title]
                current_page = target
                max_redirects = 5  # Prevent infinite loops
                
                while current_page.isRedirectPage() and len(redirect_chain) < max_redirects:
                    redirect_chain.append(current_page.title())
                    current_page = current_page.getRedirectTarget()
                    print(f"Following chained redirect to: '{current_page.title()}'")
                
                if len(redirect_chain) >= max_redirects:
                    print(f"Warning: Stopped following redirects after {max_redirects} hops to prevent loops")
                
                print(f"Final target page: '{current_page.title()}'")
                return current_page
            else:
                print(f"'{original_title}' is not a redirect page.")
                return page
                
        except pywikibot.exceptions.CircularRedirectError as e:
            print(f"Circular redirect detected: {e}")
            print(f"Using original page: '{original_title}'")
            return page
            
        except pywikibot.exceptions.InterwikiRedirectPageError as e:
            print(f"Interwiki redirect (cannot follow): {e}")
            print(f"Using original page: '{original_title}'")
            return page
            
        except Exception as e:
            print(f"Error resolving redirect: {e}")
            print(f"Using original page: '{original_title}'")
            return page
    
    def process_category(self, category_name: str, depth: int = 0, 
                        max_pages: Optional[int] = None, skip_pages: int = 0) -> Tuple[int, int]:
        """Process category efficiently."""
        site = pywikibot.Site('en', 'wikipedia')
        cat = pywikibot.Category(site, f"Category:{category_name}")
        
        print(f"\n=== Processing Category: {cat.title()} ===")
        
        all_pages = list(cat.articles(recurse=depth))
        pages = [page for page in all_pages if page.namespace() == 0]
        
        print(f"Found {len(all_pages)} total pages, {len(pages)} article space pages")
        if max_pages:
            print(f"Will process up to {max_pages} pages")
        if skip_pages:
            print(f"Skipping first {skip_pages} pages")
        
        # Reset stats
        self.stats.processed_count = self.stats.modified_count = self.stats.skipped_count = 0
        
        for i, page in enumerate(pages):
            if max_pages and self.stats.processed_count >= max_pages:
                break
            
            if self.stats.skipped_count < skip_pages:
                self.stats.skipped_count += 1
                continue
            
            print(f"\n[{i+1}/{len(pages)}] Processing article: {page.title()}")
            
            try:
                if self.process_page(page):
                    self.stats.modified_count += 1
                self.stats.processed_count += 1
            except Exception as e:
                print(f"Error processing page {page.title()}: {e}")
        
        print(f"\n=== Category Processing Complete ===")
        print(f"Processed {self.stats.processed_count} pages")
        print(f"Made changes to {self.stats.modified_count} pages")
        
        return self.stats.processed_count, self.stats.modified_count
    
    def refresh_tokens(self):
        """Force refresh of authentication tokens."""
        site = pywikibot.Site('en', 'wikipedia')
        site.tokens.clear()  # Clear cached tokens
        site.get_tokens('csrf')  # Get fresh CSRF token
        print("Authentication tokens refreshed")
    
    def reload_config(self):
        """Reload configuration and clear caches."""
        self._load_config()
        self._compile_patterns()
        # Clear all method caches
        for method_name in ['analyze_segment', 'is_valid_ipa_symbol', '_count_vowels_and_get_type', 
                           '_get_phonetic_features', '_is_valid_consonant_cluster']:
            if hasattr(self, method_name):
                getattr(self, method_name).cache_clear()
        print(f"Configuration reloaded from {self.config_path}")
