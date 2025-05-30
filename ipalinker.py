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
        
        # Valid IPA diacritics should not interfere with phonetic analysis
        if char in self.valid_ipa_diacritics:
            return False, False, False  # Diacritics don't have primary phonetic features
        
        features = self.ft.fts(char)
        if not features:
            # Try normalized form
            normalized = unicodedata.normalize('NFD', char)
            for c in normalized:
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
        normalized = unicodedata.normalize('NFD', cluster)
        base_cluster = ''.join(c for c in normalized if not unicodedata.category(c).startswith('M'))
        return base_cluster in self.multi_char_exceptions
    
    def _analyze_sound_sequence(self, segment: str) -> Tuple[bool, List[str]]:
        """Analyze sound sequences for invalid patterns."""
        # Remove brackets, spaces, and valid diacritics for analysis
        clean_segment = ''.join(c for c in segment 
                               if c not in self.bracket_chars 
                               and not c.isspace() 
                               and c not in self.valid_ipa_diacritics)
        
        if not clean_segment:
            return False, []
        
        normalized = unicodedata.normalize('NFD', clean_segment)
        sound_data = []
        
        for char in normalized:
            if not unicodedata.category(char).startswith('M'):  # Skip combining marks/diacritics
                _, sound_type = self._count_vowels_and_get_type(char)
                if sound_type != 'other':
                    sound_data.append((char, sound_type))
        
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
            # Check consonant clusters
            elif current_type == 'consonant' and next_type == 'consonant':
                cluster_end = i + 1
                while cluster_end < len(sound_data) and sound_data[cluster_end][1] == 'consonant':
                    cluster_end += 1
                
                cluster = ''.join(sound_data[j][0] for j in range(i, cluster_end))
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
        contains_valid_ipa = any(self.is_valid_ipa_symbol(char) for char in seg_clean) if seg_clean.strip() else False
        
        has_tone = any(c in self.tone_symbols for c in segment) or segment in self.tone_symbols
        
        # Updated to not treat valid IPA diacritics as non-IPA
        has_non_ipa = any(c in self.non_ipa_diacritics for c in unicodedata.normalize('NFD', seg_clean))
        
        # Count vowels for diphthong detection (excluding diacritics)
        clean_for_vowel_count = ''.join(c for c in seg_clean if c not in self.valid_ipa_diacritics)
        vowel_count, _ = self._count_vowels_and_get_type(clean_for_vowel_count)
        is_diphthong = vowel_count >= 2
        
        # Check for invalid sequences
        has_invalid, invalid_list = self._analyze_sound_sequence(segment)
        
        # Should link determination
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
        """Fast bracket detection with consolidated logic."""
        segment = segment.strip()
        
        # Check all brackets in order of priority
        for open_b in self.brackets:
            close_b = self.brackets[open_b]
            if segment.startswith(open_b) and segment.endswith(close_b):
                content = segment[len(open_b):-len(close_b)].strip()
                template_name = self.special_brackets.get(open_b, (None, None))[1]
                return open_b, close_b, content, template_name
        
        return None, None, segment, None
    
    def tokenize_content(self, content: str) -> List[Union[str, Tuple[str, str, str]]]:
        """Fast tokenization with single regex split."""
        result = []
        parts = self.separator_split_pattern.split(content)
        
        for part in parts:
            if not part:
                continue
            if part.strip() in self.separator_symbols:
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
        """Convert segments to nodes with validation."""
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
                    continue
                
                analysis = self.analyze_segment(core)
                
                if not analysis['should_link'] or not analysis['contains_valid_ipa']:
                    if not template_name:  # Only return empty for full replacement
                        return [], False
                    new_nodes.append(nodes.Text(segment))
                    continue
                
                has_valid_conversions = True
                
                # Add opening bracket if present
                if open_b:
                    new_nodes.append(nodes.Text(open_b))
                
                # Create appropriate template
                used_template = bracket_template or default_template
                new_nodes.append(self._create_template_node(used_template, core))
                
                # Add closing bracket if present
                if close_b:
                    new_nodes.append(nodes.Text(close_b))
        
        return new_nodes, has_valid_conversions
    
    def process_ipa_template(self, node: nodes.Template, parent_list: List, index: int) -> None:
        """Process IPA template with optimized logic and panphon validation."""
        if node.name.strip().lower() != _STRINGS['ipa']:
            return
        
        raw_content = str(node.params[0].value).strip()
        open_b, close_b, inner_content, template_name = self.detect_ipa_brackets(raw_content)
        
        # Handle content with separators
        if template_name and any(sep in inner_content for sep in self.separator_symbols):
            segments = self.tokenize_content(inner_content)
            new_nodes, has_valid_conversions = self._process_segments_to_nodes(segments, template_name)
            
            if new_nodes and has_valid_conversions:
                parent_list[index:index+1] = new_nodes
                self.stats.changes += 1
                print(f"Converted allophone template: {raw_content}")
            return
        
        # Handle simple special brackets
        if template_name and inner_content.strip():
            analysis = self.analyze_segment(inner_content)
            if analysis['should_link'] and analysis['contains_valid_ipa']:
                new_template = self._create_template_node(template_name, inner_content)
                parent_list[index:index+1] = [new_template]
                self.stats.changes += 1
                print(f"Converted: {raw_content} -> {{{{{template_name}|{inner_content.strip()}}}}}")
            return
        
        # Fallback processing
        segments = self.tokenize_content(raw_content)
        new_nodes, has_valid_conversions = self._process_segments_to_nodes(segments)
        
        if new_nodes and has_valid_conversions:
            parent_list[index:index+1] = new_nodes
            self.stats.changes += 1
            print(f"Converted IPA template: {raw_content}")
    
    def process_page(self, page: pywikibot.Page) -> bool:
        """Process single page efficiently."""
        print(f"\nProcessing: {page.title()}")
        
        try:
            text = page.get()
        except pywikibot.exceptions.NoPageError:
            print("Page doesn't exist!")
            return False
        
        wikicode = parse(text)
        self.stats.changes = 0
        tables = self.find_tables(text)
        
        self._process_nodes_with_context(wikicode.nodes, 0, tables)
        
        if self.stats.changes:
            new_text = str(wikicode)
            print(f"\nFound {self.stats.changes} IPA conversion(s)")
            pywikibot.showDiff(text, new_text)
            
            if input("Save changes? (y/n): ").lower() == 'y':
                page.text = new_text
                page.save(summary=f"IPA conversion in phonetic tables ({self.stats.changes} templates)", bot=True)
                return True
        else:
            print("No IPA templates needed conversion")
        
        return False
    
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

def main():
    """Main entry point with consolidated menu logic."""
    # URL decoding workaround
    if not hasattr(pywikibot, 'tools'):
        pywikibot.tools = type('', (), {})()
        pywikibot.tools.chars = type('', (), {})()
        pywikibot.tools.chars.url2string = lambda text, encodings=None: urllib.parse.unquote(
            text, encoding=(encodings or ['utf-8'])[0]
        )
    
    try:
        site = pywikibot.Site('en', 'wikipedia')
        
        if not site.logged_in():
            print("Not logged in. Please check your authentication.")
            return
        
        print(f"Successfully logged in as: {site.username()}")
        
        config_path = input("Enter config file path (or press Enter for default 'ipa_config.yaml'): ").strip()
        processor = IPAProcessor(config_path or "ipa_config.yaml")
        
        menu_options = {
            '1': ('Process a specific page', lambda: processor.process_page(
                pywikibot.Page(site, input("Enter page title: ").strip()))),
            '2': ('Process a category', lambda: _process_category_interactive(processor)),
            '3': ('Reload configuration', processor.reload_config),
            '4': ('Test IPA symbol validation', lambda: _test_ipa_symbols(processor)),
            '5': ('Exit', lambda: _exit_program())
        }
        
        while True:
            print("\nOptions:")
            for key, (description, _) in menu_options.items():
                print(f"{key}. {description}")
            
            choice = input("Enter your choice (1-5): ").strip()
            
            if choice in menu_options:
                _, action = menu_options[choice]
                result = action()
                if result == 'exit':
                    break
            else:
                print("Invalid choice. Please enter 1, 2, 3, 4, or 5.")
    
    except KeyboardInterrupt:
        print("\nOperation interrupted by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

def _process_category_interactive(processor):
    """Interactive category processing."""
    category_name = input("Enter category name (without 'Category:' prefix): ").strip()
    depth = int(input("Enter recursion depth (0 for just this category): ").strip() or "0")
    max_pages_str = input("Enter maximum pages (or enter for no limit): ").strip()
    max_pages = int(max_pages_str) if max_pages_str else None
    skip_pages = int(input("Enter pages to skip (or enter for none): ").strip() or "0")
    
    processor.process_category(category_name, depth, max_pages, skip_pages)

def _test_ipa_symbols(processor):
    """Test IPA symbol validation."""
    test_symbols = input("Enter IPA symbols to test (space-separated): ").strip().split()
    print("\nValidation results:")
    for symbol in test_symbols:
        is_valid = processor.is_valid_ipa_symbol(symbol)
        analysis = processor.analyze_segment(symbol)
        print(f"'{symbol}': Valid IPA = {is_valid}, Should Link = {analysis['should_link']}")

def _exit_program():
    """Exit the program."""
    print("Exiting program.")
    return 'exit'

if __name__ == "__main__":
    main()
