from ipa_processor import IPAProcessor
import traceback

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
        config_path = config_path or "ipa_config.yaml"
        processor = IPAProcessor(config_path)
        
        # Store config_path for the interactive function
        _process_single_page_interactive.config_path = config_path
        
        menu_options = {
            '1': ('Process a specific page', lambda: _process_single_page_with_processor(processor)),
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
        traceback.print_exc()

def _process_category_interactive(processor):
    """Interactive category processing."""
    category_name = input("Enter category name (without 'Category:' prefix): ").strip()
    depth = int(input("Enter recursion depth (0 for just this category): ").strip() or "0")
    max_pages_str = input("Enter maximum pages (or enter for no limit): ").strip()
    max_pages = int(max_pages_str) if max_pages_str else None
    skip_pages = int(input("Enter pages to skip (or enter for none): ").strip() or "0")
    
    processor.process_category(category_name, depth, max_pages, skip_pages)

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

def _process_single_page_with_processor(processor):
    """Process single page with existing processor instance."""
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
        
        success = processor.process_page(page)
        if success:
            print("Page processed and saved successfully!")
        else:
            print("Page processing completed (no changes saved).")
            
    except Exception as e:
        print(f"Error processing page '{page_title}': {e}")
        traceback.print_exc()

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
