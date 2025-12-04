#DISCLAIMER: GEMINI WROTE THIS

import argparse
import itertools
import sys

def compare_files(file1_path, file2_path):
    try:
        with open(file1_path, 'r', encoding='utf-8') as f1, \
             open(file2_path, 'r', encoding='utf-8') as f2:
            
            # Use zip_longest to handle files of different lengths
            for i, (line1, line2) in enumerate(itertools.zip_longest(f1, f2), 1):
                
                # Check for mismatch
                if line1 != line2:
                    print(f"[-] Mismatch found at line {i}:")
                    print(f"    File 1: {line1.strip() if line1 else '(End of file)'}")
                    print(f"    File 2: {line2.strip() if line2 else '(End of file)'}")
                    return False
            
            print("[+] Files are identical.")
            return True

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False

def main():
    # Initialize the parser
    parser = argparse.ArgumentParser(
        description="Compare two text files row by row to check if they are identical."
    )
    
    # Add arguments for the two files
    parser.add_argument("file1", help="Path to the first file")
    parser.add_argument("file2", help="Path to the second file")
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Run comparison
    is_identical = compare_files(args.file1, args.file2)
    
    # Exit with standard status codes (0 = Success/Same, 1 = Error/Diff)
    sys.exit(0 if is_identical else 1)

if __name__ == "__main__":
    main()