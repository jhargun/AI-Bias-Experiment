import json
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Converts raw IPUMS text data from website to JSON")

    parser.add_argument("rawFilePath", type=str, help="Path to the raw IPUMS txt file")
    parser.add_argument("outFilePath", type=str, help="Path to the output JSON file")
    parser.add_argument("-v", "--verbose", action="store_true", help="Prints out information about skipped entries")

    args = parser.parse_args()

    assert('.txt' in args.rawFilePath), f"Error: rawFilePath must be a .txt file, not {args.rawFilePath}"
    assert('.json' in args.outFilePath), f"Error: outFilePath must be a .json file, not {args.outFilePath}"

    mappings = {}

    # Read raw data
    with(open(args.rawFilePath, "r")) as f:
        lines = f.readlines()

    if args.verbose:
        print(f"{len(lines)} lines read from {args.rawFilePath}")
        skipped = 0

    # Process raw data into correct format
    for line in lines:
        lineList = line.split()
        assert(len(lineList) >= 3), f"Error in line {line}: not enough entries"

        # When copying data from IPUMS, this is the character that is used to indicate that there are no entries in a category
        if(lineList[-1] == "Â·"):
            if args.verbose:
                skipped += 1
                print(f"Skipping {' '.join(lineList[1:-1])} (id: {lineList[0]}) because no entries are in this category for the selected dataset")
            continue
        
        assert(lineList[-1] == "X"), f"Error: last entry in line {' '.join(lineList[1:-1])} (id: {lineList[0]}) is not 'X' or 'Â·', it is {lineList[-1]}"
        mappings[lineList[0]] = ' '.join(lineList[1:-1])
    
    if args.verbose:
        print(f"Skipped {skipped} of {len(lines)} entries")

    # Save to JSON
    with(open(args.outFilePath, "w")) as f:
        json.dump(mappings, f, indent=4)
