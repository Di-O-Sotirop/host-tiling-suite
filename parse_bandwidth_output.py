import csv
import re
import argparse

def parse_bandwidth_output(input_txt):
    """
    Parse bandwidth benchmark output from a .txt file.
    Extract (size in KB, bandwidth in MB/s) pairs.
    """
    size_bw_pattern = re.compile(r"Size:\s+(\d+)\s+KB,\s+BW:\s+([\d.]+)\s+MB/s")
    data = []

    with open(input_txt, 'r') as f:
        for line in f:
            match = size_bw_pattern.search(line)
            if match:
                size_kb = int(match.group(1))
                bandwidth = float(match.group(2))
                data.append((size_kb, bandwidth))

    return data

def write_csv(data, output_csv):
    """
    Write (size_kb, bandwidth) pairs to a CSV file.
    """
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Transfer Size (KB)", "Bandwidth (MB/s)"])
        writer.writerows(data)

def main():
    parser = argparse.ArgumentParser(description="Parse bandwidth benchmark output to CSV.")
    parser.add_argument("input_txt", help="Path to the benchmark output text file")
    parser.add_argument("output_csv", help="Path to the output CSV file")
    args = parser.parse_args()

    data = parse_bandwidth_output(args.input_txt)
    if not data:
        print("[WARNING] No data found in the input file.")
    else:
        write_csv(data, args.output_csv)
        print(f"[INFO] Parsed {len(data)} entries. CSV saved to: {args.output_csv}")

if __name__ == "__main__":
    main()
