#!/usr/bin/env python3
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description='Convert binary file to hex file')
    parser.add_argument('input_file', help='Input binary file')
    parser.add_argument('output_file', help='Output hex file')
    parser.add_argument('-w', '--width', type=int, default=32, help='Width in bits (default: 32)')
    args = parser.parse_args()

    if args.width <= 0 or (args.width % 8) != 0:
        print("Error: --width must be a positive multiple of 8 bits", file=sys.stderr)
        sys.exit(2)

    try:
        with open(args.input_file, 'rb') as f_in, open(args.output_file, 'w') as f_out:
            width_bytes = args.width // 8
            
            while True:
                chunk = f_in.read(width_bytes)
                if not chunk:
                    break
                
                # Pad if necessary
                if len(chunk) < width_bytes:
                    chunk = chunk + b'\x00' * (width_bytes - len(chunk))
                
                # Convert to hex string, little endian (reverse bytes then hex) or big endian?
                # sim_main.cpp expects: "Hex string is big-endian: leftmost chars are MSB"
                # "words[3] = bits[127:96] ... words[0] = bits[31:0]"
                # "For WData (wide data), index [0] is LSB"
                # If input is binary (little endian usually for RISC-V instructions),
                # byte 0 is LSB of first word.
                # If we read bytes 0..15.
                # If we treat it as a large integer, it's little endian in memory.
                # sim_main wants strings.
                
                # Let's see sim_main again:
                # "words[3] = bits[127:96] ... words[0] = bits[31:0]"
                # "line[len-1-i]" is parsed.
                # It reads ASCII.
                # If I write "00112233..."
                # It treats last char (3) as nibble 0 (LSB).
                # So the string should be MSB...LSB (Big Endian hex string).
                
                # RISC-V binary is Little Endian.
                # Address 0: LSB of word 0.
                # So we read bytes: b0, b1, ..., b15
                # We want a string where "MSB" is printed first.
                # MSB is b15.
                # So we should print b15, b14, ..., b0.
                
                hex_str = ''.join(f"{b:02x}" for b in reversed(chunk))
                
                f_out.write(hex_str + '\n')
                
    except OSError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
