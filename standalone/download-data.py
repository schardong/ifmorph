#!/usr/bin/env python
# coding: utf-8

import argparse
import sys
import requests


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Downloads a tarball from Google Drive."
    )
    parser.add_argument("file_id", help="File ID provided by Google Drive.")
    parser.add_argument("output_file", help="Path to the downloaded file.")
    args = parser.parse_args()

    url = f"https://drive.google.com/uc?export=download&id={args.file_id}"
    response = requests.get(url, allow_redirects=True)

    if response.status_code == 200:
        # Write the content to the output file
        with open(args.output_file, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded file to {args.output_file}")
    else:
        print(
            f"Failed to download file: \"{response.status_code} - {response.text}\""
        )
        sys.exit(1)
