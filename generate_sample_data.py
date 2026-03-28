"""
generate_sample_data.py
-----------------------
Creates a small synthetic headlines.csv for local development and testing.

Usage
-----
    python generate_sample_data.py            # writes headlines.csv
    python generate_sample_data.py --out my_data.csv
"""

import argparse
import random

import pandas as pd

HEADLINES = [
    # Technology
    "New smartphone model breaks all sales records",
    "Tech giant unveils revolutionary AI assistant",
    "Startup raises millions for quantum computing breakthrough",
    "Social media platform announces major privacy overhaul",
    "Self-driving car completes cross-country journey",
    "Open-source software community celebrates milestone",
    "Cloud provider suffers massive outage affecting millions",
    "Silicon Valley sees surge in robotics investment",
    "Cybersecurity firm warns of new ransomware strain",
    "Next-generation chip doubles computing performance",
    # Sports
    "Local team wins championship after decade-long drought",
    "Star athlete signs record-breaking contract extension",
    "Olympic committee announces new sports for upcoming games",
    "Marathon runner sets new world record in desert heat",
    "Football league expands with two new franchise teams",
    "Tennis player beats top seed in stunning upset",
    "Basketball superstar announces retirement after 20 seasons",
    "Soccer World Cup host city finalises stadium plans",
    "Swimmer breaks own world record at national championships",
    "Rugby union team clinches Grand Slam title",
    # Politics
    "Government announces sweeping tax reform package",
    "Senate votes to confirm new cabinet secretary",
    "International summit focuses on climate change commitments",
    "Prime minister calls early election amid economic uncertainty",
    "Opposition party unveils ambitious healthcare plan",
    "Diplomat visits region amid rising border tensions",
    "New trade agreement expected to boost exports significantly",
    "City mayor proposes major infrastructure spending bill",
    "Foreign minister meets counterpart to discuss peace talks",
    "Parliament debates controversial immigration legislation",
    # Health
    "Clinical trial shows promising results for new cancer therapy",
    "Health officials urge vaccination ahead of flu season",
    "Researchers discover new link between diet and longevity",
    "Hospital system adopts AI for early disease detection",
    "Mental health awareness campaign reaches record participation",
    "Pharmaceutical company recalls batch of blood pressure drug",
    "Study links air pollution to increased dementia risk",
    "Surge in telemedicine use reshapes primary care landscape",
    "Scientists develop bandage that accelerates wound healing",
    "New guidelines recommend earlier screening for colorectal cancer",
    # Science
    "Astronomers detect water vapour on distant exoplanet",
    "Biologists discover new species in deep-sea expedition",
    "Physicists confirm long-sought particle at particle collider",
    "Mars rover uncovers evidence of ancient river system",
    "Gene-editing technique shows promise for hereditary diseases",
    "Paleontologists unearth largest dinosaur skeleton ever found",
    "Climate researchers warn of tipping point for Arctic ice",
    "Chemists synthesise new material with remarkable properties",
    "Neuroscientists map previously unknown brain network",
    "Volcanic eruption creates new island in Pacific Ocean",
]


def generate(n_samples: int, seed: int) -> pd.DataFrame:
    random.seed(seed)
    headlines = [random.choice(HEADLINES) for _ in range(n_samples)]
    return pd.DataFrame({"headline": headlines})


def main():
    parser = argparse.ArgumentParser(description="Generate sample headlines CSV.")
    parser.add_argument("--out", default="headlines.csv", help="Output CSV path.")
    parser.add_argument("--n", type=int, default=200, help="Number of rows to generate.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    args = parser.parse_args()

    df = generate(args.n, args.seed)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows to {args.out}")


if __name__ == "__main__":
    main()
