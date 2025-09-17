#!/usr/bin/env python3
"""
Interactive demonstration of synthetic data generation and usage
"""

import sys
sys.path.append('.')

import csv
import random
import pandas as pd
from pathlib import Path
from collections import Counter
import json

# Import the generation functions
from scripts.generate_synth_data import (
    EMPLOYEES, DATE_PHRASES, ABSENCE_TPL, ABSENCE_ACT_TPL, 
    TDO_TPL, COST_TPL, TOPICS, OOS,
    inject_typos, code_switch, add_noise, gen
)

def demonstrate_template_system():
    """Show how the template system works"""
    
    print("=" * 80)
    print("SYNTHETIC DATA TEMPLATE SYSTEM DEMONSTRATION")
    print("=" * 80)
    
    print("\n1. TEMPLATE CATEGORIES")
    print("-" * 50)
    
    print(f"Absence Query Templates ({len(ABSENCE_TPL)}):")
    for i, template in enumerate(ABSENCE_TPL[:5], 1):
        print(f"  {i}. {template}")
    print(f"  ... and {len(ABSENCE_TPL)-5} more")
    
    print(f"\nAbsence Action Templates ({len(ABSENCE_ACT_TPL)}):")
    for i, template in enumerate(ABSENCE_ACT_TPL, 1):
        print(f"  {i}. {template}")
    
    print(f"\nTDO Templates ({len(TDO_TPL)}):")
    for i, template in enumerate(TDO_TPL, 1):
        print(f"  {i}. {template}")
    
    print(f"\nCost Templates ({len(COST_TPL)}):")
    for i, template in enumerate(COST_TPL[:5], 1):
        print(f"  {i}. {template}")
    print(f"  ... and {len(COST_TPL)-5} more")
    
    print("\n2. VARIABLE LISTS")
    print("-" * 50)
    
    print(f"Employee Names ({len(EMPLOYEES)}):")
    print(f"  {', '.join(EMPLOYEES[:10])}, ... and {len(EMPLOYEES)-10} more")
    
    print(f"\nDate Phrases ({len(DATE_PHRASES)}):")
    for i, date in enumerate(DATE_PHRASES[:8], 1):
        print(f"  {i}. {date}")
    print(f"  ... and {len(DATE_PHRASES)-8} more")
    
    print(f"\nTechnical Topics ({len(TOPICS)}):")
    for i, topic in enumerate(TOPICS[:8], 1):
        print(f"  {i}. {topic}")
    print(f"  ... and {len(TOPICS)-8} more")
    
    print(f"\nOut-of-Scope Examples ({len(OOS)}):")
    for i, example in enumerate(OOS[:8], 1):
        print(f"  {i}. {example}")
    print(f"  ... and {len(OOS)-8} more")

def demonstrate_base_generation():
    """Show how base examples are generated from templates"""
    
    print("\n\n" + "=" * 80)
    print("BASE EXAMPLE GENERATION DEMONSTRATION")
    print("=" * 80)
    
    print("\n3. COMBINATORIAL GENERATION")
    print("-" * 50)
    
    # Show absence query generation
    print("Absence Query Examples (template × date combinations):")
    template = "who was absent {when}"
    dates = DATE_PHRASES[:5]
    
    print(f"\nTemplate: '{template}'")
    print("Combined with dates:")
    for i, date in enumerate(dates, 1):
        result = template.format(when=date)
        print(f"  {i}. '{result}'")
    
    print(f"\nTotal combinations: {len(ABSENCE_TPL)} templates × {len(DATE_PHRASES)} dates = {len(ABSENCE_TPL) * len(DATE_PHRASES)} examples")
    
    # Show absence action generation
    print(f"\nAbsence Action Examples (template × name × date combinations):")
    template = "mark {name} absent {when}"
    names = EMPLOYEES[:3]
    action_dates = ["today", "tomorrow", "on 2025-09-01"]
    
    print(f"\nTemplate: '{template}'")
    print("Combined with names and dates:")
    for i, name in enumerate(names, 1):
        for j, date in enumerate(action_dates, 1):
            result = template.format(name=name, when=date)
            print(f"  {i}.{j}. '{result}'")
    
    print(f"\nTotal combinations: {len(ABSENCE_ACT_TPL)} templates × {len(EMPLOYEES)} names × 3 dates = {len(ABSENCE_ACT_TPL) * len(EMPLOYEES) * 3} examples")
    
    # Show TDO generation
    print(f"\nTDO Examples (template × topic combinations):")
    template = "draft a TDO for {topic}"
    topics = TOPICS[:4]
    
    print(f"\nTemplate: '{template}'")
    print("Combined with topics:")
    for i, topic in enumerate(topics, 1):
        result = template.format(topic=topic)
        print(f"  {i}. '{result}'")
    
    print(f"\nTotal combinations: {len(TDO_TPL)} templates × {len(TOPICS)} topics = {len(TDO_TPL) * len(TOPICS)} examples")

def demonstrate_noise_injection():
    """Show how noise is added to make examples realistic"""
    
    print("\n\n" + "=" * 80)
    print("NOISE INJECTION DEMONSTRATION")
    print("=" * 80)
    
    print("\n4. NOISE INJECTION TECHNIQUES")
    print("-" * 50)
    
    base_examples = [
        "who was absent last week",
        "draft a TDO for invoice microservice", 
        "estimate cost for data pipeline",
        "show absences yesterday"
    ]
    
    for example in base_examples:
        print(f"\nBase: '{example}'")
        
        # Show typo injection
        typo_version = inject_typos(example)
        print(f"Typos: '{typo_version}'")
        
        # Show code switching
        code_version = code_switch(example)
        print(f"Code-switch: '{code_version}'")
        
        # Show noise addition
        noise_version = add_noise(example)
        print(f"Noise: '{noise_version}'")
        
        # Show combined
        combined = add_noise(code_switch(inject_typos(example)))
        print(f"Combined: '{combined}'")

def demonstrate_realistic_variations():
    """Show realistic variations that would be generated"""
    
    print("\n\n" + "=" * 80)
    print("REALISTIC VARIATION DEMONSTRATION")
    print("=" * 80)
    
    print("\n5. REALISTIC VARIATIONS")
    print("-" * 50)
    
    # Set seed for reproducible demo
    random.seed(42)
    
    base_text = "who was absent last week"
    print(f"Base text: '{base_text}'")
    print("\nGenerated variations:")
    
    for i in range(10):
        # Generate variation like the real system does
        variation = base_text
        if random.random() < 0.5:
            variation = inject_typos(variation)
        if random.random() < 0.5:
            variation = code_switch(variation)
        if random.random() < 0.8:
            variation = add_noise(variation)
        
        print(f"  {i+1:2d}. '{variation}'")
    
    # Reset seed
    random.seed()

def analyze_actual_dataset():
    """Analyze the actual generated dataset"""
    
    print("\n\n" + "=" * 80)
    print("ACTUAL DATASET ANALYSIS")
    print("=" * 80)
    
    # Check if dataset exists
    dataset_path = Path("data/synth_dataset.csv")
    if not dataset_path.exists():
        print("Dataset not found. Generating now...")
        rows = gen()
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dataset_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["text", "label"])
            w.writerows(rows)
        print(f"Generated {len(rows)} examples")
    
    # Load and analyze
    df = pd.read_csv(dataset_path)
    
    print("\n6. DATASET STATISTICS")
    print("-" * 50)
    
    print(f"Total examples: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    
    # Class distribution
    print(f"\nClass distribution:")
    class_counts = df['label'].value_counts()
    for label, count in class_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {label}: {count} ({percentage:.1f}%)")
    
    # Sample examples from each class
    print(f"\nSample examples from each class:")
    for label in class_counts.index:
        samples = df[df['label'] == label]['text'].sample(3, random_state=42)
        print(f"\n{label.upper()}:")
        for i, sample in enumerate(samples, 1):
            print(f"  {i}. '{sample}'")

def demonstrate_training_impact():
    """Show how synthetic data impacts model training"""
    
    print("\n\n" + "=" * 80)
    print("TRAINING IMPACT DEMONSTRATION")
    print("=" * 80)
    
    print("\n7. TRAINING PIPELINE SIMULATION")
    print("-" * 50)
    
    # Load dataset
    df = pd.read_csv("data/synth_dataset.csv")
    
    # Simulate train/test split
    from sklearn.model_selection import train_test_split
    
    X = df["text"].tolist()
    y = df["label"].tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training examples: {len(X_train)}")
    print(f"Test examples: {len(X_test)}")
    
    # Show training data characteristics
    train_labels = Counter(y_train)
    test_labels = Counter(y_test)
    
    print(f"\nTraining set distribution:")
    for label, count in train_labels.items():
        print(f"  {label}: {count}")
    
    print(f"\nTest set distribution:")
    for label, count in test_labels.items():
        print(f"  {label}: {count}")
    
    # Show examples of challenging cases
    print(f"\nChallenging examples (with noise):")
    noisy_examples = [x for x in X_train if any(noise in x.lower() for noise in ['yo', 'pls', 'bro', 'hey', '🚀', '🙏'])]
    
    for i, example in enumerate(noisy_examples[:5], 1):
        label = y_train[X_train.index(example)]
        print(f"  {i}. '{example}' → {label}")
    
    # Show typo examples
    print(f"\nTypo examples:")
    # Simple heuristic to find likely typos
    typo_examples = []
    for x, y_label in zip(X_train, y_train):
        if any(len(word) > 4 and word.count(word[i:i+2]) > 1 for word in x.split() for i in range(len(word)-1)):
            typo_examples.append((x, y_label))
            if len(typo_examples) >= 5:
                break
    
    for i, (example, label) in enumerate(typo_examples, 1):
        print(f"  {i}. '{example}' → {label}")

def demonstrate_adversarial_examples():
    """Show adversarial examples designed to fool the system"""
    
    print("\n\n" + "=" * 80)
    print("ADVERSARIAL EXAMPLES DEMONSTRATION")
    print("=" * 80)
    
    print("\n8. ADVERSARIAL/TRICKY EXAMPLES")
    print("-" * 50)
    
    adversarial_groups = {
        "Absence Decoys (contain 'leave' but not about employee absence)": [
            "leave the page",
            "left join sql example", 
            "how to leave a review",
            "maternity leave policy pdf",
            "absent minded professor movie"
        ],
        "Cost Decoys (contain 'cost' but not about estimation)": [
            "costco membership price",
            "costume ideas for halloween",
            "cost of living in NYC", 
            "absorption cost accounting"
        ],
        "TDO Decoys (sound like 'TDO' but different)": [
            "todo list for tomorrow",
            "to-do checklist app",
            "what does TDO stand for"
        ]
    }
    
    for group_name, examples in adversarial_groups.items():
        print(f"\n{group_name}:")
        for i, example in enumerate(examples, 1):
            print(f"  {i}. '{example}' → OUT_OF_SCOPE")
        
        print(f"  Why tricky: These contain keywords that might match wrong categories")
        print(f"  System learns: Context matters, not just individual words")

def main():
    """Run all demonstrations"""
    
    print("COMPLETE SYNTHETIC DATASET DEMONSTRATION")
    print("This will show you exactly how synthetic training data is generated and used.")
    print("\nPress Enter to start...")
    input()
    
    try:
        demonstrate_template_system()
        input("\nPress Enter to continue to base generation...")
        
        demonstrate_base_generation()
        input("\nPress Enter to continue to noise injection...")
        
        demonstrate_noise_injection()
        input("\nPress Enter to continue to realistic variations...")
        
        demonstrate_realistic_variations()
        input("\nPress Enter to continue to dataset analysis...")
        
        analyze_actual_dataset()
        input("\nPress Enter to continue to training impact...")
        
        demonstrate_training_impact()
        input("\nPress Enter to continue to adversarial examples...")
        
        demonstrate_adversarial_examples()
        
        print("\n" + "=" * 80)
        print("DEMONSTRATION COMPLETE!")
        print("=" * 80)
        print("\nYou've seen:")
        print("1. How templates and variables create base examples")
        print("2. How combinatorial explosion generates thousands of examples")
        print("3. How noise injection makes examples realistic")
        print("4. How the actual dataset looks and its statistics")
        print("5. How this data is used for model training")
        print("6. How adversarial examples prevent false positives")
        print("\nKey insight: Smart synthetic data generation solves the")
        print("cold-start problem, enabling high-accuracy ML from day one!")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()