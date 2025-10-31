#!/usr/bin/env python3

import argparse
import json
import sys
from collections import defaultdict, Counter
from typing import Dict, List, Tuple


def load_data(input_path: str, min_word_count: int = 100) -> Tuple[List[Dict], List[Dict]]:
    """Load and filter data by minimum word count."""
    all_data = []
    filtered_data = []
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line.strip())
            all_data.append(obj)
            if obj['word_count'] >= min_word_count:
                filtered_data.append(obj)
    
    return all_data, filtered_data


def get_tokenizer_names(data: List[Dict]) -> List[str]:
    """Extract tokenizer names from data."""
    if not data:
        return []
    
    first_entry = data[0]
    tokenizers = sorted([
        k.replace('_token_count', '') 
        for k in first_entry.keys() 
        if k.endswith('_token_count')
    ])
    return tokenizers


def analyze_winners(data: List[Dict], tokenizers: List[str]) -> Dict:
    """Analyze which tokenizers win most often."""
    winner_counts = Counter()
    winner_details = defaultdict(list)
    
    for entry in data:
        best = entry['best_tokenizer']
        winner_counts[best] += 1
        winner_details[best].append({
            'lang': entry['lang'],
            'langname': entry['langname'],
            'word_count': entry['word_count'],
            'wpt': entry[f'{best}_wpt']
        })
    
    return {
        'counts': winner_counts,
        'details': winner_details
    }


def analyze_performance_stats(data: List[Dict], tokenizers: List[str]) -> Dict:
    """Calculate average WPT and CPT for each tokenizer."""
    stats = {}
    
    for tokenizer in tokenizers:
        wpt_values = [entry[f'{tokenizer}_wpt'] for entry in data]
        cpt_values = [entry[f'{tokenizer}_cpt'] for entry in data]
        token_counts = [entry[f'{tokenizer}_token_count'] for entry in data]
        
        stats[tokenizer] = {
            'avg_wpt': sum(wpt_values) / len(wpt_values),
            'avg_cpt': sum(cpt_values) / len(cpt_values),
            'avg_tokens': sum(token_counts) / len(token_counts),
            'min_wpt': min(wpt_values),
            'max_wpt': max(wpt_values),
        }
    
    return stats


def analyze_worst_performers(data: List[Dict], tokenizers: List[str]) -> Dict:
    """Find languages where each tokenizer performs worst."""
    worst_by_tokenizer = defaultdict(list)
    
    for entry in data:
        # Find worst tokenizer for this language (lowest WPT)
        wpt_scores = {tok: entry[f'{tok}_wpt'] for tok in tokenizers}
        worst_tok = min(wpt_scores, key=wpt_scores.get)
        
        worst_by_tokenizer[worst_tok].append({
            'lang': entry['lang'],
            'langname': entry['langname'],
            'word_count': entry['word_count'],
            'wpt': entry[f'{worst_tok}_wpt']
        })
    
    return worst_by_tokenizer


def analyze_language_groups(data: List[Dict]) -> Dict:
    """Categorize languages by linguistic family."""
    categories = {
        'Indo-Aryan': ['hi', 'bn', 'pa', 'gu', 'mr', 'ne', 'sa', 'bh', 'anp', 'awa', 'bho', 'mai', 'new', 'sat', 'ur', 'or', 'as'],
        'Dravidian': ['ta', 'te', 'kn', 'ml'],
        'Sino-Tibetan (Burmese)': ['my'],
        'Austroasiatic': ['km', 'vi'],
        'Tai-Kadai': ['th', 'lo', 'shn', 'tdd'],
        'Sinitic (Chinese)': ['zh', 'yue', 'lzh', 'nan', 'wuu', 'gan'],
        'Japonic': ['ja'],
        'Koreanic': ['ko'],
        'Semitic (Arabic)': ['ar', 'arz', 'ary'],
        'Semitic (Hebrew)': ['he', 'yi'],
        'Iranian': ['fa', 'azb', 'ckb', 'ps', 'ku'],
        'Slavic': ['ru', 'uk', 'pl', 'cs', 'sk', 'sr', 'hr', 'bg', 'be', 'mk'],
        'Romance': ['es', 'fr', 'it', 'pt', 'ro', 'ca', 'gl', 'oc', 'la'],
        'Germanic': ['en', 'de', 'nl', 'sv', 'da', 'no', 'nn', 'is', 'af', 'fy'],
        'Afro-Asiatic (non-Arabic)': ['am', 'ti', 'om', 'ha'],
        'Niger-Congo': ['yo', 'ig', 'sw', 'zu', 'xh', 'rw'],
    }
    
    group_stats = defaultdict(lambda: {'count': 0, 'total_words': 0, 'languages': []})
    uncategorized = []
    
    for entry in data:
        lang_code = entry['lang']
        categorized = False
        
        for category, codes in categories.items():
            if lang_code in codes:
                group_stats[category]['count'] += 1
                group_stats[category]['total_words'] += entry['word_count']
                group_stats[category]['languages'].append({
                    'lang': lang_code,
                    'langname': entry['langname'],
                    'word_count': entry['word_count'],
                    'best': entry['best_tokenizer']
                })
                categorized = True
                break
        
        if not categorized:
            uncategorized.append(entry)
    
    return {
        'groups': group_stats,
        'uncategorized': uncategorized
    }


def analyze_consistency(data: List[Dict], tokenizers: List[str]) -> Dict:
    """Analyze how consistent tokenizers are (variance in performance)."""
    import statistics
    
    consistency = {}
    
    for tokenizer in tokenizers:
        wpt_values = [entry[f'{tokenizer}_wpt'] for entry in data]
        
        consistency[tokenizer] = {
            'std_dev': statistics.stdev(wpt_values) if len(wpt_values) > 1 else 0,
            'variance': statistics.variance(wpt_values) if len(wpt_values) > 1 else 0,
        }
    
    return consistency


def print_report(all_data: List[Dict], filtered_data: List[Dict], min_word_count: int):
    """Generate and print the analysis report."""
    tokenizers = get_tokenizer_names(filtered_data)

    print("=" * 80)
    print("WIKICAT MULTILINGUAL TOKENIZER ANALYSIS")
    print("=" * 80)
    print("\nThis script analyzes tokenizer efficiency for Wikipedia 'cat' articles in 272 languages.")
    print("WPT = Words Per Token, CPT = Characters Per Token. Both measure efficiency: bigger is better.\n")
    print(f"Total languages: {len(all_data)}")
    print(f"Languages analyzed (>= {min_word_count} words): {len(filtered_data)}")
    print(f"Languages excluded: {len(all_data) - len(filtered_data)}")
    print(f"Tokenizers compared: {len(tokenizers)}\n")

    # 1. Winners Analysis
    print("=" * 80)
    print("1. TOKENIZER WINNERS (by frequency)")
    print("=" * 80)
    winners = analyze_winners(filtered_data, tokenizers)
    for tokenizer, count in winners['counts'].most_common():
        percentage = (count / len(filtered_data)) * 100
        print(f"\n{tokenizer}: {count} languages ({percentage:.1f}%)")
        print("  Top 3 largest languages where this tokenizer wins:")
        top_langs = sorted(winners['details'][tokenizer], key=lambda x: -x['word_count'])[:3]
        for lang in top_langs:
            print(f"    • {lang['langname']:30} ({lang['lang']:5}) - {lang['word_count']:6} words, WPT: {lang['wpt']:.3f}")

    # 2. Overall Performance Stats
    print("\n" + "=" * 80)
    print("2. AVERAGE PERFORMANCE (across all analyzed languages)")
    print("=" * 80)
    perf_stats = analyze_performance_stats(filtered_data, tokenizers)
    ranked_by_wpt = sorted(perf_stats.items(), key=lambda x: -x[1]['avg_wpt'])
    print(f"\n{'Tokenizer':<20} {'Avg WPT':<12} {'Avg CPT':<12} {'Avg Tokens':<12}")
    print("-" * 80)
    for tokenizer, stats in ranked_by_wpt:
        print(f"{tokenizer:<20} {stats['avg_wpt']:<12.3f} {stats['avg_cpt']:<12.3f} {stats['avg_tokens']:<12.1f}")

    # 3. Worst Performers
    print("\n" + "=" * 80)
    print("3. WORST PERFORMERS (languages where each tokenizer performs worst)")
    print("=" * 80)
    worst = analyze_worst_performers(filtered_data, tokenizers)
    worst_counts = [(tok, len(langs)) for tok, langs in worst.items()]
    worst_counts.sort(key=lambda x: -x[1])
    print("\nGPT-2 is expected to be the worst overall (trained on English only). The next-worst tokenizer by average WPT is also shown below.")
    # Find second-worst by average WPT (excluding GPT-2)
    non_gpt2 = [x for x in ranked_by_wpt if x[0] != 'GPT-2']
    second_worst = non_gpt2[-1][0] if non_gpt2 else None
    print(f"\nSecond-worst tokenizer by average WPT (excluding GPT-2): {second_worst}\n")
    for tokenizer, count in worst_counts[:5]:  # Show top 5 worst
        print(f"{tokenizer}: Worst in {count} languages (top 2 by word count shown)")
        examples = sorted(worst[tokenizer], key=lambda x: -x['word_count'])[:2]
        for ex in examples:
            print(f"    • {ex['langname']:30} ({ex['lang']:5}) - WPT: {ex['wpt']:.3f}")

    # 4. Language Group Analysis
    print("\n" + "=" * 80)
    print("4. PERFORMANCE BY LANGUAGE GROUP")
    print("=" * 80)
    group_analysis = analyze_language_groups(filtered_data)
    for group, stats in sorted(group_analysis['groups'].items(), key=lambda x: -x[1]['count']):
        if stats['count'] == 0:
            continue
        print(f"\n{group}: {stats['count']} languages ({stats['total_words']:,} total words)")
        best_in_group = Counter([lang['best'] for lang in stats['languages']])
        for tokenizer, count in best_in_group.most_common(3):
            pct = (count / stats['count']) * 100
            print(f"  Best: {tokenizer} ({count}/{stats['count']} = {pct:.1f}%)")

    print("\n" + "=" * 80)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 80)
    best_overall = ranked_by_wpt[0][0]
    most_wins = winners['counts'].most_common(1)[0][0]
    print(f"\n• Best Average WPT: {best_overall}")
    print(f"• Most Wins: {most_wins}")
    print(f"\n• For multilingual applications: Consider {most_wins} (wins most languages)")
    print(f"• For best average efficiency: Consider {best_overall} (highest avg WPT)")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze tokenizer performance statistics from WikiCat evaluation'
    )
    parser.add_argument(
        '--input', '-i',
        default='wikicat-tokenizer-eval.jsonl',
        help='Path to evaluation JSONL file'
    )
    parser.add_argument(
        '--min-words', '-m',
        type=int,
        default=100,
        help='Minimum word count to include language in analysis (default: 100)'
    )
    
    args = parser.parse_args()
    
    try:
        all_data, filtered_data = load_data(args.input, args.min_words)
        
        if not filtered_data:
            print(f"Error: No languages meet the minimum word count threshold of {args.min_words}", file=sys.stderr)
            sys.exit(1)
        
        print_report(all_data, filtered_data, args.min_words)
        
    except FileNotFoundError:
        print(f"Error: Input file '{args.input}' not found", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
