"""
Extended Analysis and Visualization Module
Complements the main coreference resolution system
"""

import json
from typing import List, Dict
import re

# ============================================================================
# COLORED TERMINAL OUTPUT VISUALIZATION
# ============================================================================

class TerminalVisualizer:
    """Generate colored terminal output for coreference chains."""
    
    COLORS = {
        1: '\033[94m',   # Blue
        2: '\033[92m',   # Green
        3: '\033[91m',   # Red
        4: '\033[93m',   # Yellow
        5: '\033[95m',   # Magenta
        6: '\033[96m',   # Cyan
        7: '\033[97m',   # White
        8: '\033[44m',   # Blue background
        9: '\033[42m',   # Green background
        10: '\033[41m',  # Red background
    }
    RESET = '\033[0m'
    
    @staticmethod
    def visualize(text: str, chains: List[Dict]) -> str:
        """Generate colored terminal visualization."""
        # Create mention to entity mapping
        mention_to_entity = {}
        for chain in chains:
            entity_id = chain["entity_id"]
            for mention in chain["mentions"]:
                mention_to_entity[mention.lower()] = entity_id
        
        # Process words
        words = text.split()
        output_lines = []
        current_line = ""
        
        for word in words:
            word_clean = word.rstrip('.,;:!?')
            punctuation = word[len(word_clean):]
            
            # Check if word matches a mention
            matched = False
            for mention_key, entity_id in mention_to_entity.items():
                if word_clean.lower() == mention_key:
                    color = TerminalVisualizer.COLORS.get(entity_id, '\033[94m')
                    current_line += f"{color}[{word_clean}]_{entity_id}{TerminalVisualizer.RESET}{punctuation} "
                    matched = True
                    break
            
            if not matched:
                current_line += word + " "
            
            # Line wrapping for readability
            if len(current_line) > 80:
                output_lines.append(current_line)
                current_line = ""
        
        if current_line:
            output_lines.append(current_line)
        
        return "\n".join(output_lines)

# ============================================================================
# DETAILED ERROR ANALYSIS
# ============================================================================

class ErrorAnalyzer:
    """Detailed analysis of coreference resolution errors."""
    
    @staticmethod
    def classify_error(gold_mention: str, system_chain: List[str], 
                      gold_chain: List[str]) -> str:
        """Classify the type of error made."""
        gold_set = set(m.lower() for m in gold_chain)
        system_set = set(m.lower() for m in system_chain)
        
        if gold_set == system_set:
            return "CORRECT"
        elif gold_set.issubset(system_set):
            return "FALSE_POSITIVE"  # System linked extra mentions
        elif gold_set.issuperset(system_set):
            return "FALSE_NEGATIVE"  # System missed mentions
        else:
            return "PARTIAL_MATCH"
    
    @staticmethod
    def analyze_mention_type(mention: str) -> str:
        """Classify mention type."""
        mention_lower = mention.lower()
        
        if mention_lower in ['he', 'she', 'it', 'they', 'him', 'her', 'them', 'both']:
            return "PRONOUN"
        elif mention[0].isupper() and len(mention.split()) == 1:
            return "PROPER_NOUN"
        elif mention.startswith('the '):
            return "DEFINITE_DESCRIPTION"
        elif mention.startswith('a '):
            return "INDEFINITE_DESCRIPTION"
        else:
            return "OTHER"
    
    @staticmethod
    def analyze_result(gold_chains: List[Dict], system_chains: List[Dict], 
                      text: str) -> Dict:
        """Comprehensive error analysis for a single document."""
        
        analysis = {
            "total_gold_entities": len(gold_chains),
            "total_system_entities": len(system_chains),
            "gold_mention_types": {},
            "error_types": {},
            "false_positives": [],
            "false_negatives": [],
            "partial_matches": []
        }
        
        # Count mention types in gold
        for chain in gold_chains:
            for mention in chain["mentions"]:
                m_type = ErrorAnalyzer.analyze_mention_type(mention)
                analysis["gold_mention_types"][m_type] = \
                    analysis["gold_mention_types"].get(m_type, 0) + 1
        
        # Create gold mention to chain mapping
        gold_mention_to_chain = {}
        for chain in gold_chains:
            for mention in chain["mentions"]:
                gold_mention_to_chain[mention.lower()] = chain["mentions"]
        
        # Analyze system output
        for sys_chain in system_chains:
            sys_mentions_lower = [m.lower() for m in sys_chain["mentions"]]
            
            # Find corresponding gold chain
            gold_chain_match = None
            for gold_mention in sys_mentions_lower:
                if gold_mention in gold_mention_to_chain:
                    gold_chain_match = gold_mention_to_chain[gold_mention]
                    break
            
            if gold_chain_match is None:
                # False positive: entire system chain is spurious
                analysis["false_positives"].append(sys_chain["mentions"])
            else:
                # Partial match or false positive/negative
                error_type = ErrorAnalyzer.classify_error(
                    sys_chain["mentions"][0],
                    sys_chain["mentions"],
                    gold_chain_match
                )
                analysis["error_types"][error_type] = \
                    analysis["error_types"].get(error_type, 0) + 1
                
                if error_type == "FALSE_POSITIVE":
                    analysis["false_positives"].append({
                        "system": sys_chain["mentions"],
                        "gold": gold_chain_match
                    })
                elif error_type == "PARTIAL_MATCH":
                    analysis["partial_matches"].append({
                        "system": sys_chain["mentions"],
                        "gold": gold_chain_match
                    })
        
        # Find false negatives (gold chains not covered)
        covered_gold = set()
        for sys_chain in system_chains:
            for mention in sys_chain["mentions"]:
                if mention.lower() in gold_mention_to_chain:
                    covered_gold.add(tuple(sorted([m.lower() for m in 
                                          gold_mention_to_chain[mention.lower()]])))
        
        for gold_chain in gold_chains:
            gold_key = tuple(sorted([m.lower() for m in gold_chain["mentions"]]))
            if gold_key not in covered_gold:
                analysis["false_negatives"].append(gold_chain["mentions"])
        
        return analysis

# ============================================================================
# AMBIGUITY CASE STUDY
# ============================================================================

AMBIGUITY_EXAMPLES = [
    {
        "case": "Pronoun Ambiguity",
        "text": "The lawyer called the client after he finished the report.",
        "pronouns": ["he"],
        "antecedents": ["The lawyer", "the client"],
        "linguistic_analysis": """
The pronoun 'he' has two syntactically plausible antecedents:
1. "The lawyer" (subject of main clause) - Preferred by Binding Theory
2. "the client" (object of main clause) - Possible but less likely

LINGUISTIC CUES FOR RESOLUTION:
- Syntactic c-command: The lawyer c-commands the client
- Recency: The client is closer, but subject has priority
- Semantic plausibility: Who can "finish a report"? Both are plausible.
- World knowledge: Typically lawyers write reports for clients.

CHALLENGES FOR ML SYSTEMS:
- Requires syntactic tree construction
- Needs semantic understanding of predicate-argument structure
- Depends on pragmatic/common-sense reasoning
""",
        "system_error": "Rule-based system lacks syntactic analysis and semantic reasoning"
    },
    {
        "case": "Nested Entity Ambiguity",
        "text": "Apple's CEO announced their new product.",
        "entities": ["Apple", "Apple's CEO"],
        "pronouns": ["their"],
        "linguistic_analysis": """
The possessive pronoun 'their' is ambiguous:
1. Their = Apple (organization, plural interpretation)
2. Their = Apple's CEO (person, but using plural form)

LINGUISTIC CUES FOR RESOLUTION:
- Entity type mismatch: "their" expects plural, CEO is singular
- Possessive structure: "Apple's CEO" indicates person subsumed under org
- Semantic selection: Products belong to organizations primarily
- Predicate structure: "announced" fits both interpretations

CHALLENGES FOR ML SYSTEMS:
- Requires entity type classification
- Needs handling of possessive attachment
- Understanding of implicit entity relationships
- Generic plural pronoun usage
""",
        "system_error": "System fails to handle possessive constructions and entity types"
    },
    {
        "case": "Bridging Reference",
        "text": "The company released a new phone. The device featured an advanced camera.",
        "bridging_type": "whole-part",
        "linguistic_analysis": """
"The device" is a bridging reference to "a new phone" - they are not identical
mentions but related through semantic knowledge (device part of phone).

LINGUISTIC CUES FOR RESOLUTION:
- Semantic relation: phone ⊃ device (part-of relationship)
- Discourse context: Expected to discuss phone properties
- Lexical chains: "phone" and "device" are related
- Article use: "The device" presupposes familiarity from context

CHALLENGES FOR ML SYSTEMS:
- Requires semantic/knowledge graph understanding
- Bridging references are subtle and context-dependent
- Not taught explicitly in many coreference datasets
- Depends on world knowledge (what parts do phones have?)
""",
        "system_error": "Rule-based approach treats only identical mentions as coreferent"
    }
]

# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_ambiguity_analysis() -> str:
    """Generate detailed ambiguity analysis for report."""
    output = []
    output.append("\n" + "="*80)
    output.append("DETAILED AMBIGUITY AND ERROR ANALYSIS")
    output.append("="*80)
    
    for example in AMBIGUITY_EXAMPLES:
        output.append(f"\nCASE: {example['case']}")
        output.append("-" * 60)
        output.append(f"Text: {example['text']}")
        output.append(f"\nLinguistic Analysis:\n{example['linguistic_analysis']}")
        output.append(f"System Error: {example['system_error']}")
    
    return "\n".join(output)

def generate_results_table(results: Dict) -> str:
    """Generate formatted results table."""
    output = []
    output.append("\n" + "="*80)
    output.append("EVALUATION RESULTS SUMMARY")
    output.append("="*80)
    
    muc = results['muc_average']
    ceaf = results['ceaf_average']
    
    output.append(f"\n{'Metric':<15} | {'Precision':<12} | {'Recall':<12} | {'F1':<12}")
    output.append("-" * 55)
    output.append(f"{'MUC':<15} | {muc['precision']:<12.4f} | {muc['recall']:<12.4f} | {muc['f1']:<12.4f}")
    output.append(f"{'CEAF-e':<15} | {ceaf['precision']:<12.4f} | {ceaf['recall']:<12.4f} | {ceaf['f1']:<12.4f}")
    
    return "\n".join(output)

def generate_mention_error_statistics(all_results: List[Dict]) -> str:
    """Analyze error patterns across documents."""
    output = []
    
    mention_type_performance = {
        "PRONOUN": {"correct": 0, "total": 0},
        "PROPER_NOUN": {"correct": 0, "total": 0},
        "DEFINITE_DESCRIPTION": {"correct": 0, "total": 0},
        "INDEFINITE_DESCRIPTION": {"correct": 0, "total": 0},
    }
    
    for doc_result in all_results:
        gold = doc_result["gold"]
        system = doc_result["system"]
        
        for gold_chain in gold:
            for mention in gold_chain["mentions"]:
                m_type = ErrorAnalyzer.analyze_mention_type(mention)
                mention_type_performance[m_type]["total"] += 1
                
                # Check if system got it right
                for sys_chain in system:
                    if any(m.lower() == mention.lower() for m in sys_chain["mentions"]):
                        mention_type_performance[m_type]["correct"] += 1
                        break
    
    output.append("\n" + "="*80)
    output.append("MENTION TYPE PERFORMANCE ANALYSIS")
    output.append("="*80)
    output.append(f"\n{'Mention Type':<25} | {'Correct':<10} | {'Total':<10} | {'Accuracy':<10}")
    output.append("-" * 60)
    
    for m_type, stats in mention_type_performance.items():
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        output.append(f"{m_type:<25} | {stats['correct']:<10} | {stats['total']:<10} | {acc:<10.2%}")
    
    return "\n".join(output)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print(generate_ambiguity_analysis())
    print("\n\nTo use this module with your results:")
    print("1. Load evaluation results from coref_evaluation_results.json")
    print("2. Call generate_results_table(results) for summary")
    print("3. Call generate_mention_error_statistics(results['per_document']) for error analysis")
    print("4. Use TerminalVisualizer.visualize(text, chains) for color-coded output")
    print("5. Use ErrorAnalyzer.analyze_result() for detailed per-document error analysis")