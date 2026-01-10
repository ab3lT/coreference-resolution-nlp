"""
Coreference Resolution System for Narrative Text
Master's Level NLP Project
"""

import json
import re
from typing import List, Dict, Tuple, Set
from collections import defaultdict
import math

# ============================================================================
# 1. GOLD STANDARD ANNOTATIONS
# ============================================================================

GOLD_ANNOTATIONS = [
    {
        "id": 1,
        "text": "Sarah bought a new car. She drove it to work. The vehicle performed excellently.",
        "entities": [
            {"entity_id": 1, "mentions": ["Sarah", "She"]},
            {"entity_id": 2, "mentions": ["a new car", "it", "The vehicle"]}
        ]
    },
    {
        "id": 2,
        "text": "John met Mary at the coffee shop. He ordered tea while she got a cappuccino. Both enjoyed their drinks.",
        "entities": [
            {"entity_id": 1, "mentions": ["John", "He"]},
            {"entity_id": 2, "mentions": ["Mary", "she"]},
            {"entity_id": 3, "mentions": ["the coffee shop"]},
            {"entity_id": 4, "mentions": ["tea"]},
            {"entity_id": 5, "mentions": ["a cappuccino"]},
            {"entity_id": 6, "mentions": ["Both", "their drinks"]}
        ]
    },
    {
        "id": 3,
        "text": "Apple announced its new iPhone. The device features an advanced camera. Tim Cook presented it at the conference.",
        "entities": [
            {"entity_id": 1, "mentions": ["Apple"]},
            {"entity_id": 2, "mentions": ["its new iPhone", "The device", "it"]},
            {"entity_id": 3, "mentions": ["Tim Cook"]}
        ]
    },
    {
        "id": 4,
        "text": "The lawyer called the client after he finished the report. It was crucial for the case.",
        "entities": [
            {"entity_id": 1, "mentions": ["The lawyer", "he"]},
            {"entity_id": 2, "mentions": ["the client"]},
            {"entity_id": 3, "mentions": ["the report", "It"]}
        ]
    },
    {
        "id": 5,
        "text": "Emma and her brother visited the museum. They spent hours looking at paintings. The artworks fascinated them.",
        "entities": [
            {"entity_id": 1, "mentions": ["Emma"]},
            {"entity_id": 2, "mentions": ["her brother"]},
            {"entity_id": 3, "mentions": ["the museum"]},
            {"entity_id": 4, "mentions": ["They", "them"]},
            {"entity_id": 5, "mentions": ["paintings", "The artworks"]}
        ]
    },
    {
        "id": 6,
        "text": "The company released a new product. It was designed for mobile users. Customers loved the innovation.",
        "entities": [
            {"entity_id": 1, "mentions": ["The company"]},
            {"entity_id": 2, "mentions": ["a new product", "It"]},
            {"entity_id": 3, "mentions": ["mobile users"]},
            {"entity_id": 4, "mentions": ["Customers"]},
            {"entity_id": 5, "mentions": ["the innovation"]}
        ]
    },
    {
        "id": 7,
        "text": "Peter gave his book to Michael. He read it quickly. The story impressed him.",
        "entities": [
            {"entity_id": 1, "mentions": ["Peter", "He"]},
            {"entity_id": 2, "mentions": ["his book", "it", "The story"]},
            {"entity_id": 3, "mentions": ["Michael", "him"]}
        ]
    },
    {
        "id": 8,
        "text": "The hospital admitted a patient yesterday. She required immediate treatment. The doctors examined her carefully.",
        "entities": [
            {"entity_id": 1, "mentions": ["The hospital"]},
            {"entity_id": 2, "mentions": ["a patient", "She", "her"]},
            {"entity_id": 3, "mentions": ["The doctors"]}
        ]
    },
    {
        "id": 9,
        "text": "Google acquired a startup. Its founders were thrilled. They received stock options and cash.",
        "entities": [
            {"entity_id": 1, "mentions": ["Google"]},
            {"entity_id": 2, "mentions": ["a startup"]},
            {"entity_id": 3, "mentions": ["Its founders", "They"]},
            {"entity_id": 4, "mentions": ["stock options and cash"]}
        ]
    },
    {
        "id": 10,
        "text": "The professor explained the theory. Students listened attentively. It was complex but fascinating.",
        "entities": [
            {"entity_id": 1, "mentions": ["The professor"]},
            {"entity_id": 2, "mentions": ["the theory", "It"]},
            {"entity_id": 3, "mentions": ["Students"]}
        ]
    }
]

# ============================================================================
# 2. SIMPLE RULE-BASED COREFERENCE RESOLUTION
# ============================================================================

class CoreferenceResolver:
    """
    Rule-based coreference resolution system.
    For demonstration, uses heuristics combined with mention extraction.
    """
    
    def __init__(self):
        self.pronouns = {
            'he': 'male', 'him': 'male', 'his': 'male',
            'she': 'female', 'her': 'female', 'hers': 'female',
            'it': 'neuter', 'its': 'neuter',
            'they': 'plural', 'them': 'plural', 'their': 'plural',
            'both': 'plural'
        }
    
    def extract_mentions(self, text: str) -> List[Tuple[str, int, int]]:
        """Extract mention candidates (nouns, pronouns, proper nouns)."""
        mentions = []
        
        # Extract pronouns
        for pronoun in self.pronouns.keys():
            pattern = r'\b' + pronoun + r'\b'
            for match in re.finditer(pattern, text, re.IGNORECASE):
                mentions.append((pronoun.lower(), match.start(), match.end()))
        
        # Extract noun phrases (simplified)
        noun_patterns = [
            r'\b[A-Z][a-z]+\b',  # Proper nouns
            r'\bthe\s+(?:[\w]+\s+)*[\w]+\b',  # Definite descriptions
            r'\ba\s+(?:[\w]+\s+)*[\w]+\b',  # Indefinite descriptions
        ]
        
        for pattern in noun_patterns:
            for match in re.finditer(pattern, text):
                mentions.append((match.group(), match.start(), match.end()))
        
        return mentions
    
    def resolve(self, text: str) -> List[Dict]:
        """Resolve coreferences using simple heuristics."""
        mentions = self.extract_mentions(text)
        chains = []
        processed = set()
        
        for i, (mention, start, end) in enumerate(mentions):
            if i in processed:
                continue
            
            chain = [mention]
            processed.add(i)
            
            # Find matching pronouns or similar mentions
            for j, (other_mention, other_start, other_end) in enumerate(mentions[i+1:], start=i+1):
                if j in processed:
                    continue
                
                # Simple matching heuristic
                if self._should_link(mention, other_mention):
                    chain.append(other_mention)
                    processed.add(j)
            
            if len(chain) > 0:
                chains.append({
                    "entity_id": len(chains) + 1,
                    "mentions": chain
                })
        
        return chains
    
    def _should_link(self, mention1: str, mention2: str) -> bool:
        """Determine if two mentions should be linked."""
        m1_lower = mention1.lower()
        m2_lower = mention2.lower()
        
        # Same mention
        if m1_lower == m2_lower:
            return True
        
        # Pronoun resolution heuristic
        if m2_lower in self.pronouns:
            return False  # Simplified: skip complex pronoun resolution
        
        return False

# ============================================================================
# 3. EVALUATION METRICS: MUC
# ============================================================================

class MUCMetric:
    """
    Muelas Unified Clustering (MUC) Metric
    Link-based evaluation: counts links formed/missing
    """
    
    @staticmethod
    def get_links(chains: List[Dict]) -> Set[Tuple[int, int]]:
        """
        Extract all links from coreference chains.
        A link is a pair of mentions in the same chain.
        """
        links = set()
        for chain in chains:
            mentions = chain.get("mentions", [])
            if len(mentions) < 2:
                continue
            for i in range(len(mentions) - 1):
                link = tuple(sorted([i, i + 1]))
                links.add(link)
        return links
    
    @staticmethod
    def compute(gold_chains: List[Dict], system_chains: List[Dict]) -> Dict:
        """Compute MUC precision, recall, and F1."""
        gold_links = MUCMetric.get_links(gold_chains)
        system_links = MUCMetric.get_links(system_chains)
        
        if len(system_links) == 0:
            precision = 0.0
        else:
            precision = len(gold_links & system_links) / len(system_links)
        
        if len(gold_links) == 0:
            recall = 1.0 if len(system_links) == 0 else 0.0
        else:
            recall = len(gold_links & system_links) / len(gold_links)
        
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4)
        }

# ============================================================================
# 4. EVALUATION METRICS: CEAF
# ============================================================================

class CEAFMetric:
    """
    CEAF-e (Entity-based CEAF)
    Optimal alignment between system and gold entities using mention overlap
    """
    
    @staticmethod
    def mention_overlap(gold_mentions: List[str], system_mentions: List[str]) -> float:
        """Compute similarity as number of overlapping mentions."""
        gold_set = set(m.lower() for m in gold_mentions)
        system_set = set(m.lower() for m in system_mentions)
        overlap = len(gold_set & system_set)
        return overlap
    
    @staticmethod
    def optimal_alignment(gold_chains: List[Dict], system_chains: List[Dict]) -> Tuple[float, List]:
        """
        Find optimal alignment using greedy matching.
        Returns (total_similarity, alignment_list)
        """
        if not gold_chains or not system_chains:
            return 0.0, []
        
        gold_entities = [(i, chain["mentions"]) for i, chain in enumerate(gold_chains)]
        system_entities = [(i, chain["mentions"]) for i, chain in enumerate(system_chains)]
        
        matched_gold = set()
        matched_system = set()
        alignment = []
        total_similarity = 0.0
        
        # Greedy matching: highest overlap first
        similarities = []
        for g_idx, (g_id, g_mentions) in enumerate(gold_entities):
            for s_idx, (s_id, s_mentions) in enumerate(system_entities):
                sim = CEAFMetric.mention_overlap(g_mentions, s_mentions)
                similarities.append((sim, g_idx, s_idx))
        
        similarities.sort(reverse=True)
        
        for sim, g_idx, s_idx in similarities:
            if g_idx not in matched_gold and s_idx not in matched_system and sim > 0:
                matched_gold.add(g_idx)
                matched_system.add(s_idx)
                alignment.append((g_idx, s_idx, sim))
                total_similarity += sim
        
        return total_similarity, alignment
    
    @staticmethod
    def compute(gold_chains: List[Dict], system_chains: List[Dict]) -> Dict:
        """Compute CEAF-e precision, recall, and F1."""
        total_sim, _ = CEAFMetric.optimal_alignment(gold_chains, system_chains)
        
        # Precision: total similarity / sum of system entity sizes
        system_size = sum(len(chain["mentions"]) for chain in system_chains)
        precision = total_sim / system_size if system_size > 0 else 0.0
        
        # Recall: total similarity / sum of gold entity sizes
        gold_size = sum(len(chain["mentions"]) for chain in gold_chains)
        recall = total_sim / gold_size if gold_size > 0 else 0.0
        
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4)
        }

# ============================================================================
# 5. EVALUATION RUNNER
# ============================================================================

def evaluate_system(gold_annotations: List[Dict]) -> Dict:
    """Evaluate the coreference resolver on gold annotations."""
    resolver = CoreferenceResolver()
    
    all_muc_scores = {"precision": [], "recall": [], "f1": []}
    all_ceaf_scores = {"precision": [], "recall": [], "f1": []}
    
    results_per_document = []
    
    for doc in gold_annotations:
        text = doc["text"]
        gold_chains = doc["entities"]
        system_chains = resolver.resolve(text)
        
        muc_result = MUCMetric.compute(gold_chains, system_chains)
        ceaf_result = CEAFMetric.compute(gold_chains, system_chains)
        
        for key in all_muc_scores:
            all_muc_scores[key].append(muc_result[key])
        for key in all_ceaf_scores:
            all_ceaf_scores[key].append(ceaf_result[key])
        
        results_per_document.append({
            "doc_id": doc["id"],
            "text": text,
            "gold": gold_chains,
            "system": system_chains,
            "muc": muc_result,
            "ceaf": ceaf_result
        })
    
    # Compute macro averages
    avg_muc = {k: round(sum(v) / len(v), 4) for k, v in all_muc_scores.items()}
    avg_ceaf = {k: round(sum(v) / len(v), 4) for k, v in all_ceaf_scores.items()}
    
    return {
        "per_document": results_per_document,
        "muc_average": avg_muc,
        "ceaf_average": avg_ceaf
    }

# ============================================================================
# 6. VISUALIZATION
# ============================================================================

def generate_html_visualization(text: str, chains: List[Dict]) -> str:
    """Generate HTML visualization with color-coded entities."""
    colors = [
        "#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A", "#98D8C8",
        "#F7DC6F", "#BB8FCE", "#85C1E2", "#F8B88B", "#ABEBC6"
    ]
    
    # Create mention to entity mapping
    mention_to_entity = {}
    for chain in chains:
        entity_id = chain["entity_id"]
        for mention in chain["mentions"]:
            mention_to_entity[mention.lower()] = entity_id
    
    # Tokenize and colorize
    words = text.split()
    colored_html = '<div style="font-size: 16px; line-height: 1.8; font-family: Arial;">'
    
    for word in words:
        word_clean = word.rstrip('.,;:!?')
        punctuation = word[len(word_clean):]
        
        if word_clean.lower() in mention_to_entity:
            entity_id = mention_to_entity[word_clean.lower()]
            color = colors[(entity_id - 1) % len(colors)]
            colored_html += f'<span style="background-color: {color}; padding: 2px 4px; margin: 0 2px;">{word_clean}</span>{punctuation} '
        else:
            colored_html += word + ' '
    
    colored_html += '</div>'
    return colored_html

# ============================================================================
# 7. MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("COREFERENCE RESOLUTION SYSTEM - EVALUATION RESULTS")
    print("="*80)
    
    # Run evaluation
    results = evaluate_system(GOLD_ANNOTATIONS)
    
    # Print overall metrics
    print("\n" + "="*80)
    print("OVERALL METRICS (Macro-Averaged)")
    print("="*80)
    print(f"\n{'Metric':<10} | {'Precision':<12} | {'Recall':<12} | {'F1':<12}")
    print("-"*50)
    print(f"{'MUC':<10} | {results['muc_average']['precision']:<12} | {results['muc_average']['recall']:<12} | {results['muc_average']['f1']:<12}")
    print(f"{'CEAF':<10} | {results['ceaf_average']['precision']:<12} | {results['ceaf_average']['recall']:<12} | {results['ceaf_average']['f1']:<12}")
    
    # Sample detailed results (first 3 documents)
    print("\n" + "="*80)
    print("SAMPLE DETAILED RESULTS (First 3 Documents)")
    print("="*80)
    
    for result in results["per_document"][:3]:
        print(f"\nDocument {result['doc_id']}: {result['text'][:60]}...")
        print(f"  Gold Entities: {result['gold']}")
        print(f"  System Entities: {result['system']}")
        print(f"  MUC: P={result['muc']['precision']}, R={result['muc']['recall']}, F1={result['muc']['f1']}")
        print(f"  CEAF: P={result['ceaf']['precision']}, R={result['ceaf']['recall']}, F1={result['ceaf']['f1']}")
    
    # Save results to JSON
    with open("coref_evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*80)
    print("Results saved to: coref_evaluation_results.json")
    print("="*80)