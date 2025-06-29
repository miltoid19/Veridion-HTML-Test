# Just in case the command "pip install beautifulsoup4 scikit-learn numpy lxml " is run before running the code. Claudeai helped a little

import os
import json
import re
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Set
import hashlib
from dataclasses import dataclass
from bs4 import BeautifulSoup, Comment
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

@dataclass
class DocumentFeatures:
    """Container for all extracted features from an HTML document"""
    filename: str
    title: str
    meta_description: str
    headings_structure: List[str]
    text_content: str
    dom_structure: str
    css_features: Dict[str, any]
    form_features: List[Dict[str, str]]
    link_patterns: List[str]
    image_features: List[Dict[str, str]]
    semantic_tags: Counter
    layout_indicators: Dict[str, int]
    visual_hash: str

class HTMLSimilarityAnalyzer:
    """
    Advanced HTML document similarity analyzer that considers multiple dimensions
    of similarity from a user's perspective in a web browser.
    """
    
    def __init__(self, similarity_threshold=0.3, min_cluster_size=2):
        self.similarity_threshold = similarity_threshold
        self.min_cluster_size = min_cluster_size
        self.documents = []
        self.features = []
        
    def extract_features(self, html_content: str, filename: str) -> DocumentFeatures:
        """Extract comprehensive features from HTML document"""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove comments and script/style content for cleaner analysis
        for element in soup(["script", "style"]):
            element.decompose()
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()
        
        return DocumentFeatures(
            filename=filename,
            title=self._extract_title(soup),
            meta_description=self._extract_meta_description(soup),
            headings_structure=self._extract_headings_structure(soup),
            text_content=self._extract_clean_text(soup),
            dom_structure=self._extract_dom_structure(soup),
            css_features=self._extract_css_features(soup, html_content),
            form_features=self._extract_form_features(soup),
            link_patterns=self._extract_link_patterns(soup),
            image_features=self._extract_image_features(soup),
            semantic_tags=self._extract_semantic_tags(soup),
            layout_indicators=self._extract_layout_indicators(soup),
            visual_hash=self._compute_visual_hash(soup)
        )
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract page title"""
        title_tag = soup.find('title')
        return title_tag.get_text().strip() if title_tag else ""
    
    def _extract_meta_description(self, soup: BeautifulSoup) -> str:
        """Extract meta description"""
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        return meta_desc.get('content', '') if meta_desc else ""
    
    def _extract_headings_structure(self, soup: BeautifulSoup) -> List[str]:
        """Extract hierarchical heading structure"""
        headings = []
        for level in range(1, 7):
            for heading in soup.find_all(f'h{level}'):
                text = heading.get_text().strip()
                if text:
                    headings.append(f"h{level}:{text[:50]}")
        return headings
    
    def _extract_clean_text(self, soup: BeautifulSoup) -> str:
        """Extract clean text content"""
        # Remove navigation, footer, sidebar content that might be boilerplate
        for element in soup.find_all(['nav', 'footer', 'aside']):
            element.decompose()
        
        text = soup.get_text()
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def _extract_dom_structure(self, soup: BeautifulSoup) -> str:
        """Extract DOM structure signature"""
        def get_structure(element, depth=0, max_depth=5):
            if depth > max_depth:
                return ""
            
            structure = element.name if element.name else ""
            
            # Add class information for layout-significant elements
            if element.name in ['div', 'section', 'article', 'main', 'aside', 'header', 'footer']:
                classes = element.get('class', [])
                if classes:
                    structure += "." + ".".join(classes[:2])  # Limit to first 2 classes
            
            children_structure = []
            for child in element.children:
                if hasattr(child, 'name') and child.name:
                    child_struct = get_structure(child, depth + 1, max_depth)
                    if child_struct:
                        children_structure.append(child_struct)
            
            if children_structure:
                structure += "[" + ",".join(children_structure) + "]"
            
            return structure
        
        return get_structure(soup.body if soup.body else soup)
    
    def _extract_css_features(self, soup: BeautifulSoup, html_content: str) -> Dict[str, any]:
        """Extract CSS-related features that affect visual appearance"""
        features = {
            'inline_styles': [],
            'class_patterns': [],
            'embedded_css_rules': 0,
            'external_stylesheets': 0,
            'bootstrap_detected': False,
            'css_frameworks': []
        }
        
        # Extract inline styles
        for element in soup.find_all(attrs={'style': True}):
            features['inline_styles'].append(element.get('style', ''))
        
        # Extract class patterns
        for element in soup.find_all(attrs={'class': True}):
            classes = element.get('class', [])
            features['class_patterns'].extend(classes)
        
        # Count embedded CSS
        for style_tag in soup.find_all('style'):
            if style_tag.string:
                features['embedded_css_rules'] += style_tag.string.count('{')
        
        # Count external stylesheets
        for link in soup.find_all('link', rel='stylesheet'):
            features['external_stylesheets'] += 1
            href = link.get('href', '').lower()
            if 'bootstrap' in href:
                features['bootstrap_detected'] = True
            # Detect other common frameworks
            for framework in ['foundation', 'bulma', 'tailwind', 'materialize']:
                if framework in href:
                    features['css_frameworks'].append(framework)
        
        return features
    
    def _extract_form_features(self, soup: BeautifulSoup) -> List[Dict[str, str]]:
        """Extract form structures and input types"""
        forms = []
        for form in soup.find_all('form'):
            form_data = {
                'method': form.get('method', 'get').lower(),
                'action': form.get('action', ''),
                'inputs': []
            }
            
            for input_elem in form.find_all(['input', 'textarea', 'select']):
                input_type = input_elem.get('type', 'text') if input_elem.name == 'input' else input_elem.name
                form_data['inputs'].append(input_type)
            
            forms.append(form_data)
        
        return forms
    
    def _extract_link_patterns(self, soup: BeautifulSoup) -> List[str]:
        """Extract navigation and link patterns"""
        patterns = []
        
        # Navigation links
        nav_sections = soup.find_all(['nav', 'ul', 'ol'])
        for nav in nav_sections:
            links = nav.find_all('a')
            if len(links) > 2:  # Likely a navigation menu
                pattern = 'nav:' + str(len(links))
                patterns.append(pattern)
        
        # Footer links
        footer = soup.find('footer')
        if footer:
            footer_links = len(footer.find_all('a'))
            patterns.append(f'footer_links:{footer_links}')
        
        return patterns
    
    def _extract_image_features(self, soup: BeautifulSoup) -> List[Dict[str, str]]:
        """Extract image-related features"""
        images = []
        for img in soup.find_all('img'):
            img_data = {
                'alt': img.get('alt', ''),
                'src_pattern': self._categorize_image_src(img.get('src', '')),
                'has_dimensions': bool(img.get('width') or img.get('height'))
            }
            images.append(img_data)
        
        return images
    
    def _categorize_image_src(self, src: str) -> str:
        """Categorize image source patterns"""
        src = src.lower()
        if 'logo' in src:
            return 'logo'
        elif any(x in src for x in ['banner', 'hero', 'header']):
            return 'banner'
        elif any(x in src for x in ['thumb', 'small', 'icon']):
            return 'thumbnail'
        elif src.startswith('data:'):
            return 'inline'
        elif 'placeholder' in src or 'dummy' in src:
            return 'placeholder'
        else:
            return 'content'
    
    def _extract_semantic_tags(self, soup: BeautifulSoup) -> Counter:
        """Count semantic HTML5 tags"""
        semantic_tags = [
            'article', 'section', 'nav', 'aside', 'header', 'footer', 
            'main', 'figure', 'figcaption', 'time', 'mark'
        ]
        
        tag_counts = Counter()
        for tag in semantic_tags:
            tag_counts[tag] = len(soup.find_all(tag))
        
        return tag_counts
    
    def _extract_layout_indicators(self, soup: BeautifulSoup) -> Dict[str, int]:
        """Extract indicators of layout structure"""
        indicators = {
            'containers': len(soup.find_all(['div', 'section'], class_=re.compile(r'container|wrapper|content'))),
            'columns': len(soup.find_all(['div'], class_=re.compile(r'col|column|grid'))),
            'cards': len(soup.find_all(['div', 'article'], class_=re.compile(r'card|item|box'))),
            'buttons': len(soup.find_all(['button', 'a'], class_=re.compile(r'btn|button'))),
            'total_divs': len(soup.find_all('div')),
            'lists': len(soup.find_all(['ul', 'ol'])),
            'tables': len(soup.find_all('table'))
        }
        
        return indicators
    
    def _compute_visual_hash(self, soup: BeautifulSoup) -> str:
        """Compute a hash representing the visual structure"""
        # Create a simplified representation focusing on visual elements
        visual_elements = []
        
        for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'div', 'section', 'article']):
            # Include tag name and significant classes
            elem_repr = element.name
            classes = element.get('class', [])
            significant_classes = [c for c in classes if any(keyword in c.lower() 
                                 for keyword in ['header', 'nav', 'footer', 'content', 'main', 'sidebar', 'container'])]
            if significant_classes:
                elem_repr += "." + ".".join(significant_classes[:2])
            
            visual_elements.append(elem_repr)
        
        visual_signature = "|".join(visual_elements[:50])  # Limit to prevent too long signatures
        return hashlib.md5(visual_signature.encode()).hexdigest()[:16]
    
    def compute_similarity_matrix(self, features_list: List[DocumentFeatures]) -> np.ndarray:
        """Compute comprehensive similarity matrix between documents"""
        n_docs = len(features_list)
        similarity_matrix = np.zeros((n_docs, n_docs))
        
        # Prepare text data for TF-IDF
        text_data = []
        for features in features_list:
            combined_text = f"{features.title} {features.meta_description} {features.text_content[:1000]}"
            # Ensure we have some text content
            if not combined_text.strip():
                combined_text = f"document {features.filename}"
            text_data.append(combined_text)
        
        # TF-IDF similarity for text content
        text_similarity = np.identity(n_docs)  # Default to identity matrix
        if len(set(text_data)) > 1:  # Check if we have varied content
            try:
                tfidf = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2), min_df=1)
                tfidf_matrix = tfidf.fit_transform(text_data)
                if tfidf_matrix.shape[1] > 0:  # Check if we have features
                    text_similarity = cosine_similarity(tfidf_matrix)
                    # Ensure no negative values and handle NaN
                    text_similarity = np.nan_to_num(text_similarity, nan=0.0, posinf=1.0, neginf=0.0)
                    text_similarity = np.clip(text_similarity, 0.0, 1.0)
            except Exception as e:
                print(f"Warning: TF-IDF computation failed: {e}")
                text_similarity = np.identity(n_docs)
        
        for i in range(n_docs):
            for j in range(i, n_docs):
                if i == j:
                    similarity_matrix[i][j] = 1.0
                    continue
                
                # Multiple similarity components with error handling
                similarities = {}
                try:
                    similarities['text'] = float(text_similarity[i][j])
                except:
                    similarities['text'] = 0.0
                
                try:
                    similarities['structure'] = self._compute_structure_similarity(features_list[i], features_list[j])
                except:
                    similarities['structure'] = 0.0
                
                try:
                    similarities['visual'] = self._compute_visual_similarity(features_list[i], features_list[j])
                except:
                    similarities['visual'] = 0.0
                
                try:
                    similarities['semantic'] = self._compute_semantic_similarity(features_list[i], features_list[j])
                except:
                    similarities['semantic'] = 0.0
                
                try:
                    similarities['functional'] = self._compute_functional_similarity(features_list[i], features_list[j])
                except:
                    similarities['functional'] = 0.0
                
                # Ensure all similarities are in [0, 1] range
                for key in similarities:
                    similarities[key] = max(0.0, min(1.0, similarities[key]))
                    if np.isnan(similarities[key]):
                        similarities[key] = 0.0
                
                # Weighted combination of similarities
                weights = {
                    'text': 0.25,
                    'structure': 0.25,
                    'visual': 0.25,
                    'semantic': 0.15,
                    'functional': 0.10
                }
                
                combined_similarity = sum(similarities[key] * weights[key] for key in similarities)
                combined_similarity = max(0.0, min(1.0, combined_similarity))  # Ensure [0, 1] range
                
                similarity_matrix[i][j] = similarity_matrix[j][i] = combined_similarity
        
        # Final validation of similarity matrix
        similarity_matrix = np.nan_to_num(similarity_matrix, nan=0.0, posinf=1.0, neginf=0.0)
        similarity_matrix = np.clip(similarity_matrix, 0.0, 1.0)
        
        return similarity_matrix
    
    def _compute_structure_similarity(self, feat1: DocumentFeatures, feat2: DocumentFeatures) -> float:
        """Compare DOM structure similarity"""
        # Visual hash similarity
        visual_sim = 1.0 if feat1.visual_hash == feat2.visual_hash else 0.0
        
        # Headings structure similarity
        headings_sim = self._compute_sequence_similarity(feat1.headings_structure, feat2.headings_structure)
        
        # Layout indicators similarity
        layout_sim = self._compute_dict_similarity(feat1.layout_indicators, feat2.layout_indicators)
        
        return (visual_sim * 0.4 + headings_sim * 0.3 + layout_sim * 0.3)
    
    def _compute_visual_similarity(self, feat1: DocumentFeatures, feat2: DocumentFeatures) -> float:
        """Compare visual/CSS features similarity"""
        css_sim = 0.0
        
        # Class patterns similarity
        if feat1.css_features['class_patterns'] and feat2.css_features['class_patterns']:
            common_classes = set(feat1.css_features['class_patterns']) & set(feat2.css_features['class_patterns'])
            total_classes = set(feat1.css_features['class_patterns']) | set(feat2.css_features['class_patterns'])
            css_sim += (len(common_classes) / len(total_classes)) if total_classes else 0
        
        # Framework detection
        framework_sim = 0.0
        if feat1.css_features['bootstrap_detected'] == feat2.css_features['bootstrap_detected']:
            framework_sim += 0.5
        if set(feat1.css_features['css_frameworks']) == set(feat2.css_features['css_frameworks']):
            framework_sim += 0.5
        
        return (css_sim * 0.7 + framework_sim * 0.3)
    
    def _compute_semantic_similarity(self, feat1: DocumentFeatures, feat2: DocumentFeatures) -> float:
        """Compare semantic HTML usage"""
        # Semantic tags similarity
        all_tags = set(feat1.semantic_tags.keys()) | set(feat2.semantic_tags.keys())
        if not all_tags:
            return 1.0
        
        similarity = 0.0
        for tag in all_tags:
            count1 = feat1.semantic_tags.get(tag, 0)
            count2 = feat2.semantic_tags.get(tag, 0)
            # Normalize by max count to handle different scales
            max_count = max(count1, count2, 1)
            tag_sim = 1 - abs(count1 - count2) / max_count
            similarity += tag_sim
        
        return similarity / len(all_tags)
    
    def _compute_functional_similarity(self, feat1: DocumentFeatures, feat2: DocumentFeatures) -> float:
        """Compare functional elements (forms, links, etc.)"""
        # Form similarity
        form_sim = self._compare_forms(feat1.form_features, feat2.form_features)
        
        # Link patterns similarity
        link_sim = self._compute_sequence_similarity(feat1.link_patterns, feat2.link_patterns)
        
        # Image features similarity
        img_sim = self._compare_images(feat1.image_features, feat2.image_features)
        
        return (form_sim * 0.4 + link_sim * 0.3 + img_sim * 0.3)
    
    def _compare_forms(self, forms1: List[Dict], forms2: List[Dict]) -> float:
        """Compare form structures"""
        if not forms1 and not forms2:
            return 1.0
        if not forms1 or not forms2:
            return 0.0
        
        # Simple comparison based on form count and input types
        if len(forms1) != len(forms2):
            return 0.5  # Different number of forms
        
        similarity = 0.0
        for f1, f2 in zip(forms1, forms2):
            if f1['method'] == f2['method']:
                similarity += 0.3
            
            # Compare input types
            inputs1 = set(f1['inputs'])
            inputs2 = set(f2['inputs'])
            if inputs1 or inputs2:
                input_sim = len(inputs1 & inputs2) / len(inputs1 | inputs2)
            else:
                input_sim = 1.0
            similarity += input_sim * 0.7
        
        return similarity / len(forms1)
    
    def _compare_images(self, images1: List[Dict], images2: List[Dict]) -> float:
        """Compare image features"""
        if not images1 and not images2:
            return 1.0
        if not images1 or not images2:
            return 0.0
        
        # Compare image source patterns
        patterns1 = [img['src_pattern'] for img in images1]
        patterns2 = [img['src_pattern'] for img in images2]
        
        return self._compute_sequence_similarity(patterns1, patterns2)
    
    def _compute_sequence_similarity(self, seq1: List, seq2: List) -> float:
        """Compute similarity between two sequences"""
        if not seq1 and not seq2:
            return 1.0
        if not seq1 or not seq2:
            return 0.0
        
        set1, set2 = set(seq1), set(seq2)
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def _compute_dict_similarity(self, dict1: Dict, dict2: Dict) -> float:
        """Compute similarity between two dictionaries with numeric values"""
        all_keys = set(dict1.keys()) | set(dict2.keys())
        if not all_keys:
            return 1.0
        
        similarity = 0.0
        for key in all_keys:
            val1 = dict1.get(key, 0)
            val2 = dict2.get(key, 0)
            max_val = max(val1, val2, 1)
            key_sim = 1 - abs(val1 - val2) / max_val
            similarity += key_sim
        
        return similarity / len(all_keys)
    
    def cluster_documents(self, directory_path: str) -> List[List[str]]:
        """Main method to cluster HTML documents in a directory"""
        print(f"Processing directory: {directory_path}")
        
        # Load and process all HTML files
        html_files = list(Path(directory_path).glob("*.html"))
        print(f"Found {len(html_files)} HTML files")
        
        if len(html_files) == 0:
            return []
        
        features_list = []
        for html_file in html_files:
            try:
                with open(html_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                features = self.extract_features(content, html_file.name)
                features_list.append(features)
                print(f"Processed: {html_file.name}")
            except Exception as e:
                print(f"Error processing {html_file.name}: {e}")
        
        if len(features_list) <= 1:
            return [[f.filename] for f in features_list]
        
        # Compute similarity matrix
        print("Computing similarity matrix...")
        try:
            similarity_matrix = self.compute_similarity_matrix(features_list)
            
            # Validate similarity matrix
            if np.any(np.isnan(similarity_matrix)) or np.any(similarity_matrix < 0):
                print("Warning: Invalid similarity matrix detected, using fallback clustering")
                # Fallback: group by exact visual hash matches
                hash_groups = defaultdict(list)
                for features in features_list:
                    hash_groups[features.visual_hash].append(features.filename)
                return list(hash_groups.values())
            
            # Convert similarity to distance for clustering
            distance_matrix = 1 - similarity_matrix
            
            # Ensure distance matrix is valid
            distance_matrix = np.clip(distance_matrix, 0.0, 1.0)
            
            # Use DBSCAN clustering with precomputed distances
            eps = 1 - self.similarity_threshold  # Convert similarity threshold to distance
            eps = max(0.1, min(0.9, eps))  # Ensure eps is in valid range
            
            clustering = DBSCAN(eps=eps, min_samples=self.min_cluster_size, metric='precomputed')
            cluster_labels = clustering.fit_predict(distance_matrix)
            
        except Exception as e:
            print(f"Clustering error: {e}")
            # Fallback clustering by filename similarity
            result_clusters = []
            for features in features_list:
                result_clusters.append([features.filename])
            return result_clusters
        
        # Group documents by cluster labels
        clusters = defaultdict(list)
        for i, label in enumerate(cluster_labels):
            clusters[label].append(features_list[i].filename)
        
        # Convert to list format, handle noise points (-1 label) as individual clusters
        result_clusters = []
        for label, filenames in clusters.items():
            if label == -1:  # Noise points from DBSCAN
                # Each noise point becomes its own cluster
                for filename in filenames:
                    result_clusters.append([filename])
            else:
                result_clusters.append(filenames)
        
        # Sort clusters by size (largest first) and then by first filename
        result_clusters.sort(key=lambda x: (-len(x), x[0]))
        
        print(f"Created {len(result_clusters)} clusters")
        return result_clusters
    
    def analyze_and_report(self, directory_path: str) -> Dict:
        """Analyze directory and provide detailed report"""
        clusters = self.cluster_documents(directory_path)
        
        # Generate report
        report = {
            'directory': directory_path,
            'total_documents': sum(len(cluster) for cluster in clusters),
            'num_clusters': len(clusters),
            'clusters': clusters,
            'cluster_sizes': [len(cluster) for cluster in clusters],
            'singleton_clusters': len([c for c in clusters if len(c) == 1]),
            'largest_cluster_size': max(len(cluster) for cluster in clusters) if clusters else 0
        }
        
        return report

def main():
    """Main function to process all tier directories"""
    # Get the directory where code.py is located
    script_dir = Path(__file__).parent
    base_path = script_dir / "clones"
    
    # Initialize analyzer with tuned parameters
    analyzer = HTMLSimilarityAnalyzer(similarity_threshold=0.4, min_cluster_size=2)
    
    results = {}
    
    # Open results.txt file for writing in the same directory as code.py
    results_file = script_dir / "results.txt"
    
    def print_and_write(text, file_handle=None):
        """Print to console and write to file"""
        print(text)
        if file_handle:
            file_handle.write(text + '\n')
    
    with open(results_file, 'w', encoding='utf-8') as f:
        print_and_write("HTML DOCUMENT CLUSTERING RESULTS", f)
        print_and_write("=" * 50, f)
        print_and_write(f"Script directory: {script_dir}", f)
        print_and_write(f"Base directory: {base_path}", f)
        print_and_write("", f)
        
        for tier in range(1, 5):
            tier_path = base_path / f"tier{tier}"
            if tier_path.exists():
                header = f"PROCESSING TIER {tier}"
                print_and_write(f"\n{'='*50}", f)
                print_and_write(header, f)
                print_and_write('='*50, f)
                
                try:
                    report = analyzer.analyze_and_report(str(tier_path))
                    results[f"tier{tier}"] = report
                    
                    # Print and write results
                    print_and_write(f"\nResults for Tier {tier}:", f)
                    print_and_write(f"Total documents: {report['total_documents']}", f)
                    print_and_write(f"Number of clusters: {report['num_clusters']}", f)
                    print_and_write(f"Cluster sizes: {report['cluster_sizes']}", f)
                    print_and_write(f"Singleton clusters: {report['singleton_clusters']}", f)
                    
                    print_and_write(f"\nClusters:", f)
                    for i, cluster in enumerate(report['clusters'], 1):
                        cluster_text = f"Cluster {i}: {cluster}"
                        print_and_write(cluster_text, f)
                    
                except Exception as e:
                    error_msg = f"Error processing tier {tier}: {e}"
                    print_and_write(error_msg, f)
                    results[f"tier{tier}"] = {"error": str(e)}
            else:
                not_found_msg = f"Tier {tier} directory not found: {tier_path}"
                print_and_write(not_found_msg, f)
        
        # Summary section
        print_and_write(f"\n{'='*50}", f)
        print_and_write("SUMMARY", f)
        print_and_write('='*50, f)
        
        for tier, data in results.items():
            if 'error' not in data:
                summary_line = f"{tier.upper()}: {data['num_clusters']} clusters from {data['total_documents']} documents"
            else:
                summary_line = f"{tier.upper()}: Error - {data['error']}"
            print_and_write(summary_line, f)
        
        print_and_write("", f)
        print_and_write("=" * 50, f)
        print_and_write("Results saved to both console output and results.txt", f)
    
    # Save detailed results to JSON in the same directory as code.py
    json_output_file = script_dir / "clustering_results.json"
    with open(json_output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nDetailed JSON results saved to: {json_output_file}")
    print(f"Text results saved to: {results_file}")

if __name__ == "__main__":
    main()