"""
Knowledge Graph Module
Interactive visualization of entity relationships using PyVis and NetworkX
"""

import os
import json
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    
try:
    from pyvis.network import Network
    PYVIS_AVAILABLE = True
except ImportError:
    PYVIS_AVAILABLE = False


# ============================================================
# CONFIGURATION
# ============================================================

# Node colors by entity type
ENTITY_COLORS = {
    'GENE': '#FF6B6B',       # Coral red
    'PROTEIN': '#4ECDC4',    # Teal
    'DRUG': '#45B7D1',       # Sky blue
    'DISEASE': '#96CEB4',    # Mint green
    'MUTATION': '#FFEAA7',   # Yellow
    'PATHWAY': '#DDA0DD',    # Plum
    'CELL_TYPE': '#FFB6C1',  # Light pink
    'BIOMARKER': '#87CEEB',  # Light sky blue
    'DEFAULT': '#95A5A6'     # Gray
}

# Node sizes
NODE_SIZE_BASE = 20
NODE_SIZE_SCALE = 2

# Edge styles
EDGE_STYLES = {
    'ASSOCIATED_WITH': {'color': '#2C3E50', 'width': 2},
    'REGULATES': {'color': '#E74C3C', 'width': 2.5},
    'INHIBITS': {'color': '#9B59B6', 'width': 2},
    'ACTIVATES': {'color': '#27AE60', 'width': 2},
    'EXPRESSED_IN': {'color': '#3498DB', 'width': 1.5},
    'TREATS': {'color': '#1ABC9C', 'width': 2.5},
    'CAUSES': {'color': '#E74C3C', 'width': 2},
    'INTERACTS': {'color': '#7F8C8D', 'width': 1.5},
    'DEFAULT': {'color': '#BDC3C7', 'width': 1}
}


# ============================================================
# KNOWLEDGE GRAPH CLASS
# ============================================================

class KnowledgeGraph:
    """
    Knowledge Graph for biomedical entity relationships
    Supports both interactive (PyVis) and static (NetworkX) visualization
    """
    
    def __init__(self):
        """Initialize empty knowledge graph"""
        self.nodes: Dict[str, Dict] = {}  # node_id -> {name, type, count, ...}
        self.edges: List[Dict] = []        # [{source, target, relation, weight}, ...]
        self.entity_counts: Dict[str, int] = defaultdict(int)
        
        if NETWORKX_AVAILABLE:
            self.graph = nx.DiGraph()
        else:
            self.graph = None
    
    def add_node(
        self,
        node_id: str,
        name: str,
        entity_type: str,
        properties: Dict = None
    ):
        """
        Add or update a node
        
        Args:
            node_id: Unique identifier
            name: Display name
            entity_type: Type of entity (GENE, DRUG, etc.)
            properties: Additional properties
        """
        if node_id in self.nodes:
            self.nodes[node_id]['count'] += 1
        else:
            self.nodes[node_id] = {
                'name': name,
                'type': entity_type,
                'count': 1,
                'properties': properties or {}
            }
            self.entity_counts[entity_type] += 1
            
            if self.graph is not None:
                self.graph.add_node(
                    node_id,
                    label=name,
                    entity_type=entity_type,
                    **properties or {}
                )
    
    def add_edge(
        self,
        source: str,
        target: str,
        relation: str,
        weight: float = 1.0,
        properties: Dict = None
    ):
        """
        Add an edge between nodes
        
        Args:
            source: Source node ID
            target: Target node ID
            relation: Relationship type
            weight: Edge weight
            properties: Additional properties
        """
        edge = {
            'source': source,
            'target': target,
            'relation': relation,
            'weight': weight,
            'properties': properties or {}
        }
        self.edges.append(edge)
        
        if self.graph is not None:
            self.graph.add_edge(
                source, target,
                relation=relation,
                weight=weight,
                **properties or {}
            )
    
    def from_database(self, db_data: Dict):
        """
        Load graph from database export
        
        Args:
            db_data: Dict with 'nodes' and 'edges' lists
        """
        # Add nodes
        for node in db_data.get('nodes', []):
            self.add_node(
                node_id=node.get('id', node.get('name')),
                name=node.get('name'),
                entity_type=node.get('type', 'DEFAULT'),
                properties=node.get('properties', {})
            )
        
        # Add edges
        for edge in db_data.get('edges', []):
            self.add_edge(
                source=edge.get('source'),
                target=edge.get('target'),
                relation=edge.get('relation', 'ASSOCIATED_WITH'),
                weight=edge.get('weight', 1.0),
                properties=edge.get('properties', {})
            )
    
    def from_entities(
        self,
        entities: List[Dict],
        papers: List[Dict] = None,
        create_cooccurrence: bool = True
    ):
        """
        Build graph from extracted entities
        
        Args:
            entities: List of entity dicts with 'name'/'text' and 'type'
            papers: List of paper dicts with 'pmid', 'title'
            create_cooccurrence: Create edges for co-occurring entities
        """
        # Group entities by paper/source
        paper_entities = defaultdict(list)
        
        for ent in entities:
            # Handle both 'name' and 'text' keys (compatibility)
            ent_name = ent.get('name') or ent.get('text', 'unknown')
            
            # Add node
            node_id = f"{ent['type']}_{ent_name}"
            self.add_node(
                node_id=node_id,
                name=ent_name,
                entity_type=ent.get('type', 'DEFAULT'),
                properties={'pmid': ent.get('pmid'), 'source': ent.get('source')}
            )
            
            # Track by paper
            source_id = ent.get('pmid') or ent.get('source', 'unknown')
            paper_entities[source_id].append(node_id)
        
        # Create co-occurrence edges
        if create_cooccurrence:
            for source_id, node_ids in paper_entities.items():
                for i, src in enumerate(node_ids):
                    for tgt in node_ids[i+1:]:
                        self.add_edge(
                            source=src,
                            target=tgt,
                            relation='CO_OCCURS',
                            weight=1.0,
                            properties={'source': source_id}
                        )
    
    def get_statistics(self) -> Dict:
        """Get graph statistics"""
        stats = {
            'total_nodes': len(self.nodes),
            'total_edges': len(self.edges),
            'entity_counts': dict(self.entity_counts),
            'density': 0.0
        }
        
        if self.graph is not None and len(self.nodes) > 1:
            stats['density'] = nx.density(self.graph)
            
            # Degree statistics
            degrees = dict(self.graph.degree())
            if degrees:
                stats['avg_degree'] = sum(degrees.values()) / len(degrees)
                stats['max_degree'] = max(degrees.values())
                stats['hub_nodes'] = sorted(degrees, key=degrees.get, reverse=True)[:5]
        
        return stats
    
    def get_subgraph(
        self,
        center_node: str,
        depth: int = 2,
        max_nodes: int = 50
    ) -> 'KnowledgeGraph':
        """
        Get subgraph centered on a node
        
        Args:
            center_node: Central node ID
            depth: Number of hops from center
            max_nodes: Maximum nodes to include
        
        Returns:
            New KnowledgeGraph with subgraph
        """
        if self.graph is None or center_node not in self.graph:
            return KnowledgeGraph()
        
        # BFS to find connected nodes
        visited = {center_node}
        frontier = [center_node]
        
        for _ in range(depth):
            new_frontier = []
            for node in frontier:
                for neighbor in self.graph.neighbors(node):
                    if neighbor not in visited and len(visited) < max_nodes:
                        visited.add(neighbor)
                        new_frontier.append(neighbor)
            frontier = new_frontier
        
        # Create subgraph
        subgraph = KnowledgeGraph()
        
        for node_id in visited:
            if node_id in self.nodes:
                node = self.nodes[node_id]
                subgraph.add_node(
                    node_id=node_id,
                    name=node['name'],
                    entity_type=node['type'],
                    properties=node['properties']
                )
        
        for edge in self.edges:
            if edge['source'] in visited and edge['target'] in visited:
                subgraph.add_edge(
                    source=edge['source'],
                    target=edge['target'],
                    relation=edge['relation'],
                    weight=edge['weight'],
                    properties=edge['properties']
                )
        
        return subgraph
    
    # --------------------------------------------------------
    # VISUALIZATION METHODS
    # --------------------------------------------------------
    
    def to_pyvis(
        self,
        height: str = "600px",
        width: str = "100%",
        notebook: bool = False,
        physics: bool = True,
        show_legend: bool = True
    ) -> Optional['Network']:
        """
        Create interactive PyVis network
        
        Args:
            height: Canvas height
            width: Canvas width
            notebook: Whether running in Jupyter
            physics: Enable physics simulation
            show_legend: Show entity type legend
        
        Returns:
            PyVis Network object
        """
        if not PYVIS_AVAILABLE:
            print("PyVis not installed. Install with: pip install pyvis")
            return None
        
        net = Network(
            height=height,
            width=width,
            notebook=notebook,
            directed=True,
            bgcolor="#FFFFFF",
            font_color="#2C3E50"
        )
        
        # Configure physics
        if physics:
            net.barnes_hut(
                gravity=-3000,
                central_gravity=0.3,
                spring_length=200,
                spring_strength=0.05,
                damping=0.09
            )
        else:
            net.toggle_physics(False)
        
        # Add nodes
        for node_id, node in self.nodes.items():
            color = ENTITY_COLORS.get(node['type'], ENTITY_COLORS['DEFAULT'])
            size = NODE_SIZE_BASE + node['count'] * NODE_SIZE_SCALE
            
            net.add_node(
                node_id,
                label=node['name'],
                title=f"{node['type']}: {node['name']}\nMentions: {node['count']}",
                color=color,
                size=size,
                font={'size': 12}
            )
        
        # Add edges
        for edge in self.edges:
            style = EDGE_STYLES.get(edge['relation'], EDGE_STYLES['DEFAULT'])
            net.add_edge(
                edge['source'],
                edge['target'],
                title=edge['relation'],
                color=style['color'],
                width=style['width'] * edge['weight']
            )
        
        return net
    
    def save_interactive_html(
        self,
        filepath: str,
        height: str = "800px",
        width: str = "100%",
        physics: bool = True
    ) -> str:
        """
        Save interactive visualization as HTML
        
        Args:
            filepath: Output file path
            height: Canvas height
            width: Canvas width
            physics: Enable physics
        
        Returns:
            Path to saved file
        """
        net = self.to_pyvis(height=height, width=width, physics=physics)
        
        if net is None:
            return ""
        
        # Add custom CSS
        net.set_options("""
        var options = {
            "nodes": {
                "borderWidth": 2,
                "borderWidthSelected": 4,
                "font": {"face": "Arial"}
            },
            "edges": {
                "arrows": {"to": {"enabled": true, "scaleFactor": 0.5}},
                "smooth": {"type": "continuous"}
            },
            "interaction": {
                "hover": true,
                "tooltipDelay": 100
            }
        }
        """)
        
        # Generate legend HTML
        legend_html = self._generate_legend_html()
        
        # Save and inject legend
        net.save_graph(filepath)
        
        # Inject legend into HTML
        with open(filepath, 'r') as f:
            html = f.read()
        
        html = html.replace('</body>', f'{legend_html}</body>')
        
        with open(filepath, 'w') as f:
            f.write(html)
        
        return filepath
    
    def _generate_legend_html(self) -> str:
        """Generate HTML legend for entity types"""
        legend_items = []
        
        for entity_type in self.entity_counts.keys():
            color = ENTITY_COLORS.get(entity_type, ENTITY_COLORS['DEFAULT'])
            count = self.entity_counts[entity_type]
            legend_items.append(f'''
                <div style="display: flex; align-items: center; margin: 5px 0;">
                    <div style="width: 20px; height: 20px; border-radius: 50%; 
                         background-color: {color}; margin-right: 10px;"></div>
                    <span>{entity_type} ({count})</span>
                </div>
            ''')
        
        return f'''
        <div id="legend" style="position: fixed; top: 20px; right: 20px; 
             background: white; padding: 15px; border-radius: 8px;
             box-shadow: 0 2px 10px rgba(0,0,0,0.1); font-family: Arial;
             font-size: 12px; z-index: 1000;">
            <h4 style="margin: 0 0 10px 0; color: #2C3E50;">Entity Types</h4>
            {''.join(legend_items)}
        </div>
        '''
    
    def to_matplotlib(
        self,
        figsize: Tuple[int, int] = (12, 8),
        layout: str = 'spring',
        with_labels: bool = True,
        save_path: str = None
    ):
        """
        Create static matplotlib visualization
        
        Args:
            figsize: Figure size
            layout: Layout algorithm (spring, circular, kamada_kawai)
            with_labels: Show node labels
            save_path: Path to save figure
        
        Returns:
            matplotlib Figure
        """
        import matplotlib.pyplot as plt
        
        if not NETWORKX_AVAILABLE or self.graph is None:
            print("NetworkX not available")
            return None
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get layout
        if layout == 'spring':
            pos = nx.spring_layout(self.graph, k=2, iterations=50)
        elif layout == 'circular':
            pos = nx.circular_layout(self.graph)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(self.graph)
        else:
            pos = nx.spring_layout(self.graph)
        
        # Draw by entity type
        for entity_type in self.entity_counts.keys():
            nodes = [n for n, d in self.nodes.items() if d['type'] == entity_type]
            color = ENTITY_COLORS.get(entity_type, ENTITY_COLORS['DEFAULT'])
            
            nx.draw_networkx_nodes(
                self.graph, pos,
                nodelist=nodes,
                node_color=color,
                node_size=[300 + self.nodes[n]['count'] * 100 for n in nodes],
                alpha=0.8,
                ax=ax
            )
        
        # Draw edges
        nx.draw_networkx_edges(
            self.graph, pos,
            edge_color='#BDC3C7',
            arrows=True,
            arrowsize=15,
            alpha=0.6,
            ax=ax
        )
        
        # Draw labels
        if with_labels:
            labels = {n: d['name'][:15] for n, d in self.nodes.items()}
            nx.draw_networkx_labels(
                self.graph, pos,
                labels=labels,
                font_size=8,
                font_family='sans-serif',
                ax=ax
            )
        
        ax.set_title('Knowledge Graph', fontsize=14, fontweight='bold')
        ax.axis('off')
        
        # Add legend
        legend_elements = []
        for entity_type, color in ENTITY_COLORS.items():
            if entity_type in self.entity_counts:
                from matplotlib.patches import Patch
                legend_elements.append(
                    Patch(facecolor=color, label=f'{entity_type} ({self.entity_counts[entity_type]})')
                )
        
        ax.legend(handles=legend_elements, loc='upper left', framealpha=0.9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def to_json(self) -> str:
        """Export graph as JSON"""
        data = {
            'nodes': [
                {
                    'id': node_id,
                    'name': node['name'],
                    'type': node['type'],
                    'count': node['count'],
                    **node['properties']
                }
                for node_id, node in self.nodes.items()
            ],
            'edges': [
                {
                    'source': e['source'],
                    'target': e['target'],
                    'relation': e['relation'],
                    'weight': e['weight'],
                    **e['properties']
                }
                for e in self.edges
            ],
            'statistics': self.get_statistics()
        }
        return json.dumps(data, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'KnowledgeGraph':
        """Load graph from JSON"""
        data = json.loads(json_str)
        kg = cls()
        kg.from_database(data)
        return kg


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def create_graph_from_literature(
    papers: List[Dict],
    entities_per_paper: Dict[str, List[Dict]]
) -> KnowledgeGraph:
    """
    Create knowledge graph from literature mining results
    
    Args:
        papers: List of paper dicts with 'pmid', 'title'
        entities_per_paper: Dict mapping pmid to list of entities
    
    Returns:
        KnowledgeGraph
    """
    kg = KnowledgeGraph()
    
    # Add all entities with their paper context
    for pmid, entities in entities_per_paper.items():
        for ent in entities:
            ent['pmid'] = pmid
    
    # Flatten entity list
    all_entities = []
    for entities in entities_per_paper.values():
        all_entities.extend(entities)
    
    kg.from_entities(all_entities, papers, create_cooccurrence=True)
    
    return kg


def visualize_biomarker_network(
    biomarkers: List[Tuple[str, float]],
    target_disease: str = "Breast Cancer"
) -> KnowledgeGraph:
    """
    Create simple biomarker-disease network
    
    Args:
        biomarkers: List of (gene_name, importance) tuples
        target_disease: Disease name
    
    Returns:
        KnowledgeGraph
    """
    kg = KnowledgeGraph()
    
    # Add disease node
    disease_id = f"DISEASE_{target_disease}"
    kg.add_node(disease_id, target_disease, "DISEASE")
    
    # Add biomarker nodes and edges
    for gene, importance in biomarkers:
        gene_id = f"GENE_{gene}"
        kg.add_node(gene_id, gene, "GENE", {'importance': importance})
        kg.add_edge(
            gene_id, disease_id,
            relation="ASSOCIATED_WITH",
            weight=importance
        )
    
    # Add some gene-gene interactions for known pairs
    known_interactions = [
        ('BRCA1', 'BRCA2', 'INTERACTS'),
        ('TP53', 'MDM2', 'INHIBITS'),
        ('HER2', 'EGFR', 'INTERACTS'),
        ('ESR1', 'PGR', 'REGULATES'),
        ('PIK3CA', 'AKT1', 'ACTIVATES'),
    ]
    
    biomarker_names = {b[0] for b in biomarkers}
    for src, tgt, rel in known_interactions:
        if src in biomarker_names and tgt in biomarker_names:
            kg.add_edge(f"GENE_{src}", f"GENE_{tgt}", relation=rel)
    
    return kg


if __name__ == "__main__":
    print("Testing Knowledge Graph Module...")
    
    # Create test graph
    kg = KnowledgeGraph()
    
    # Add some entities
    entities = [
        {'name': 'BRCA1', 'type': 'GENE', 'pmid': '12345'},
        {'name': 'BRCA2', 'type': 'GENE', 'pmid': '12345'},
        {'name': 'Breast Cancer', 'type': 'DISEASE', 'pmid': '12345'},
        {'name': 'Tamoxifen', 'type': 'DRUG', 'pmid': '12345'},
        {'name': 'TP53', 'type': 'GENE', 'pmid': '67890'},
        {'name': 'Breast Cancer', 'type': 'DISEASE', 'pmid': '67890'},
    ]
    
    kg.from_entities(entities)
    
    # Add explicit relationships
    kg.add_edge('GENE_BRCA1', 'GENE_BRCA2', 'INTERACTS')
    kg.add_edge('DRUG_Tamoxifen', 'DISEASE_Breast Cancer', 'TREATS')
    
    # Get statistics
    stats = kg.get_statistics()
    print(f"\nGraph Statistics:")
    print(f"  Nodes: {stats['total_nodes']}")
    print(f"  Edges: {stats['total_edges']}")
    print(f"  Entity counts: {stats['entity_counts']}")
    
    # Export JSON
    json_data = kg.to_json()
    print(f"\nJSON Export (first 300 chars):\n{json_data[:300]}...")
    
    # Test biomarker network
    biomarkers = [
        ('BRCA1', 0.15), ('TP53', 0.12), ('HER2', 0.10),
        ('ESR1', 0.08), ('PIK3CA', 0.07), ('BRCA2', 0.06)
    ]
    
    bm_graph = visualize_biomarker_network(biomarkers)
    print(f"\nBiomarker Network Nodes: {len(bm_graph.nodes)}")
    print(f"Biomarker Network Edges: {len(bm_graph.edges)}")
    
    print("\n✅ Knowledge Graph module working correctly!")
