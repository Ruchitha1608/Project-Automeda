"""
Database Module - SQLite storage for entities, papers, and predictions
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any
import sys

# Add parent to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from config import DATABASE_PATH
except ImportError:
    DATABASE_PATH = Path(__file__).parent / "entities.db"


class EntitiesDatabase:
    """SQLite database for storing extracted entities, papers, and predictions"""
    
    def __init__(self, db_path: str = None):
        """Initialize database connection"""
        self.db_path = db_path or str(DATABASE_PATH)
        self.conn = None
        self._connect()
        self._create_tables()
    
    def _connect(self):
        """Establish database connection"""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
    
    def _create_tables(self):
        """Create all required tables"""
        cursor = self.conn.cursor()
        
        # Papers table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS papers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pmid TEXT UNIQUE,
                title TEXT,
                authors TEXT,
                journal TEXT,
                abstract TEXT,
                pub_date TEXT,
                query_used TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Entities table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                normalized_name TEXT,
                count INTEGER DEFAULT 1,
                source_pmid TEXT,
                context TEXT,
                confidence REAL DEFAULT 1.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (source_pmid) REFERENCES papers(pmid)
            )
        """)
        
        # Relations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS relations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_entity TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                target_entity TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                source_pmid TEXT,
                evidence TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (source_pmid) REFERENCES papers(pmid)
            )
        """)
        
        # Predictions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_path TEXT,
                prediction_class TEXT NOT NULL,
                confidence REAL NOT NULL,
                model_used TEXT,
                heatmap_path TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Biomarkers table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS biomarkers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                gene_name TEXT NOT NULL,
                importance_score REAL,
                expression_level TEXT,
                sample_id TEXT,
                prediction_id INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (prediction_id) REFERENCES predictions(id)
            )
        """)
        
        # Query history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS query_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_text TEXT NOT NULL,
                query_type TEXT,
                results_count INTEGER,
                execution_time REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes for faster queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_relations_source ON relations(source_entity)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_relations_target ON relations(target_entity)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_papers_pmid ON papers(pmid)")
        
        self.conn.commit()
    
    # ============================================================
    # PAPER OPERATIONS
    # ============================================================
    
    def insert_paper(self, paper: Dict) -> int:
        """Insert a paper into the database"""
        cursor = self.conn.cursor()
        
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO papers 
                (pmid, title, authors, journal, abstract, pub_date, query_used)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                paper.get('pmid'),
                paper.get('title'),
                paper.get('authors'),
                paper.get('journal'),
                paper.get('abstract'),
                paper.get('pub_date'),
                paper.get('query_used')
            ))
            self.conn.commit()
            return cursor.lastrowid
        except Exception as e:
            print(f"Error inserting paper: {e}")
            return -1
    
    def get_paper_by_pmid(self, pmid: str) -> Optional[Dict]:
        """Get a paper by its PMID"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM papers WHERE pmid = ?", (pmid,))
        row = cursor.fetchone()
        return dict(row) if row else None
    
    def get_all_papers(self, limit: int = 100) -> List[Dict]:
        """Get all papers"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM papers ORDER BY created_at DESC LIMIT ?", (limit,))
        return [dict(row) for row in cursor.fetchall()]
    
    def search_papers(self, query: str) -> List[Dict]:
        """Search papers by title or abstract"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM papers 
            WHERE title LIKE ? OR abstract LIKE ?
            ORDER BY created_at DESC
        """, (f"%{query}%", f"%{query}%"))
        return [dict(row) for row in cursor.fetchall()]
    
    # ============================================================
    # ENTITY OPERATIONS
    # ============================================================
    
    def insert_entity(self, entity: Dict) -> int:
        """Insert an entity into the database"""
        cursor = self.conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO entities 
                (name, entity_type, normalized_name, count, source_pmid, context, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                entity.get('name'),
                entity.get('entity_type'),
                entity.get('normalized_name', entity.get('name', '').upper()),
                entity.get('count', 1),
                entity.get('source_pmid'),
                entity.get('context'),
                entity.get('confidence', 1.0)
            ))
            self.conn.commit()
            return cursor.lastrowid
        except Exception as e:
            print(f"Error inserting entity: {e}")
            return -1
    
    def insert_entities_batch(self, entities: List[Dict]) -> int:
        """Insert multiple entities at once"""
        count = 0
        for entity in entities:
            if self.insert_entity(entity) > 0:
                count += 1
        return count
    
    def get_entities_by_type(self, entity_type: str) -> List[Dict]:
        """Get all entities of a specific type"""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM entities WHERE entity_type = ? ORDER BY count DESC",
            (entity_type,)
        )
        return [dict(row) for row in cursor.fetchall()]
    
    def get_top_entities(self, entity_type: str = None, limit: int = 20) -> List[Dict]:
        """Get top entities by occurrence count"""
        cursor = self.conn.cursor()
        
        if entity_type:
            cursor.execute("""
                SELECT name, entity_type, SUM(count) as total_count, 
                       AVG(confidence) as avg_confidence
                FROM entities 
                WHERE entity_type = ?
                GROUP BY normalized_name
                ORDER BY total_count DESC
                LIMIT ?
            """, (entity_type, limit))
        else:
            cursor.execute("""
                SELECT name, entity_type, SUM(count) as total_count,
                       AVG(confidence) as avg_confidence
                FROM entities 
                GROUP BY normalized_name
                ORDER BY total_count DESC
                LIMIT ?
            """, (limit,))
        
        return [dict(row) for row in cursor.fetchall()]
    
    def get_entity_stats(self) -> Dict:
        """Get entity statistics by type"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT entity_type, COUNT(*) as count, 
                   COUNT(DISTINCT normalized_name) as unique_count
            FROM entities
            GROUP BY entity_type
        """)
        
        stats = {}
        for row in cursor.fetchall():
            stats[row['entity_type']] = {
                'total': row['count'],
                'unique': row['unique_count']
            }
        return stats
    
    # ============================================================
    # RELATION OPERATIONS
    # ============================================================
    
    def insert_relation(self, relation: Dict) -> int:
        """Insert a relation into the database"""
        cursor = self.conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO relations 
                (source_entity, relation_type, target_entity, confidence, source_pmid, evidence)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                relation.get('source_entity'),
                relation.get('relation_type'),
                relation.get('target_entity'),
                relation.get('confidence', 1.0),
                relation.get('source_pmid'),
                relation.get('evidence')
            ))
            self.conn.commit()
            return cursor.lastrowid
        except Exception as e:
            print(f"Error inserting relation: {e}")
            return -1
    
    def get_relations_for_entity(self, entity_name: str) -> List[Dict]:
        """Get all relations involving an entity"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM relations 
            WHERE source_entity = ? OR target_entity = ?
            ORDER BY confidence DESC
        """, (entity_name, entity_name))
        return [dict(row) for row in cursor.fetchall()]
    
    def get_all_relations(self, limit: int = 100) -> List[Dict]:
        """Get all relations"""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM relations ORDER BY confidence DESC LIMIT ?", 
            (limit,)
        )
        return [dict(row) for row in cursor.fetchall()]
    
    def get_relation_stats(self) -> Dict:
        """Get relation statistics"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT relation_type, COUNT(*) as count, AVG(confidence) as avg_confidence
            FROM relations
            GROUP BY relation_type
        """)
        
        stats = {}
        for row in cursor.fetchall():
            stats[row['relation_type']] = {
                'count': row['count'],
                'avg_confidence': row['avg_confidence']
            }
        return stats
    
    # ============================================================
    # PREDICTION OPERATIONS
    # ============================================================
    
    def insert_prediction(self, prediction: Dict) -> int:
        """Insert a prediction record"""
        cursor = self.conn.cursor()
        
        try:
            metadata = json.dumps(prediction.get('metadata', {}))
            
            cursor.execute("""
                INSERT INTO predictions 
                (image_path, prediction_class, confidence, model_used, heatmap_path, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                prediction.get('image_path'),
                prediction.get('prediction_class'),
                prediction.get('confidence'),
                prediction.get('model_used', 'ResNet50'),
                prediction.get('heatmap_path'),
                metadata
            ))
            self.conn.commit()
            return cursor.lastrowid
        except Exception as e:
            print(f"Error inserting prediction: {e}")
            return -1
    
    def get_prediction_history(self, limit: int = 50) -> List[Dict]:
        """Get prediction history"""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM predictions ORDER BY created_at DESC LIMIT ?",
            (limit,)
        )
        results = []
        for row in cursor.fetchall():
            d = dict(row)
            if d.get('metadata'):
                d['metadata'] = json.loads(d['metadata'])
            results.append(d)
        return results
    
    def get_prediction_stats(self) -> Dict:
        """Get prediction statistics"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT 
                prediction_class,
                COUNT(*) as count,
                AVG(confidence) as avg_confidence
            FROM predictions
            GROUP BY prediction_class
        """)
        
        stats = {'total': 0, 'by_class': {}}
        for row in cursor.fetchall():
            stats['by_class'][row['prediction_class']] = {
                'count': row['count'],
                'avg_confidence': row['avg_confidence']
            }
            stats['total'] += row['count']
        return stats
    
    # ============================================================
    # BIOMARKER OPERATIONS
    # ============================================================
    
    def insert_biomarkers(self, biomarkers: List[tuple], prediction_id: int = None) -> int:
        """Insert biomarkers from analysis"""
        cursor = self.conn.cursor()
        count = 0
        
        for gene_name, importance in biomarkers:
            try:
                cursor.execute("""
                    INSERT INTO biomarkers 
                    (gene_name, importance_score, prediction_id)
                    VALUES (?, ?, ?)
                """, (gene_name, importance, prediction_id))
                count += 1
            except Exception as e:
                print(f"Error inserting biomarker {gene_name}: {e}")
        
        self.conn.commit()
        return count
    
    def get_top_biomarkers(self, limit: int = 20) -> List[Dict]:
        """Get top biomarkers by average importance"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT gene_name, AVG(importance_score) as avg_importance, COUNT(*) as occurrences
            FROM biomarkers
            GROUP BY gene_name
            ORDER BY avg_importance DESC
            LIMIT ?
        """, (limit,))
        return [dict(row) for row in cursor.fetchall()]
    
    # ============================================================
    # QUERY HISTORY
    # ============================================================
    
    def log_query(self, query: str, query_type: str, results_count: int, execution_time: float):
        """Log a query to history"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO query_history (query_text, query_type, results_count, execution_time)
            VALUES (?, ?, ?, ?)
        """, (query, query_type, results_count, execution_time))
        self.conn.commit()
    
    def get_query_history(self, limit: int = 50) -> List[Dict]:
        """Get query history"""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM query_history ORDER BY created_at DESC LIMIT ?",
            (limit,)
        )
        return [dict(row) for row in cursor.fetchall()]
    
    # ============================================================
    # EXPORT OPERATIONS
    # ============================================================
    
    def export_entities_csv(self, filepath: str):
        """Export entities to CSV"""
        import csv
        
        entities = self.get_top_entities(limit=1000)
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['name', 'entity_type', 'total_count', 'avg_confidence'])
            writer.writeheader()
            writer.writerows(entities)
    
    def export_relations_csv(self, filepath: str):
        """Export relations to CSV"""
        import csv
        
        relations = self.get_all_relations(limit=1000)
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            if relations:
                writer = csv.DictWriter(f, fieldnames=relations[0].keys())
                writer.writeheader()
                writer.writerows(relations)
    
    # ============================================================
    # KNOWLEDGE GRAPH DATA
    # ============================================================
    
    def get_graph_data(self) -> Dict:
        """Get data formatted for knowledge graph visualization"""
        entities = self.get_top_entities(limit=50)
        relations = self.get_all_relations(limit=100)
        
        nodes = []
        node_ids = set()
        
        # Add entities as nodes
        for entity in entities:
            if entity['name'] not in node_ids:
                nodes.append({
                    'id': entity['name'],
                    'label': entity['name'],
                    'type': entity['entity_type'],
                    'size': min(entity['total_count'] * 5, 50)
                })
                node_ids.add(entity['name'])
        
        # Add relation endpoints as nodes if not present
        edges = []
        for rel in relations:
            source = rel['source_entity']
            target = rel['target_entity']
            
            if source not in node_ids:
                nodes.append({'id': source, 'label': source, 'type': 'UNKNOWN', 'size': 10})
                node_ids.add(source)
            
            if target not in node_ids:
                nodes.append({'id': target, 'label': target, 'type': 'UNKNOWN', 'size': 10})
                node_ids.add(target)
            
            edges.append({
                'source': source,
                'target': target,
                'label': rel['relation_type'],
                'weight': rel['confidence']
            })
        
        return {'nodes': nodes, 'edges': edges}
    
    # ============================================================
    # UTILITY METHODS
    # ============================================================
    
    def clear_all(self):
        """Clear all data from database"""
        cursor = self.conn.cursor()
        for table in ['papers', 'entities', 'relations', 'predictions', 'biomarkers', 'query_history']:
            cursor.execute(f"DELETE FROM {table}")
        self.conn.commit()
    
    def get_database_stats(self) -> Dict:
        """Get overall database statistics"""
        cursor = self.conn.cursor()
        
        stats = {}
        for table in ['papers', 'entities', 'relations', 'predictions', 'biomarkers']:
            cursor.execute(f"SELECT COUNT(*) as count FROM {table}")
            stats[table] = cursor.fetchone()['count']
        
        return stats
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
    
    def __del__(self):
        """Destructor to ensure connection is closed"""
        self.close()


# Module-level database instance
_db_instance = None

def get_database() -> EntitiesDatabase:
    """Get or create database singleton instance"""
    global _db_instance
    if _db_instance is None:
        _db_instance = EntitiesDatabase()
    return _db_instance


if __name__ == "__main__":
    # Test database
    db = EntitiesDatabase()
    
    # Test paper insertion
    test_paper = {
        'pmid': '12345678',
        'title': 'Test Paper on BRCA1',
        'authors': 'Smith J, Jones K',
        'journal': 'Nature',
        'abstract': 'This is a test abstract about BRCA1 and breast cancer.',
        'query_used': 'BRCA1 breast cancer'
    }
    db.insert_paper(test_paper)
    
    # Test entity insertion
    test_entities = [
        {'name': 'BRCA1', 'entity_type': 'GENE', 'source_pmid': '12345678'},
        {'name': 'breast cancer', 'entity_type': 'DISEASE', 'source_pmid': '12345678'},
        {'name': 'Tamoxifen', 'entity_type': 'DRUG', 'source_pmid': '12345678'},
    ]
    db.insert_entities_batch(test_entities)
    
    # Test relation insertion
    test_relation = {
        'source_entity': 'Tamoxifen',
        'relation_type': 'treats',
        'target_entity': 'breast cancer',
        'confidence': 0.95,
        'source_pmid': '12345678'
    }
    db.insert_relation(test_relation)
    
    # Print stats
    print("Database Statistics:")
    print(db.get_database_stats())
    print("\nEntity Statistics:")
    print(db.get_entity_stats())
    print("\nTop Entities:")
    print(db.get_top_entities(limit=5))
    
    print("\n✅ Database module working correctly!")
