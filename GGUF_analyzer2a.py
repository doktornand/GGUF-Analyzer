import struct
import numpy as np
import json
import os
from pathlib import Path
from collections import defaultdict, OrderedDict
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
import re
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy import stats
from scipy.spatial.distance import cosine
import networkx as nx
from enum import IntEnum

class GGMLType(IntEnum):
    """Types de données GGML/GGUF"""
    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q8_1 = 9
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    Q8_K = 15
    IQ2_XXS = 16
    IQ2_XS = 17
    IQ3_XXS = 18
    IQ1_S = 19
    IQ4_NL = 20
    IQ3_S = 21
    IQ2_S = 22
    IQ4_XS = 23
    I8 = 24
    I16 = 25
    I32 = 26
    I64 = 27
    F64 = 28
    IQ1_M = 29

# Mapping des types vers leurs propriétés
GGML_TYPE_INFO = {
    GGMLType.F32: {"name": "F32", "block_size": 1, "type_size": 4},
    GGMLType.F16: {"name": "F16", "block_size": 1, "type_size": 2},
    GGMLType.Q4_0: {"name": "Q4_0", "block_size": 32, "type_size": 18},
    GGMLType.Q4_1: {"name": "Q4_1", "block_size": 32, "type_size": 20},
    GGMLType.Q5_0: {"name": "Q5_0", "block_size": 32, "type_size": 22},
    GGMLType.Q5_1: {"name": "Q5_1", "block_size": 32, "type_size": 24},
    GGMLType.Q8_0: {"name": "Q8_0", "block_size": 32, "type_size": 34},
    GGMLType.Q8_1: {"name": "Q8_1", "block_size": 32, "type_size": 36},
    GGMLType.Q2_K: {"name": "Q2_K", "block_size": 256, "type_size": 82},
    GGMLType.Q3_K: {"name": "Q3_K", "block_size": 256, "type_size": 110},
    GGMLType.Q4_K: {"name": "Q4_K", "block_size": 256, "type_size": 144},
    GGMLType.Q5_K: {"name": "Q5_K", "block_size": 256, "type_size": 176},
    GGMLType.Q6_K: {"name": "Q6_K", "block_size": 256, "type_size": 210},
    GGMLType.Q8_K: {"name": "Q8_K", "block_size": 256, "type_size": 292},
    GGMLType.I8: {"name": "I8", "block_size": 1, "type_size": 1},
    GGMLType.I16: {"name": "I16", "block_size": 1, "type_size": 2},
    GGMLType.I32: {"name": "I32", "block_size": 1, "type_size": 4},
    GGMLType.I64: {"name": "I64", "block_size": 1, "type_size": 8},
    GGMLType.F64: {"name": "F64", "block_size": 1, "type_size": 8},
}

class GGUFAnalyzer:
    """
    Analyseur progressif de fichiers GGUF
    Permet une analyse en profondeur des modèles quantifiés stockés au format GGUF
    """
    
    def __init__(self, model_path: str):
        """
        Initialise l'analyseur avec le chemin vers le fichier GGUF
        
        Args:
            model_path: Chemin vers le fichier .gguf
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Fichier GGUF non trouvé: {model_path}")
        
        self.file_size = self.model_path.stat().st_size
        self.metadata = {}
        self.tensors_info = {}
        self.analysis_results = {}
        self.header_info = {}
        
        # Chargement initial de la structure GGUF
        self._load_gguf_structure()
    
    def _load_gguf_structure(self):
        """Charge la structure de base du fichier GGUF"""
        try:
            with open(self.model_path, 'rb') as f:
                # Lecture de l'en-tête GGUF
                magic = f.read(4)
                if magic != b'GGUF':
                    raise ValueError("Fichier GGUF invalide: magic number incorrect")
                
                version = struct.unpack('<I', f.read(4))[0]
                tensor_count = struct.unpack('<Q', f.read(8))[0]
                metadata_kv_count = struct.unpack('<Q', f.read(8))[0]
                
                self.header_info = {
                    'magic': magic.decode('utf-8'),
                    'version': version,
                    'tensor_count': tensor_count,
                    'metadata_kv_count': metadata_kv_count
                }
                
                # Lecture des métadonnées
                self._read_metadata(f, metadata_kv_count)
                
                # Lecture des informations sur les tenseurs
                self._read_tensor_info(f, tensor_count)
                
                print(f"✓ Structure GGUF chargée: {tensor_count} tenseurs, {metadata_kv_count} métadonnées")
                
        except Exception as e:
            print(f"⚠ Erreur lors du chargement de la structure GGUF: {e}")
    
    def _read_metadata(self, f, count: int):
        """Lit les métadonnées du fichier GGUF"""
        for i in range(count):
            try:
                # Lecture de la clé
                key_length = struct.unpack('<Q', f.read(8))[0]
                key = f.read(key_length).decode('utf-8')
                
                # Lecture du type de valeur
                value_type = struct.unpack('<I', f.read(4))[0]
                
                # Lecture de la valeur selon son type
                value = self._read_value_by_type(f, value_type)
                
                self.metadata[key] = {
                    'value': value,
                    'type': value_type
                }
                
            except Exception as e:
                print(f"⚠ Erreur lecture métadonnée {i}: {e}")
    
    def _read_value_by_type(self, f, value_type: int):
        """Lit une valeur selon son type GGUF"""
        try:
            if value_type == 0:  # UINT8
                return struct.unpack('<B', f.read(1))[0]
            elif value_type == 1:  # INT8
                return struct.unpack('<b', f.read(1))[0]
            elif value_type == 2:  # UINT16
                return struct.unpack('<H', f.read(2))[0]
            elif value_type == 3:  # INT16
                return struct.unpack('<h', f.read(2))[0]
            elif value_type == 4:  # UINT32
                return struct.unpack('<I', f.read(4))[0]
            elif value_type == 5:  # INT32
                return struct.unpack('<i', f.read(4))[0]
            elif value_type == 6:  # FLOAT32
                return struct.unpack('<f', f.read(4))[0]
            elif value_type == 7:  # BOOL
                return struct.unpack('<B', f.read(1))[0] != 0
            elif value_type == 8:  # STRING
                length = struct.unpack('<Q', f.read(8))[0]
                return f.read(length).decode('utf-8')
            elif value_type == 9:  # ARRAY
                array_type = struct.unpack('<I', f.read(4))[0]
                array_length = struct.unpack('<Q', f.read(8))[0]
                array_values = []
                for _ in range(array_length):
                    array_values.append(self._read_value_by_type(f, array_type))
                return array_values
            elif value_type == 10:  # UINT64
                return struct.unpack('<Q', f.read(8))[0]
            elif value_type == 11:  # INT64
                return struct.unpack('<q', f.read(8))[0]
            elif value_type == 12:  # FLOAT64
                return struct.unpack('<d', f.read(8))[0]
            else:
                return f"Unknown type {value_type}"
        except Exception as e:
            return f"Error reading type {value_type}: {e}"
    
    def _read_tensor_info(self, f, count: int):
        """Lit les informations sur les tenseurs"""
        for i in range(count):
            try:
                # Nom du tenseur
                name_length = struct.unpack('<Q', f.read(8))[0]
                name = f.read(name_length).decode('utf-8')
                
                # Nombre de dimensions
                n_dimensions = struct.unpack('<I', f.read(4))[0]
                
                # Dimensions
                shape = []
                for _ in range(n_dimensions):
                    dim = struct.unpack('<Q', f.read(8))[0]
                    shape.append(dim)
                
                # Type GGML
                ggml_type = struct.unpack('<I', f.read(4))[0]
                
                # Offset dans le fichier
                offset = struct.unpack('<Q', f.read(8))[0]
                
                # Calcul de la taille
                element_count = np.prod(shape) if shape else 0
                type_info = GGML_TYPE_INFO.get(ggml_type, {"name": f"Unknown_{ggml_type}", "block_size": 1, "type_size": 1})
                
                # Calcul de la taille en tenant compte de la quantification
                if type_info["block_size"] > 1:
                    # Type quantifié
                    n_blocks = (element_count + type_info["block_size"] - 1) // type_info["block_size"]
                    size_bytes = n_blocks * type_info["type_size"]
                else:
                    # Type non quantifié
                    size_bytes = element_count * type_info["type_size"]
                
                self.tensors_info[name] = {
                    'shape': shape,
                    'ggml_type': ggml_type,
                    'type_name': type_info["name"],
                    'offset': offset,
                    'size_bytes': size_bytes,
                    'size_mb': size_bytes / (1024 * 1024),
                    'element_count': element_count,
                    'block_size': type_info["block_size"],
                    'is_quantized': type_info["block_size"] > 1
                }
                
            except Exception as e:
                print(f"⚠ Erreur lecture tenseur {i}: {e}")
    
    def analyze_structure(self) -> Dict[str, Any]:
        """
        Niveau 1: Analyse structurelle basique
        """
        print("🔍 Analyse structurelle en cours...")
        
        structure_info = {
            'file_size_mb': round(self.file_size / (1024 * 1024), 2),
            'gguf_version': self.header_info.get('version', 0),
            'tensor_count': self.header_info.get('tensor_count', 0),
            'metadata_count': self.header_info.get('metadata_kv_count', 0),
            'model_info': {},
            'quantization_summary': {},
            'architecture_hints': {}
        }
        
        # Extraction des informations du modèle depuis les métadonnées
        if self.metadata:
            structure_info['model_info'] = {
                'name': self.metadata.get('general.name', {}).get('value', 'Unknown'),
                'architecture': self.metadata.get('general.architecture', {}).get('value', 'Unknown'),
                'file_type': self.metadata.get('general.file_type', {}).get('value', 'Unknown'),
                'quantization_version': self.metadata.get('general.quantization_version', {}).get('value', 'Unknown')
            }
            
            # Informations architecturales
            arch_keys = [k for k in self.metadata.keys() if k.startswith(structure_info['model_info']['architecture'] + '.')]
            for key in arch_keys:
                param_name = key.split('.', 1)[1]
                structure_info['architecture_hints'][param_name] = self.metadata[key]['value']
        
        # Analyse de la quantification
        quantization_types = defaultdict(int)
        total_size = 0
        
        for tensor_info in self.tensors_info.values():
            quantization_types[tensor_info['type_name']] += 1
            total_size += tensor_info['size_bytes']
        
        structure_info['quantization_summary'] = {
            'types_distribution': dict(quantization_types),
            'total_model_size_mb': round(total_size / (1024 * 1024), 2),
            'quantized_tensors': sum(1 for t in self.tensors_info.values() if t['is_quantized']),
            'unquantized_tensors': sum(1 for t in self.tensors_info.values() if not t['is_quantized'])
        }
        
        self.analysis_results['structure'] = structure_info
        return structure_info
    
    def analyze_tensors(self, sample_size: int = 10) -> Dict[str, Any]:
        """
        Niveau 2: Analyse des tenseurs
        """
        print("🔬 Analyse des tenseurs en cours...")
        
        if not self.tensors_info:
            print("⚠ Aucune information de tenseur disponible")
            return {}
        
        tensor_analysis = {
            'layer_classification': defaultdict(int),
            'quantization_analysis': {},
            'size_distribution': [],
            'largest_tensors': [],
            'shape_analysis': {},
            'compression_analysis': {},
            'detailed_sample': {}
        }
        
        # Classification des couches par nom
        for name, info in self.tensors_info.items():
            # Classification basée sur les patterns de noms
            if any(pattern in name.lower() for pattern in ['embed', 'token', 'wte']):
                tensor_analysis['layer_classification']['embedding'] += 1
            elif any(pattern in name.lower() for pattern in ['attn', 'attention', 'self_attn']):
                tensor_analysis['layer_classification']['attention'] += 1
            elif any(pattern in name.lower() for pattern in ['mlp', 'ffn', 'feed_forward', 'fc']):
                tensor_analysis['layer_classification']['mlp'] += 1
            elif any(pattern in name.lower() for pattern in ['norm', 'ln', 'layer_norm']):
                tensor_analysis['layer_classification']['normalization'] += 1
            elif any(pattern in name.lower() for pattern in ['head', 'lm_head', 'output', 'classifier']):
                tensor_analysis['layer_classification']['output'] += 1
            else:
                tensor_analysis['layer_classification']['other'] += 1
            
            # Collecte pour analyse de distribution
            tensor_analysis['size_distribution'].append({
                'name': name,
                'size_mb': info['size_mb'],
                'shape': info['shape'],
                'type': info['type_name'],
                'is_quantized': info['is_quantized']
            })
        
        # Analyse de quantification détaillée
        quantization_stats = defaultdict(lambda: {'count': 0, 'total_size': 0, 'total_elements': 0})
        
        for info in self.tensors_info.values():
            qtype = info['type_name']
            quantization_stats[qtype]['count'] += 1
            quantization_stats[qtype]['total_size'] += info['size_bytes']
            quantization_stats[qtype]['total_elements'] += info['element_count']
        
        # Calcul des ratios de compression
        for qtype, stats in quantization_stats.items():
            if stats['total_elements'] > 0:
                # Comparaison avec F32 (4 bytes par élément)
                f32_size = stats['total_elements'] * 4
                compression_ratio = f32_size / stats['total_size'] if stats['total_size'] > 0 else 1
                quantization_stats[qtype]['compression_ratio'] = compression_ratio
                quantization_stats[qtype]['bits_per_element'] = (stats['total_size'] * 8) / stats['total_elements']
        
        tensor_analysis['quantization_analysis'] = dict(quantization_stats)
        
        # Top tenseurs par taille
        tensor_analysis['largest_tensors'] = sorted(
            tensor_analysis['size_distribution'],
            key=lambda x: x['size_mb'],
            reverse=True
        )[:15]
        
        # Analyse des formes
        shape_patterns = defaultdict(int)
        dimension_stats = {'1d': 0, '2d': 0, '3d': 0, '4d+': 0}
        
        for info in self.tensors_info.values():
            shape = info['shape']
            ndim = len(shape)
            
            if ndim == 1:
                dimension_stats['1d'] += 1
                shape_patterns[f"1D({shape[0]})"] += 1
            elif ndim == 2:
                dimension_stats['2d'] += 1
                shape_patterns[f"2D({shape[0]}×{shape[1]})"] += 1
            elif ndim == 3:
                dimension_stats['3d'] += 1
            else:
                dimension_stats['4d+'] += 1
        
        tensor_analysis['shape_analysis'] = {
            'dimension_distribution': dict(dimension_stats),
            'common_shapes': dict(sorted(shape_patterns.items(), key=lambda x: x[1], reverse=True)[:10])
        }
        
        # Analyse de compression globale
        total_uncompressed = sum(info['element_count'] * 4 for info in self.tensors_info.values())  # F32 baseline
        total_compressed = sum(info['size_bytes'] for info in self.tensors_info.values())
        
        tensor_analysis['compression_analysis'] = {
            'global_compression_ratio': total_uncompressed / total_compressed if total_compressed > 0 else 1,
            'space_saved_mb': (total_uncompressed - total_compressed) / (1024 * 1024),
            'average_bits_per_element': (total_compressed * 8) / sum(info['element_count'] for info in self.tensors_info.values()) if self.tensors_info else 0
        }
        
        # Échantillonnage détaillé si possible
        sample_tensors = list(self.tensors_info.keys())[:sample_size]
        for tensor_name in sample_tensors:
            tensor_analysis['detailed_sample'][tensor_name] = self._analyze_single_tensor_detailed(tensor_name)
        
        self.analysis_results['tensors'] = tensor_analysis
        return tensor_analysis
    
    def _analyze_single_tensor_detailed(self, tensor_name: str) -> Dict[str, Any]:
        """Analyse détaillée d'un tenseur spécifique"""
        if tensor_name not in self.tensors_info:
            return {}
        
        info = self.tensors_info[tensor_name]
        
        analysis = {
            'basic_info': {
                'shape': info['shape'],
                'type': info['type_name'],
                'size_mb': info['size_mb'],
                'element_count': info['element_count'],
                'is_quantized': info['is_quantized']
            },
            'quantization_details': {},
            'memory_efficiency': {},
            'layer_role': self._classify_tensor_role(tensor_name)
        }
        
        # Détails de quantification
        if info['is_quantized']:
            type_info = GGML_TYPE_INFO.get(info['ggml_type'], {})
            f32_size = info['element_count'] * 4
            
            analysis['quantization_details'] = {
                'quantization_type': info['type_name'],
                'block_size': type_info.get('block_size', 1),
                'bytes_per_block': type_info.get('type_size', 1),
                'compression_ratio': f32_size / info['size_bytes'] if info['size_bytes'] > 0 else 1,
                'bits_per_element': (info['size_bytes'] * 8) / info['element_count'] if info['element_count'] > 0 else 0
            }
        
        # Efficacité mémoire
        analysis['memory_efficiency'] = {
            'density': info['element_count'] / info['size_bytes'] if info['size_bytes'] > 0 else 0,
            'overhead_ratio': 1 - (info['element_count'] * self._get_base_type_size(info['ggml_type'])) / info['size_bytes'] if info['size_bytes'] > 0 else 0
        }
        
        return analysis
    
    def _classify_tensor_role(self, tensor_name: str) -> str:
        """Classifie le rôle d'un tenseur basé sur son nom"""
        name_lower = tensor_name.lower()
        
        if any(pattern in name_lower for pattern in ['embed', 'token', 'wte']):
            return 'embedding'
        elif any(pattern in name_lower for pattern in ['q_proj', 'k_proj', 'v_proj', 'query', 'key', 'value']):
            return 'attention_projection'
        elif any(pattern in name_lower for pattern in ['o_proj', 'out_proj', 'attn.out']):
            return 'attention_output'
        elif any(pattern in name_lower for pattern in ['gate_proj', 'up_proj', 'down_proj']):
            return 'mlp_projection'
        elif any(pattern in name_lower for pattern in ['norm', 'ln']):
            return 'normalization'
        elif any(pattern in name_lower for pattern in ['head', 'lm_head', 'output']):
            return 'output_head'
        else:
            return 'other'
    
    def _get_base_type_size(self, ggml_type: int) -> int:
        """Retourne la taille de base d'un type GGML"""
        type_info = GGML_TYPE_INFO.get(ggml_type, {"type_size": 1})
        if type_info.get("block_size", 1) > 1:
            # Pour les types quantifiés, on estime la taille par élément
            return type_info["type_size"] / type_info["block_size"]
        return type_info["type_size"]
    
    def analyze_architecture(self) -> Dict[str, Any]:
        """
        Niveau 3: Analyse architecturale avancée
        """
        print("🏗️ Analyse architecturale en cours...")
        
        arch_analysis = {
            'reconstructed_architecture': {},
            'parameter_distribution': {},
            'layer_analysis': {},
            'attention_analysis': {},
            'quantization_strategy': {},
            'model_topology': {}
        }
        
        # Reconstruction de l'architecture depuis les métadonnées et tenseurs
        arch_analysis['reconstructed_architecture'] = self._reconstruct_architecture()
        
        # Distribution des paramètres
        arch_analysis['parameter_distribution'] = self._analyze_parameter_distribution()
        
        # Analyse des couches
        arch_analysis['layer_analysis'] = self._analyze_layers()
        
        # Analyse de l'attention
        arch_analysis['attention_analysis'] = self._analyze_attention_architecture()
        
        # Stratégie de quantification
        arch_analysis['quantization_strategy'] = self._analyze_quantization_strategy()
        
        # Topologie du modèle
        arch_analysis['model_topology'] = self._create_model_topology()
        
        self.analysis_results['architecture'] = arch_analysis
        return arch_analysis
    
    def _reconstruct_architecture(self) -> Dict[str, Any]:
        """Reconstruit l'architecture depuis les métadonnées et tenseurs"""
        
        architecture = {
            'model_type': 'unknown',
            'num_layers': 0,
            'hidden_size': 0,
            'num_attention_heads': 0,
            'num_key_value_heads': 0,
            'intermediate_size': 0,
            'vocab_size': 0,
            'context_length': 0,
            'rope_theta': 0,
            'architecture_specific': {}
        }
        
        # Extraction depuis les métadonnées
        if self.metadata:
            arch_name = self.metadata.get('general.architecture', {}).get('value', 'unknown')
            architecture['model_type'] = arch_name
            
            # Paramètres architecturaux spécifiques
            arch_prefix = arch_name + '.'
            for key, meta_info in self.metadata.items():
                if key.startswith(arch_prefix):
                    param_name = key[len(arch_prefix):]
                    value = meta_info['value']
                    
                    # Mapping des paramètres communs
                    if param_name == 'block_count':
                        architecture['num_layers'] = value
                    elif param_name == 'embedding_length':
                        architecture['hidden_size'] = value
                    elif param_name == 'attention.head_count':
                        architecture['num_attention_heads'] = value
                    elif param_name == 'attention.head_count_kv':
                        architecture['num_key_value_heads'] = value
                    elif param_name == 'feed_forward_length':
                        architecture['intermediate_size'] = value
                    elif param_name == 'context_length':
                        architecture['context_length'] = value
                    elif param_name == 'rope.freq_base':
                        architecture['rope_theta'] = value
                    else:
                        architecture['architecture_specific'][param_name] = value
            
            # Taille du vocabulaire depuis les métadonnées du tokenizer
            if 'tokenizer.ggml.tokens' in self.metadata:
                tokens = self.metadata['tokenizer.ggml.tokens']['value']
                if isinstance(tokens, list):
                    architecture['vocab_size'] = len(tokens)
        
        # Validation et correction depuis les tenseurs
        self._validate_architecture_from_tensors(architecture)
        
        return architecture
    
    def _validate_architecture_from_tensors(self, architecture: Dict[str, Any]):
        """Valide et corrige l'architecture en analysant les tenseurs"""
        
        # Recherche des embeddings pour la taille du vocabulaire
        for name, info in self.tensors_info.items():
            if 'embed' in name.lower() or 'token' in name.lower():
                shape = info['shape']
                if len(shape) >= 2:
                    # Généralement [vocab_size, hidden_size]
                    vocab_candidate = max(shape)
                    hidden_candidate = min(shape)
                    
                    if architecture['vocab_size'] == 0:
                        architecture['vocab_size'] = vocab_candidate
                    if architecture['hidden_size'] == 0:
                        architecture['hidden_size'] = hidden_candidate
        
        # Détection du nombre de couches depuis les noms de tenseurs
        layer_pattern = re.compile(r'(?:layers?|blocks?)\.(\d+)\.')
        layers_found = set()
        
        for name in self.tensors_info.keys():
            match = layer_pattern.search(name)
            if match:
                layer_num = int(match.group(1))
                layers_found.add(layer_num)
        
        if layers_found and architecture['num_layers'] == 0:
            architecture['num_layers'] = max(layers_found) + 1
        
        # Détection des têtes d'attention depuis les poids
        if architecture['num_attention_heads'] == 0:
            for name, info in self.tensors_info.items():
                if 'attn' in name.lower() and 'weight' in name.lower():
                    shape = info['shape']
                    if len(shape) == 2 and architecture['hidden_size'] > 0:
                        # Estimation basée sur des diviseurs communs
                        hidden_size = architecture['hidden_size']
                        for heads in [8, 12, 16, 32, 64, 128]:
                            if hidden_size % heads == 0:
                                architecture['num_attention_heads'] = heads
                                break
    
    def _analyze_parameter_distribution(self) -> Dict[str, Any]:
        """Analyse la distribution des paramètres par composant"""
        
        distribution = {
            'by_component': defaultdict(int),
            'by_layer': defaultdict(int),
            'by_quantization': defaultdict(int),
            'total_parameters': 0,
            'efficiency_metrics': {}
        }
        
        for name, info in self.tensors_info.items():
            param_count = info['element_count']
            distribution['total_parameters'] += param_count
            
            # Classification par composant
            component = self._classify_tensor_role(name)
            distribution['by_component'][component] += param_count
            
            # Classification par couche
            layer_match = re.search(r'(?:layers?|blocks?)\.(\d+)\.', name)
            if layer_match:
                layer_num = int(layer_match.group(1))
                distribution['by_layer'][f'layer_{layer_num}'] += param_count
            else:
                distribution['by_layer']['global'] += param_count
            
            # Classification par type de quantification
            distribution['by_quantization'][info['type_name']] += param_count
        
        # Métriques d'efficacité
        total = distribution['total_parameters']
        if total > 0:
            distribution['efficiency_metrics'] = {
                'embedding_ratio': distribution['by_component']['embedding'] / total,
                'attention_ratio': (distribution['by_component']['attention_projection'] + 
                                  distribution['by_component']['attention_output']) / total,
                'mlp_ratio': distribution['by_component']['mlp_projection'] / total,
                'normalization_ratio': distribution['by_component']['normalization'] / total,
                'params_per_layer': total / max(1, len([k for k in distribution['by_layer'].keys() if k.startswith('layer_')]))
            }
        
        return distribution
    
    def _analyze_layers_bogus(self) -> Dict[str, Any]:
        """Analyse détaillée des couches"""
        
        layer_analysis = {
            'layer_structure': {},
            'layer_similarities': {},
            'bottlenecks': [],
            'parameter_progression': {}
        }
        
        # Regroupement des tenseurs par couche
        layers = defaultdict(list)
        layer_pattern = re.compile(r'(?:layers?|blocks?)\.(\d+)\.')
        
        for name, info in self.tensors_info.items():
            match = layer_pattern.search(name)
            if match:
                layer_num = int(match.group(1))
                layers[layer_num].append((name, info))
        
        # Analyse de chaque couche
        for layer_num, layer_tensors in layers.items():
            layer_info = {
                'tensor_count': len(layer_tensors),
                'total_parameters': sum(info['element_count'] for _, info in layer_tensors),
                'total_size_mb': sum(info['size_mb'] for _, info in layer_tensors),
                'components': {},
                'quantization_mix': defaultdict(int)
            }
            
            # Analyse des composants de la couche
            for name, info in layer_tensors:
                component = self._classify_tensor_role(name)
                if component not in layer_info['components']:
                    layer_info['components'][component] = {
                        'tensors': [],
                        'parameters': 0,
                        'size_mb': 0
                    }
                
                layer_info['components'][component]['tensors'].append(name)
                layer_info['components'][component]['parameters'] += info['element_count']
                layer_info['components'][component]['size_mb'] += info['size_mb']
                
                # Mix de quantification
                layer_info['quantization_mix'][info['type_name']] += 1
            
            layer_analysis['layer_structure'][f'layer_{layer_num}'] = layer_info
        
        # Analyse de la progression des paramètres
        if layers:
            sorted_layers = sorted(layers.keys())
            param_progression = []
            
            for layer_num in sorted_layers:
                layer_tensors = layers[layer_num]
                total_params = sum(info['element_count'] for _, info in layer_tensors)
                param_progression.append(total_params)
            
            layer_analysis['parameter_progression'] = {
                'layer_params': param_progression,
                'mean_params': np.mean(param_progression),
                'std_params': np.std(param_progression),
                'min_params': min(param_progression),
                'max_params': max(param_progression)
            }
        
        return layer_analysis
    
    
    def _analyze_layers(self) -> Dict[str, Any]:
        """Analyse détaillée des couches"""
    
        layer_analysis = {
            'layer_structure': {},
            'layer_similarities': {},
            'bottlenecks': [],
            'parameter_progression': {}
        }
    
        # Regroupement des tenseurs par couche
        layers = defaultdict(list)
        layer_pattern = re.compile(r'(?:layers?|blocks?)\.(\d+)\.')
    
        for name, info in self.tensors_info.items():
            match = layer_pattern.search(name)
            if match:
                layer_num = int(match.group(1))
                layers[layer_num].append((name, info))
    
        # Analyse de chaque couche
        for layer_num, layer_tensors in layers.items():
            layer_info = {
                'tensor_count': len(layer_tensors),
                'total_parameters': sum(info['element_count'] for _, info in layer_tensors),
                'total_size_mb': sum(info['size_mb'] for _, info in layer_tensors),
                'components': {},
                'quantization_mix': defaultdict(int)
            }
        
            # Analyse des composants de la couche
            for name, info in layer_tensors:
                component = self._classify_tensor_role(name)
                if component not in layer_info['components']:
                    layer_info['components'][component] = {
                        'tensors': [],
                        'parameters': 0,
                        'size_mb': 0
                    }
            
                layer_info['components'][component]['tensors'].append(name)
                layer_info['components'][component]['parameters'] += info['element_count']
                layer_info['components'][component]['size_mb'] += info['size_mb']
            
                # Mix de quantification
                layer_info['quantization_mix'][info['type_name']] += 1
        
            layer_analysis['layer_structure'][f'layer_{layer_num}'] = layer_info
    
        # Analyse de la progression des paramètres - TOUJOURS initialisée
        if layers:
            sorted_layers = sorted(layers.keys())
            param_progression = []
        
            for layer_num in sorted_layers:
                layer_tensors = layers[layer_num]
                total_params = sum(info['element_count'] for _, info in layer_tensors)
                param_progression.append(total_params)
        
            # Calcul des statistiques uniquement si on a des données
            if param_progression:
                layer_analysis['parameter_progression'] = {
                    'layer_params': param_progression,
                    'mean_params': float(np.mean(param_progression)),
                    'std_params': float(np.std(param_progression)),
                    'min_params': int(min(param_progression)),
                    'max_params': int(max(param_progression)),
                    'layer_count': len(param_progression)
                }
            else:
                # Valeurs par défaut si aucune donnée
                layer_analysis['parameter_progression'] = {
                    'layer_params': [],
                    'mean_params': 0.0,
                    'std_params': 0.0,
                    'min_params': 0,
                    'max_params': 0,
                    'layer_count': 0
                }
        else:
            # Aucune couche détectée - valeurs par défaut
            layer_analysis['parameter_progression'] = {
                'layer_params': [],
                'mean_params': 0.0,
                'std_params': 0.0,
                'min_params': 0,
                'max_params': 0,
                'layer_count': 0
            }
    
        return layer_analysis
    
    
    
    
    def _analyze_attention_architecture(self) -> Dict[str, Any]:
        """Analyse spécialisée de l'architecture d'attention"""
        
        attention_analysis = {
            'attention_patterns': {},
            'head_analysis': {},
            'attention_efficiency': {},
            'kv_cache_analysis': {}
        }
        
        # Recherche des tenseurs d'attention
        attention_tensors = {}
        for name, info in self.tensors_info.items():
            if any(pattern in name.lower() for pattern in ['attn', 'attention']):
                attention_tensors[name] = info
        
        # Classification des types d'attention
        attention_types = {
            'query': [],
            'key': [],
            'value': [],
            'output': [],
            'other': []
        }
        
        for name, info in attention_tensors.items():
            if any(pattern in name.lower() for pattern in ['q_proj', 'query']):
                attention_types['query'].append((name, info))
            elif any(pattern in name.lower() for pattern in ['k_proj', 'key']):
                attention_types['key'].append((name, info))
            elif any(pattern in name.lower() for pattern in ['v_proj', 'value']):
                attention_types['value'].append((name, info))
            elif any(pattern in name.lower() for pattern in ['o_proj', 'out_proj', 'output']):
                attention_types['output'].append((name, info))
            else:
                attention_types['other'].append((name, info))
        
        # Analyse des têtes d'attention
        if attention_types['query']:
            sample_q = attention_types['query'][0][1]  # Premier tenseur query
            shape = sample_q['shape']
            
            if len(shape) >= 2:
                # Estimation du nombre de têtes et de la dimension par tête
                total_dim = shape[-1] if len(shape) == 2 else shape[-2]
                
                # Depuis les métadonnées si disponible
                arch_info = self.analysis_results.get('architecture', {}).get('reconstructed_architecture', {})
                num_heads = arch_info.get('num_attention_heads', 0)
                
                if num_heads > 0:
                    head_dim = total_dim // num_heads
                    attention_analysis['head_analysis'] = {
                        'num_heads': num_heads,
                        'head_dimension': head_dim,
                        'total_attention_dim': total_dim,
                        'kv_heads': arch_info.get('num_key_value_heads', num_heads)
                    }
        
        # Analyse de l'efficacité de l'attention
        total_attention_params = sum(info['element_count'] for tensors in attention_types.values() for _, info in tensors)
        total_model_params = sum(info['element_count'] for info in self.tensors_info.values())
        
        attention_analysis['attention_efficiency'] = {
            'attention_param_ratio': total_attention_params / total_model_params if total_model_params > 0 else 0,
            'total_attention_params': total_attention_params,
            'attention_size_mb': sum(info['size_mb'] for tensors in attention_types.values() for _, info in tensors)
        }
        
        # Analyse du cache KV si GQA/MQA détecté
        arch_info = self.analysis_results.get('architecture', {}).get('reconstructed_architecture', {})
        num_heads = arch_info.get('num_attention_heads', 0)
        kv_heads = arch_info.get('num_key_value_heads', num_heads)
        
        if kv_heads != num_heads and kv_heads > 0:
            attention_analysis['kv_cache_analysis'] = {
                'is_grouped_query': True,
                'kv_reduction_ratio': num_heads / kv_heads,
                'cache_efficiency': f"{kv_heads}/{num_heads} heads for K/V"
            }
        
        attention_analysis['attention_patterns'] = attention_types
        
        return attention_analysis
    
    def _analyze_quantization_strategy(self) -> Dict[str, Any]:
        """Analyse la stratégie de quantification utilisée"""
        
        strategy_analysis = {
            'quantization_scheme': {},
            'layer_specific_quantization': {},
            'efficiency_analysis': {},
            'quality_vs_compression': {}
        }
        
        # Analyse du schéma global
        quantization_by_component = defaultdict(lambda: defaultdict(int))
        
        for name, info in self.tensors_info.items():
            component = self._classify_tensor_role(name)
            qtype = info['type_name']
            quantization_by_component[component][qtype] += 1
        
        strategy_analysis['quantization_scheme'] = dict(quantization_by_component)
        
        # Analyse spécifique par couche
        layer_quantization = defaultdict(lambda: defaultdict(int))
        layer_pattern = re.compile(r'(?:layers?|blocks?)\.(\d+)\.')
        
        for name, info in self.tensors_info.items():
            match = layer_pattern.search(name)
            layer_key = f"layer_{match.group(1)}" if match else "global"
            layer_quantization[layer_key][info['type_name']] += 1
        
        strategy_analysis['layer_specific_quantization'] = dict(layer_quantization)
        
        # Analyse d'efficacité
        total_compressed_size = sum(info['size_bytes'] for info in self.tensors_info.values())
        total_uncompressed_size = sum(info['element_count'] * 4 for info in self.tensors_info.values())  # F32 baseline
        
        strategy_analysis['efficiency_analysis'] = {
            'global_compression_ratio': total_uncompressed_size / total_compressed_size if total_compressed_size > 0 else 1,
            'space_saved_gb': (total_uncompressed_size - total_compressed_size) / (1024**3),
            'average_bits_per_weight': (total_compressed_size * 8) / sum(info['element_count'] for info in self.tensors_info.values())
        }
        
        # Analyse qualité vs compression par type
        type_analysis = {}
        for qtype in set(info['type_name'] for info in self.tensors_info.values()):
            type_tensors = [info for info in self.tensors_info.values() if info['type_name'] == qtype]
            
            if type_tensors:
                total_elements = sum(t['element_count'] for t in type_tensors)
                total_size = sum(t['size_bytes'] for t in type_tensors)
                f32_size = total_elements * 4
                
                type_analysis[qtype] = {
                    'tensor_count': len(type_tensors),
                    'compression_ratio': f32_size / total_size if total_size > 0 else 1,
                    'bits_per_element': (total_size * 8) / total_elements if total_elements > 0 else 0,
                    'quality_estimate': self._estimate_quantization_quality(qtype)
                }
        
        strategy_analysis['quality_vs_compression'] = type_analysis
        
        return strategy_analysis
    
    def _estimate_quantization_quality(self, qtype: str) -> str:
        """Estime la qualité d'une quantification donnée"""
        quality_map = {
            'F32': 'Excellent (pas de perte)',
            'F16': 'Très bon (perte minimale)',
            'Q8_0': 'Bon (perte faible)',
            'Q6_K': 'Bon (perte modérée)',
            'Q5_K': 'Moyen (perte visible)',
            'Q4_K': 'Moyen (perte notable)',
            'Q3_K': 'Faible (perte importante)',
            'Q2_K': 'Très faible (perte majeure)'
        }
        
        return quality_map.get(qtype, 'Qualité inconnue')
    
    def _create_model_topology(self) -> Dict[str, Any]:
        """Crée une représentation topologique du modèle"""
        
        topology = {
            'graph_structure': {},
            'data_flow': {},
            'critical_paths': [],
            'parallelization_opportunities': {}
        }
        
        # Construction d'un graphe simplifié
        G = nx.DiGraph()
        
        # Ajout des nœuds (couches)
        layer_pattern = re.compile(r'(?:layers?|blocks?)\.(\d+)\.')
        layers = set()
        
        for name in self.tensors_info.keys():
            match = layer_pattern.search(name)
            if match:
                layer_num = int(match.group(1))
                layers.add(layer_num)
                G.add_node(f"layer_{layer_num}")
        
        # Ajout des connexions séquentielles
        sorted_layers = sorted(layers)
        for i in range(len(sorted_layers) - 1):
            G.add_edge(f"layer_{sorted_layers[i]}", f"layer_{sorted_layers[i+1]}")
        
        # Ajout des nœuds globaux
        if any('embed' in name.lower() for name in self.tensors_info.keys()):
            G.add_node("embedding")
            if sorted_layers:
                G.add_edge("embedding", f"layer_{sorted_layers[0]}")
        
        if any('lm_head' in name.lower() or 'output' in name.lower() for name in self.tensors_info.keys()):
            G.add_node("output_head")
            if sorted_layers:
                G.add_edge(f"layer_{sorted_layers[-1]}", "output_head")
        
        topology['graph_structure'] = {
            'nodes': list(G.nodes()),
            'edges': list(G.edges()),
            'num_layers': len(layers),
            'is_sequential': len(G.edges()) == len(G.nodes()) - 1
        }
        
        # Analyse du flux de données
        topology['data_flow'] = {
            'input_nodes': [node for node in G.nodes() if G.in_degree(node) == 0],
            'output_nodes': [node for node in G.nodes() if G.out_degree(node) == 0],
            'intermediate_nodes': [node for node in G.nodes() if G.in_degree(node) > 0 and G.out_degree(node) > 0]
        }
        
        return topology
    
    def analyze_advanced_patterns(self) -> Dict[str, Any]:
        """
        Niveau 4: Analyse avancée des patterns et optimisations
        """
        print("🔬 Analyse avancée des patterns en cours...")
        
        advanced_analysis = {
            'quantization_patterns': {},
            'memory_access_patterns': {},
            'optimization_opportunities': [],
            'performance_estimation': {},
            'compatibility_analysis': {},
            'quality_assessment': {}
        }
        
        # Analyse des patterns de quantification
        advanced_analysis['quantization_patterns'] = self._analyze_quantization_patterns()
        
        # Analyse des patterns d'accès mémoire
        advanced_analysis['memory_access_patterns'] = self._analyze_memory_patterns()
        
        # Opportunités d'optimisation
        advanced_analysis['optimization_opportunities'] = self._find_optimization_opportunities()
        
        # Estimation des performances
        advanced_analysis['performance_estimation'] = self._estimate_model_performance()
        
        # Analyse de compatibilité
        advanced_analysis['compatibility_analysis'] = self._analyze_compatibility()
        
        # Évaluation de la qualité
        advanced_analysis['quality_assessment'] = self._assess_model_quality()
        
        self.analysis_results['advanced'] = advanced_analysis
        return advanced_analysis
    
    def _analyze_quantization_patterns(self) -> Dict[str, Any]:
        """Analyse les patterns de quantification avancés"""
        
        patterns = {
            'mixed_precision_analysis': {},
            'quantization_transitions': {},
            'outlier_detection': {},
            'compression_efficiency': {}
        }
        
        # Analyse de la précision mixte
        precision_by_layer = defaultdict(lambda: defaultdict(int))
        layer_pattern = re.compile(r'(?:layers?|blocks?)\.(\d+)\.')
        
        for name, info in self.tensors_info.items():
            match = layer_pattern.search(name)
            layer_key = f"layer_{match.group(1)}" if match else "global"
            component = self._classify_tensor_role(name)
            
            precision_by_layer[layer_key][info['type_name']] += 1
        
        # Détection des patterns de transition
        transitions = []
        sorted_layers = sorted([k for k in precision_by_layer.keys() if k.startswith('layer_')],
                              key=lambda x: int(x.split('_')[1]))
        
        for i in range(len(sorted_layers) - 1):
            current_layer = precision_by_layer[sorted_layers[i]]
            next_layer = precision_by_layer[sorted_layers[i + 1]]
            
            if current_layer != next_layer:
                transitions.append({
                    'from_layer': sorted_layers[i],
                    'to_layer': sorted_layers[i + 1],
                    'change': {
                        'from': dict(current_layer),
                        'to': dict(next_layer)
                    }
                })
        
        patterns['quantization_transitions'] = transitions
        patterns['mixed_precision_analysis'] = dict(precision_by_layer)
        
        # Détection d'outliers en termes de taille
        sizes = [info['size_mb'] for info in self.tensors_info.values()]
        if sizes:
            q75, q25 = np.percentile(sizes, [75, 25])
            iqr = q75 - q25
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            
            outliers = []
            for name, info in self.tensors_info.items():
                if info['size_mb'] < lower_bound or info['size_mb'] > upper_bound:
                    outliers.append({
                        'name': name,
                        'size_mb': info['size_mb'],
                        'type': 'oversized' if info['size_mb'] > upper_bound else 'undersized',
                        'deviation_factor': info['size_mb'] / np.median(sizes)
                    })
            
            patterns['outlier_detection'] = {
                'outliers': outliers,
                'outlier_count': len(outliers),
                'size_statistics': {
                    'median': float(np.median(sizes)),
                    'q25': float(q25),
                    'q75': float(q75),
                    'iqr': float(iqr)
                }
            }
        
        return patterns
    
    def _analyze_memory_patterns(self) -> Dict[str, Any]:
        """Analyse les patterns d'accès mémoire"""
        
        memory_patterns = {
            'tensor_layout': {},
            'cache_efficiency': {},
            'memory_hierarchy': {},
            'bandwidth_requirements': {}
        }
        
        # Analyse de la disposition des tenseurs
        tensor_offsets = [(name, info['offset'], info['size_bytes']) 
                         for name, info in self.tensors_info.items()]
        tensor_offsets.sort(key=lambda x: x[1])  # Tri par offset
        
        # Détection de patterns de disposition
        gaps = []
        for i in range(len(tensor_offsets) - 1):
            current_end = tensor_offsets[i][1] + tensor_offsets[i][2]
            next_start = tensor_offsets[i + 1][1]
            gap = next_start - current_end
            
            if gap > 0:
                gaps.append({
                    'after_tensor': tensor_offsets[i][0],
                    'before_tensor': tensor_offsets[i + 1][0],
                    'gap_bytes': gap
                })
        
        memory_patterns['tensor_layout'] = {
            'total_gaps': len(gaps),
            'total_gap_bytes': sum(g['gap_bytes'] for g in gaps),
            'largest_gap': max((g['gap_bytes'] for g in gaps), default=0),
            'fragmentation_ratio': sum(g['gap_bytes'] for g in gaps) / self.file_size if gaps else 0
        }
        
        # Analyse de l'efficacité du cache
        sequential_access_score = 0
        for i in range(len(tensor_offsets) - 1):
            current_end = tensor_offsets[i][1] + tensor_offsets[i][2]
            next_start = tensor_offsets[i + 1][1]
            
            if next_start == current_end:  # Accès séquentiel parfait
                sequential_access_score += 1
        
        memory_patterns['cache_efficiency'] = {
            'sequential_access_ratio': sequential_access_score / max(len(tensor_offsets) - 1, 1),
            'fragmentation_level': 'low' if len(gaps) < 5 else 'medium' if len(gaps) < 20 else 'high'
        }
        
        return memory_patterns
    
    def _find_optimization_opportunities(self) -> List[Dict[str, Any]]:
        """Identifie les opportunités d'optimisation"""
        
        opportunities = []
        
        # Analyse de la distribution des types de quantification
        type_counts = defaultdict(int)
        type_sizes = defaultdict(float)
        
        for info in self.tensors_info.values():
            type_counts[info['type_name']] += 1
            type_sizes[info['type_name']] += info['size_mb']
        
        # Opportunité: Quantification plus agressive
        if 'F32' in type_counts or 'F16' in type_counts:
            unquantized_size = type_sizes.get('F32', 0) + type_sizes.get('F16', 0)
            opportunities.append({
                'type': 'aggressive_quantization',
                'priority': 'high' if unquantized_size > 100 else 'medium',
                'description': f'Quantifier les tenseurs F32/F16 ({unquantized_size:.1f} MB)',
                'potential_reduction': f'{unquantized_size * 0.5:.1f} MB avec Q4_K',
                'risk': 'Perte de qualité modérée'
            })
        
        # Opportunité: Uniformisation de la quantification
        unique_types = len(set(info['type_name'] for info in self.tensors_info.values() if info['is_quantized']))
        if unique_types > 3:
            opportunities.append({
                'type': 'quantization_uniformization',
                'priority': 'medium',
                'description': f'Trop de types de quantification différents ({unique_types})',
                'potential_benefit': 'Simplification du code et optimisation hardware',
                'recommendation': 'Standardiser sur Q4_K ou Q5_K'
            })
        
        # Opportunité: Optimisation des embeddings
        embedding_size = sum(info['size_mb'] for name, info in self.tensors_info.items() 
                           if 'embed' in name.lower())
        total_size = sum(info['size_mb'] for info in self.tensors_info.values())
        
        if embedding_size / total_size > 0.15:  # Plus de 15% du modèle
            opportunities.append({
                'type': 'embedding_optimization',
                'priority': 'medium',
                'description': f'Embeddings représentent {embedding_size/total_size*100:.1f}% du modèle',
                'potential_techniques': ['Vocabulary pruning', 'Embedding compression', 'Shared embeddings'],
                'potential_reduction': f'{embedding_size * 0.3:.1f} MB'
            })
        
        # Opportunité: Détection de redondance potentielle
        layer_analysis = self.analysis_results.get('architecture', {}).get('layer_analysis', {})
        if 'parameter_progression' in layer_analysis:
            param_progression = layer_analysis['parameter_progression']
            std_params = param_progression.get('std_params', 0)
            mean_params = param_progression.get('mean_params', 1)
            
            if std_params / mean_params < 0.1:  # Très peu de variation entre couches
                opportunities.append({
                    'type': 'layer_sharing',
                    'priority': 'low',
                    'description': 'Couches très similaires détectées',
                    'potential_technique': 'Weight sharing ou layer sharing',
                    'risk': 'Perte potentielle de capacité du modèle'
                })
        
        return opportunities
    
    def _estimate_model_performance(self) -> Dict[str, Any]:
        """Estime les performances théoriques du modèle"""
        
        performance = {
            'inference_metrics': {},
            'memory_requirements': {},
            'throughput_estimation': {},
            'hardware_compatibility': {}
        }
        
        # Calcul des métriques d'inférence
        total_params = sum(info['element_count'] for info in self.tensors_info.values())
        total_size_mb = sum(info['size_mb'] for info in self.tensors_info.values())
        
        # Estimation de la latence (approximative)
        arch_info = self.analysis_results.get('architecture', {}).get('reconstructed_architecture', {})
        num_layers = arch_info.get('num_layers', 0)
        hidden_size = arch_info.get('hidden_size', 0)
        context_length = arch_info.get('context_length', 2048)
        
        if num_layers > 0 and hidden_size > 0:
            # Calcul approximatif des FLOPS par token
            flops_per_token = (
                # Attention: O(seq_len * hidden_size^2)
                context_length * hidden_size * hidden_size +
                # MLP: O(seq_len * hidden_size * intermediate_size)  
                context_length * hidden_size * arch_info.get('intermediate_size', hidden_size * 4)
            ) * num_layers
            
            performance['inference_metrics'] = {
                'params_billion': total_params / 1e9,
                'model_size_gb': total_size_mb / 1024,
                'flops_per_token': flops_per_token,
                'estimated_tokens_per_second': self._estimate_tokens_per_second(flops_per_token, total_size_mb)
            }
        
        # Exigences mémoire
        performance['memory_requirements'] = {
            'model_memory_gb': total_size_mb / 1024,
            'kv_cache_gb': self._estimate_kv_cache_size(arch_info, context_length),
            'total_inference_memory_gb': (total_size_mb / 1024) * 1.2,  # +20% pour les activations
            'batch_size_1_memory_gb': (total_size_mb / 1024) * 1.5
        }
        
        # Compatibilité hardware
        performance['hardware_compatibility'] = {
            'cpu_inference': 'Possible' if total_size_mb < 16384 else 'Difficile',  # < 16GB
            'mobile_inference': 'Possible' if total_size_mb < 4096 else 'Non recommandé',  # < 4GB
            'edge_inference': 'Possible' if total_size_mb < 1024 else 'Limité',  # < 1GB
            'recommended_gpu_memory_gb': max(8, (total_size_mb / 1024) * 1.5)
        }
        
        return performance
    
    def _estimate_tokens_per_second(self, flops_per_token: int, model_size_mb: float) -> Dict[str, float]:
        """Estime les tokens par seconde sur différents hardware"""
        
        # Estimations très approximatives basées sur les performances typiques
        hardware_specs = {
            'cpu_high_end': {'flops_per_sec': 1e12, 'memory_bandwidth_gb_s': 100},
            'gpu_consumer': {'flops_per_sec': 20e12, 'memory_bandwidth_gb_s': 500},
            'gpu_datacenter': {'flops_per_sec': 100e12, 'memory_bandwidth_gb_s': 1000}
        }
        
        estimates = {}
        for hardware, specs in hardware_specs.items():
            # Limitation par compute
            compute_limited_tps = specs['flops_per_sec'] / flops_per_token
            
            # Limitation par bande passante mémoire (approximation)
            memory_limited_tps = (specs['memory_bandwidth_gb_s'] * 1024) / model_size_mb
            
            # Le facteur limitant
            estimated_tps = min(compute_limited_tps, memory_limited_tps)
            estimates[hardware] = round(estimated_tps, 2)
        
        return estimates
    
    def _estimate_kv_cache_size(self, arch_info: Dict, context_length: int) -> float:
        """Estime la taille du cache KV"""
        
        num_layers = arch_info.get('num_layers', 0)
        hidden_size = arch_info.get('hidden_size', 0)
        num_heads = arch_info.get('num_attention_heads', 0)
        kv_heads = arch_info.get('num_key_value_heads', num_heads)
        
        if num_layers > 0 and hidden_size > 0 and kv_heads > 0:
            # 2 pour K et V, 2 pour FP16, batch size = 1
            kv_cache_bytes = 2 * num_layers * context_length * hidden_size * (kv_heads / max(num_heads, 1)) * 2
            return kv_cache_bytes / (1024**3)  # GB
        
        return 0.0
    
    def _analyze_compatibility(self) -> Dict[str, Any]:
        """Analyse la compatibilité avec différents frameworks"""
        
        compatibility = {
            'framework_support': {},
            'quantization_support': {},
            'hardware_acceleration': {},
            'deployment_readiness': {}
        }
        
        # Support par framework
        gguf_version = self.header_info.get('version', 0)
        quantization_types = set(info['type_name'] for info in self.tensors_info.values())
        
        compatibility['framework_support'] = {
            'llama_cpp': 'Full' if gguf_version >= 2 else 'Limited',
            'ggml': 'Full',
            'transformers': 'Via conversion',
            'onnx': 'Via conversion (complex)',
            'tensorrt': 'Via conversion (quantization may be lost)'
        }
        
        # Support de la quantification
        advanced_quants = {'Q2_K', 'Q3_K', 'Q4_K', 'Q5_K', 'Q6_K', 'Q8_K'}
        has_advanced = bool(quantization_types.intersection(advanced_quants))
        
        compatibility['quantization_support'] = {
            'has_k_quants': has_advanced,
            'supported_types': list(quantization_types),
            'cpu_optimized': 'Q4_0' in quantization_types or 'Q8_0' in quantization_types,
            'gpu_optimized': has_advanced
        }
        
        # Accélération matérielle
        model_size_gb = sum(info['size_mb'] for info in self.tensors_info.values()) / 1024
        
        compatibility['hardware_acceleration'] = {
            'cpu_avx2': 'Recommended',
            'cpu_avx512': 'Optimal' if model_size_gb < 32 else 'Good',
            'gpu_cuda': 'Good' if has_advanced else 'Basic',
            'gpu_rocm': 'Limited',
            'gpu_metal': 'Good' if has_advanced else 'Basic',
            'mobile_neon': 'Possible' if model_size_gb < 4 else 'Not recommended'
        }
        
        # Préparation au déploiement
        compatibility['deployment_readiness'] = {
            'production_ready': model_size_gb < 64 and has_advanced,
            'edge_deployment': model_size_gb < 2,
            'cloud_deployment': True,
            'containerization': 'Ready',
            'serving_frameworks': ['llama.cpp', 'vLLM (via conversion)', 'TGI (via conversion)']
        }
        
        return compatibility
    
    def _assess_model_quality(self) -> Dict[str, Any]:
        """Évalue la qualité globale du modèle"""
        
        quality = {
            'quantization_quality': {},
            'architecture_coherence': {},
            'optimization_level': {},
            'overall_score': {}
        }
        
        # Évaluation de la qualité de quantification
        quantization_types = [info['type_name'] for info in self.tensors_info.values()]
        type_counts = defaultdict(int)
        for qtype in quantization_types:
            type_counts[qtype] += 1
        
        # Score basé sur les types de quantification utilisés
        quality_scores = {
            'F32': 100, 'F16': 95, 'Q8_0': 90, 'Q6_K': 85, 'Q5_K': 80,
            'Q4_K': 75, 'Q4_0': 70, 'Q3_K': 60, 'Q2_K': 45
        }
        
        weighted_quality = 0
        total_tensors = len(quantization_types)
        
        for qtype, count in type_counts.items():
            score = quality_scores.get(qtype, 50)
            weighted_quality += (score * count) / total_tensors
        
        quality['quantization_quality'] = {
            'weighted_score': round(weighted_quality, 1),
            'grade': self._score_to_grade(weighted_quality),
            'dominant_type': max(type_counts.items(), key=lambda x: x[1])[0],
            'uniformity': len(type_counts) <= 3  # Bon si <= 3 types différents
        }
        
        # Cohérence architecturale
        arch_analysis = self.analysis_results.get('architecture', {})
        arch_info = arch_analysis.get('reconstructed_architecture', {})
        
        coherence_score = 0
        if arch_info.get('num_layers', 0) > 0:
            coherence_score += 25
        if arch_info.get('hidden_size', 0) > 0:
            coherence_score += 25
        if arch_info.get('num_attention_heads', 0) > 0:
            coherence_score += 25
        if arch_info.get('vocab_size', 0) > 0:
            coherence_score += 25
        
        quality['architecture_coherence'] = {
            'score': coherence_score,
            'grade': self._score_to_grade(coherence_score),
            'complete_metadata': coherence_score == 100
        }
        
        # Niveau d'optimisation
        optimization_score = 0
        
        # +30 si quantification présente
        if any(info['is_quantized'] for info in self.tensors_info.values()):
            optimization_score += 30
        
        # +20 si utilisation de K-quants modernes
        advanced_quants = {'Q2_K', 'Q3_K', 'Q4_K', 'Q5_K', 'Q6_K', 'Q8_K'}
        if any(info['type_name'] in advanced_quants for info in self.tensors_info.values()):
            optimization_score += 20
        
        # +20 si taille optimisée (compression ratio > 2)
        total_uncompressed = sum(info['element_count'] * 4 for info in self.tensors_info.values())
        total_compressed = sum(info['size_bytes'] for info in self.tensors_info.values())
        compression_ratio = total_uncompressed / total_compressed if total_compressed > 0 else 1
        
        if compression_ratio > 2:
            optimization_score += 20
        
        # +15 si structure mémoire optimisée
        memory_analysis = self.analysis_results.get('advanced', {}).get('memory_access_patterns', {})
        if memory_analysis.get('cache_efficiency', {}).get('sequential_access_ratio', 0) > 0.8:
            optimization_score += 15
        
        # +15 si quantification uniforme
        if len(set(info['type_name'] for info in self.tensors_info.values())) <= 3:
            optimization_score += 15
        
        quality['optimization_level'] = {
            'score': min(optimization_score, 100),
            'grade': self._score_to_grade(min(optimization_score, 100)),
            'compression_ratio': round(compression_ratio, 2)
        }
        
        # Score global
        overall = (weighted_quality * 0.4 + coherence_score * 0.3 + min(optimization_score, 100) * 0.3)
        
        quality['overall_score'] = {
            'score': round(overall, 1),
            'grade': self._score_to_grade(overall),
            'recommendation': self._get_quality_recommendation(overall)
        }
        
        return quality
    
    def _score_to_grade(self, score: float) -> str:
        """Convertit un score en note littérale"""
        if score >= 90:
            return 'A+'
        elif score >= 80:
            return 'A'
        elif score >= 70:
            return 'B'
        elif score >= 60:
            return 'C'
        elif score >= 50:
            return 'D'
        else:
            return 'F'
    
    def _get_quality_recommendation(self, score: float) -> str:
        """Génère une recommandation basée sur le score"""
        if score >= 85:
            return "Excellent modèle optimisé, prêt pour la production"
        elif score >= 75:
            return "Bon modèle, quelques optimisations possibles"
        elif score >= 65:
            return "Modèle correct, optimisations recommandées"
        elif score >= 50:
            return "Modèle nécessitant des améliorations significatives"
        else:
            return "Modèle de faible qualité, reconstruction recommandée"
    
    def generate_report(self, save_path: Optional[str] = None) -> str:
        """
        Génère un rapport basique de l'analyse
        """
        report = []
        report.append("=" * 60)
        report.append("RAPPORT D'ANALYSE - MODÈLE GGUF")
        report.append("=" * 60)
        report.append("")
        
        # Section structure
        if 'structure' in self.analysis_results:
            struct = self.analysis_results['structure']
            report.append("📁 STRUCTURE DU MODÈLE")
            report.append("-" * 30)
            report.append(f"Fichier: {self.model_path.name}")
            report.append(f"Taille: {struct['file_size_mb']} MB")
            report.append(f"Version GGUF: {struct['gguf_version']}")
            report.append(f"Architecture: {struct['model_info']['architecture']}")
            report.append(f"Nombre de tenseurs: {struct['tensor_count']}")
            report.append("")
            
            # Résumé de quantification
            quant_summary = struct['quantization_summary']
            report.append("Quantification:")
            for qtype, count in quant_summary['types_distribution'].items():
                report.append(f"  • {qtype}: {count} tenseurs")
            report.append(f"Tenseurs quantifiés: {quant_summary['quantized_tensors']}")
            report.append(f"Tenseurs non-quantifiés: {quant_summary['unquantized_tensors']}")
            report.append("")
        
        # Section tenseurs
        if 'tensors' in self.analysis_results:
            tensors = self.analysis_results['tensors']
            report.append("🔬 ANALYSE DES TENSEURS")
            report.append("-" * 30)
            
            report.append("Classification des couches:")
            for layer_type, count in tensors['layer_classification'].items():
                report.append(f"  • {layer_type}: {count} tenseurs")
            report.append("")
            
            # Compression globale
            compression = tensors['compression_analysis']
            report.append(f"Ratio de compression global: {compression['global_compression_ratio']:.2f}x")
            report.append(f"Espace économisé: {compression['space_saved_mb']:.1f} MB")
            report.append(f"Bits moyens par élément: {compression['average_bits_per_element']:.2f}")
            report.append("")
            
            report.append("Top 5 des plus gros tenseurs:")
            for i, tensor_info in enumerate(tensors['largest_tensors'][:5], 1):
                report.append(f"  {i}. {tensor_info['name']}: {tensor_info['size_mb']:.2f} MB ({tensor_info['type']})")
            report.append("")
        
        # Section architecture
        if 'architecture' in self.analysis_results:
            arch = self.analysis_results['architecture']['reconstructed_architecture']
            report.append("🏗️ ARCHITECTURE RECONSTRUITE")
            report.append("-" * 30)
            report.append(f"Type: {arch['model_type']}")
            report.append(f"Couches: {arch['num_layers']}")
            report.append(f"Dimension cachée: {arch['hidden_size']}")
            report.append(f"Têtes d'attention: {arch['num_attention_heads']}")
            report.append(f"Taille vocabulaire: {arch['vocab_size']}")
            report.append(f"Longueur contexte: {arch['context_length']}")
            report.append("")
        
        # Section analyse avancée
        if 'advanced' in self.analysis_results:
            advanced = self.analysis_results['advanced']
            
            # Opportunités d'optimisation
            opportunities = advanced.get('optimization_opportunities', [])
            if opportunities:
                report.append("💡 OPPORTUNITÉS D'OPTIMISATION")
                report.append("-" * 30)
                
                high_priority = [o for o in opportunities if o.get('priority') == 'high']
                if high_priority:
                    report.append("🔴 Priorité élevée:")
                    for opp in high_priority:
                        report.append(f"  • {opp['description']}")
                        if 'potential_reduction' in opp:
                            report.append(f"    Réduction: {opp['potential_reduction']}")
                    report.append("")
                
                medium_priority = [o for o in opportunities if o.get('priority') == 'medium']
                if medium_priority:
                    report.append("🟡 Priorité moyenne:")
                    for opp in medium_priority[:3]:  # Limite à 3
                        report.append(f"  • {opp['description']}")
                    report.append("")
            
            # Évaluation de qualité
            quality = advanced.get('quality_assessment', {})
            if quality:
                overall = quality.get('overall_score', {})
                report.append("📊 ÉVALUATION DE QUALITÉ")
                report.append("-" * 30)
                report.append(f"Score global: {overall.get('score', 0)}/100 ({overall.get('grade', 'N/A')})")
                report.append(f"Recommandation: {overall.get('recommendation', 'N/A')}")
                report.append("")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"📄 Rapport sauvegardé: {save_path}")
        
        return report_text
    
    def generate_comprehensive_report(self, save_path: Optional[str] = None) -> str:
        """
        Génère un rapport complet incluant tous les niveaux d'analyse
        """
        report = []
        report.append("=" * 80)
        report.append("RAPPORT COMPLET D'ANALYSE - MODÈLE GGUF")
        report.append("=" * 80)
        report.append("")
        
        # Informations générales
        report.append("📋 INFORMATIONS GÉNÉRALES")
        report.append("-" * 40)
        report.append(f"Fichier: {self.model_path.name}")
        report.append(f"Chemin: {self.model_path}")
        report.append(f"Taille fichier: {self.file_size / (1024**2):.1f} MB")
        report.append("")
        
        # Section structure (Niveau 1)
        if 'structure' in self.analysis_results:
            struct = self.analysis_results['structure']
            report.append("📁 1. ANALYSE STRUCTURELLE")
            report.append("-" * 40)
            
            model_info = struct['model_info']
            report.append(f"Nom du modèle: {model_info['name']}")
            report.append(f"Architecture: {model_info['architecture']}")
            report.append(f"Type de fichier: {model_info['file_type']}")
            report.append(f"Version quantification: {model_info['quantization_version']}")
            report.append(f"Version GGUF: {struct['gguf_version']}")
            report.append("")
            
            report.append(f"Tenseurs: {struct['tensor_count']}")
            report.append(f"Métadonnées: {struct['metadata_count']}")
            report.append("")
            
            # Résumé quantification détaillé
            quant = struct['quantization_summary']
            report.append("Distribution de la quantification:")
            for qtype, count in sorted(quant['types_distribution'].items()):
                percentage = (count / struct['tensor_count']) * 100
                report.append(f"  • {qtype}: {count} tenseurs ({percentage:.1f}%)")
            
            report.append("")
            report.append(f"Taille modèle: {quant['total_model_size_mb']:.1f} MB")
            report.append(f"Tenseurs quantifiés: {quant['quantized_tensors']}")
            report.append(f"Tenseurs full precision: {quant['unquantized_tensors']}")
            report.append("")
        
        # Section tenseurs (Niveau 2)
        if 'tensors' in self.analysis_results:
            tensors = self.analysis_results['tensors']
            report.append("🔬 2. ANALYSE DES TENSEURS")
            report.append("-" * 40)
            
            # Classification des couches
            report.append("Types de couches:")
            total_classified = sum(tensors['layer_classification'].values())
            for layer_type, count in sorted(tensors['layer_classification'].items()):
                percentage = (count / total_classified) * 100
                report.append(f"  • {layer_type}: {count} tenseurs ({percentage:.1f}%)")
            report.append("")
            
            # Analyse de compression
            compression = tensors['compression_analysis']
            report.append("Analyse de compression:")
            report.append(f"  • Ratio global: {compression['global_compression_ratio']:.2f}x")
            report.append(f"  • Espace économisé: {compression['space_saved_mb']:.1f} MB")
            report.append(f"  • Bits par élément: {compression['average_bits_per_element']:.2f}")
            report.append("")
            
            # Analyse détaillée de quantification
            report.append("Détails par type de quantification:")
            for qtype, details in tensors['quantization_analysis'].items():
                if isinstance(details, dict) and 'compression_ratio' in details:
                    report.append(f"  • {qtype}:")
                    report.append(f"    Tenseurs: {details['count']}")
                    report.append(f"    Compression: {details['compression_ratio']:.2f}x")
                    report.append(f"    Bits/élément: {details['bits_per_element']:.2f}")
            report.append("")
            
            # Top tenseurs
            report.append("Plus gros tenseurs:")
            for i, tensor in enumerate(tensors['largest_tensors'][:8], 1):
                report.append(f"  {i}. {tensor['name']}")
                report.append(f"     Taille: {tensor['size_mb']:.2f} MB, Type: {tensor['type']}")
                report.append(f"     Shape: {tensor['shape']}")
            report.append("")
        
        # Section architecture (Niveau 3)
        if 'architecture' in self.analysis_results:
            arch_analysis = self.analysis_results['architecture']
            report.append("🏗️ 3. ANALYSE ARCHITECTURALE")
            report.append("-" * 40)
            
            # Architecture reconstruite
            arch = arch_analysis['reconstructed_architecture']
            report.append("Architecture reconstruite:")
            report.append(f"  • Type: {arch['model_type']}")
            report.append(f"  • Nombre de couches: {arch['num_layers']}")
            report.append(f"  • Dimension cachée: {arch['hidden_size']}")
            report.append(f"  • Têtes d'attention: {arch['num_attention_heads']}")
            
            if arch['num_key_value_heads'] != arch['num_attention_heads']:
                report.append(f"  • Têtes KV: {arch['num_key_value_heads']} (GQA activé)")
            
            report.append(f"  • Taille intermédiaire: {arch['intermediate_size']}")
            report.append(f"  • Vocabulaire: {arch['vocab_size']} tokens")
            report.append(f"  • Contexte max: {arch['context_length']} tokens")
            
            if arch['rope_theta'] > 0:
                report.append(f"  • RoPE theta: {arch['rope_theta']}")
            
            report.append("")
            
            # Distribution des paramètres
            param_dist = arch_analysis['parameter_distribution']
            report.append("Distribution des paramètres:")
            total_params = param_dist['total_parameters']
            report.append(f"  • Total: {total_params:,} paramètres ({total_params/1e9:.2f}B)")
            
            for component, count in param_dist['by_component'].items():
                if count > 0:
                    percentage = (count / total_params) * 100
                    report.append(f"  • {component}: {count:,} ({percentage:.1f}%)")
            report.append("")
            
            # Stratégie de quantification
            quant_strategy = arch_analysis.get('quantization_strategy', {})
            if 'efficiency_analysis' in quant_strategy:
                efficiency = quant_strategy['efficiency_analysis']
                report.append("Efficacité de la quantification:")
                report.append(f"  • Compression globale: {efficiency['global_compression_ratio']:.2f}x")
                report.append(f"  • Espace économisé: {efficiency['space_saved_gb']:.2f} GB")
                report.append(f"  • Bits moyens par poids: {efficiency['average_bits_per_weight']:.2f}")
                report.append("")
        
        # Section analyse avancée (Niveau 4)
        if 'advanced' in self.analysis_results:
            advanced = self.analysis_results['advanced']
            report.append("🔬 4. ANALYSE AVANCÉE")
            report.append("-" * 40)
            
            # Estimation des performances
            performance = advanced.get('performance_estimation', {})
            if 'inference_metrics' in performance:
                metrics = performance['inference_metrics']
                report.append("Métriques d'inférence:")
                report.append(f"  • Paramètres: {metrics.get('params_billion', 0):.2f}B")
                report.append(f"  • Taille modèle: {metrics.get('model_size_gb', 0):.2f} GB")
                
                if 'estimated_tokens_per_second' in metrics:
                    tps = metrics['estimated_tokens_per_second']
                    report.append("  • Estimation tokens/sec:")
                    for hardware, speed in tps.items():
                        report.append(f"    {hardware}: {speed:.1f} t/s")
                report.append("")
            
            # Exigences mémoire
            if 'memory_requirements' in performance:
                memory = performance['memory_requirements']
                report.append("Exigences mémoire:")
                report.append(f"  • Modèle: {memory.get('model_memory_gb', 0):.2f} GB")
                report.append(f"  • Cache KV: {memory.get('kv_cache_gb', 0):.2f} GB")
                report.append(f"  • Total inférence: {memory.get('total_inference_memory_gb', 0):.2f} GB")
                report.append("")
            
            # Compatibilité
            compatibility = advanced.get('compatibility_analysis', {})
            if 'deployment_readiness' in compatibility:
                deployment = compatibility['deployment_readiness']
                report.append("Préparation au déploiement:")
                report.append(f"  • Production: {'✅' if deployment.get('production_ready') else '❌'}")
                report.append(f"  • Edge: {'✅' if deployment.get('edge_deployment') else '❌'}")
                report.append(f"  • Cloud: {'✅' if deployment.get('cloud_deployment') else '❌'}")
                report.append("")
            
            # Évaluation de qualité
            quality = advanced.get('quality_assessment', {})
            if quality:
                report.append("📊 ÉVALUATION DE QUALITÉ")
                report.append("-" * 30)
                
                overall = quality.get('overall_score', {})
                report.append(f"Score global: {overall.get('score', 0):.1f}/100 ({overall.get('grade', 'N/A')})")
                
                quant_quality = quality.get('quantization_quality', {})
                report.append(f"Qualité quantification: {quant_quality.get('weighted_score', 0):.1f}/100 ({quant_quality.get('grade', 'N/A')})")
                
                arch_coherence = quality.get('architecture_coherence', {})
                report.append(f"Cohérence architecture: {arch_coherence.get('score', 0)}/100 ({arch_coherence.get('grade', 'N/A')})")
                
                optimization = quality.get('optimization_level', {})
                report.append(f"Niveau optimisation: {optimization.get('score', 0)}/100 ({optimization.get('grade', 'N/A')})")
                
                report.append("")
                report.append(f"💡 Recommandation: {overall.get('recommendation', 'N/A')}")
                report.append("")
            
            # Opportunités d'optimisation
            opportunities = advanced.get('optimization_opportunities', [])
            if opportunities:
                report.append("🔧 OPPORTUNITÉS D'OPTIMISATION")
                report.append("-" * 40)
                
                for priority in ['high', 'medium', 'low']:
                    priority_ops = [o for o in opportunities if o.get('priority') == priority]
                    if priority_ops:
                        priority_icon = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}[priority]
                        report.append(f"{priority_icon} Priorité {priority}:")
                        
                        for opp in priority_ops:
                            report.append(f"  • {opp['description']}")
                            if 'potential_reduction' in opp:
                                report.append(f"    Réduction: {opp['potential_reduction']}")
                            if 'potential_benefit' in opp:
                                report.append(f"    Bénéfice: {opp['potential_benefit']}")
                            if 'risk' in opp:
                                report.append(f"    Risque: {opp['risk']}")
                        report.append("")
        
        # Section résumé exécutif
        report.append("📋 RÉSUMÉ EXÉCUTIF")
        report.append("-" * 40)
        
        if 'structure' in self.analysis_results:
            struct = self.analysis_results['structure']
            model_size = struct['quantization_summary']['total_model_size_mb']
            tensor_count = struct['tensor_count']
            report.append(f"• Modèle GGUF de {model_size:.1f} MB avec {tensor_count:,} tenseurs")
        
        if 'architecture' in self.analysis_results:
            arch = self.analysis_results['architecture']['reconstructed_architecture']
            if arch['num_layers'] > 0:
                params = self.analysis_results['architecture']['parameter_distribution']['total_parameters']
                report.append(f"• Architecture {arch['model_type']}: {arch['num_layers']} couches, {params/1e9:.2f}B paramètres")
        
        if 'advanced' in self.analysis_results:
            advanced = self.analysis_results['advanced']
            quality = advanced.get('quality_assessment', {})
            if quality:
                overall_score = quality.get('overall_score', {}).get('score', 0)
                grade = quality.get('overall_score', {}).get('grade', 'N/A')
                report.append(f"• Qualité globale: {overall_score:.1f}/100 ({grade})")
            
            opportunities = advanced.get('optimization_opportunities', [])
            high_priority = len([o for o in opportunities if o.get('priority') == 'high'])
            if opportunities:
                report.append(f"• {len(opportunities)} opportunités d'optimisation dont {high_priority} prioritaires")
            
            performance = advanced.get('performance_estimation', {})
            if 'memory_requirements' in performance:
                memory_gb = performance['memory_requirements'].get('total_inference_memory_gb', 0)
                report.append(f"• Mémoire requise pour inférence: {memory_gb:.1f} GB")
        
        report.append("")
        report.append("=" * 80)
        
        comprehensive_report = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(comprehensive_report)
            print(f"📄 Rapport complet sauvegardé: {save_path}")
        
        return comprehensive_report
    
    def visualize_analysis(self, figsize: Tuple[int, int] = (15, 10)):
        """
        Génère des visualisations de l'analyse
        """
        if 'tensors' not in self.analysis_results:
            print("⚠ Exécutez d'abord analyze_tensors()")
            return
        
        try:
            tensors = self.analysis_results['tensors']
            
            fig, axes = plt.subplots(2, 2, figsize=figsize)
            fig.suptitle(f'Analyse du Modèle GGUF - {self.model_path.name}', fontsize=16, fontweight='bold')
            
            # Distribution des types de quantification
            quant_types = list(tensors['quantization_analysis'].keys())
            quant_counts = [tensors['quantization_analysis'][qt]['count'] 
                           for qt in quant_types if isinstance(tensors['quantization_analysis'][qt], dict)]
            
            if quant_counts:
                axes[0, 0].pie(quant_counts, labels=quant_types, autopct='%1.1f%%', startangle=90)
                axes[0, 0].set_title('Distribution des Types de Quantification')
            
            # Distribution des tailles de tenseurs
            sizes = [t['size_mb'] for t in tensors['size_distribution']]
            if sizes:
                axes[0, 1].hist(sizes, bins=min(20, len(sizes)), alpha=0.7, color='skyblue', edgecolor='black')
                axes[0, 1].set_xlabel('Taille (MB)')
                axes[0, 1].set_ylabel('Nombre de tenseurs')
                axes[0, 1].set_title('Distribution des Tailles de Tenseurs')
                if max(sizes) / min(sizes) > 100:  # Grande variation
                    axes[0, 1].set_yscale('log')
            
            # Ratios de compression par type
            compression_ratios = {}
            for qtype, details in tensors['quantization_analysis'].items():
                if isinstance(details, dict) and 'compression_ratio' in details:
                    compression_ratios[qtype] = details['compression_ratio']
            
            if compression_ratios:
                qtypes = list(compression_ratios.keys())
                ratios = list(compression_ratios.values())
                
                bars = axes[1, 0].bar(qtypes, ratios, color='lightcoral', alpha=0.7)
                axes[1, 0].set_xlabel('Type de Quantification')
                axes[1, 0].set_ylabel('Ratio de Compression')
                axes[1, 0].set_title('Efficacité de Compression par Type')
                axes[1, 0].tick_params(axis='x', rotation=45)
                
                # Ajout des valeurs sur les barres
                for bar, ratio in zip(bars, ratios):
                    height = bar.get_height()
                    axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                   f'{ratio:.1f}x', ha='center', va='bottom')
            
            # Distribution des couches
            layer_types = list(tensors['layer_classification'].keys())
            layer_counts = list(tensors['layer_classification'].values())
            
            if layer_counts:
                colors = plt.cm.Set3(np.linspace(0, 1, len(layer_types)))
                axes[1, 1].bar(layer_types, layer_counts, color=colors, alpha=0.7)
                axes[1, 1].set_xlabel('Type de Couche')
                axes[1, 1].set_ylabel('Nombre de Tenseurs')
                axes[1, 1].set_title('Distribution des Types de Couches')
                axes[1, 1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"⚠ Erreur lors de la visualisation: {e}")
    
    def visualize_advanced_analysis(self, figsize: Tuple[int, int] = (16, 12)):
        """
        Génère des visualisations avancées
        """
        if 'advanced' not in self.analysis_results:
            print("⚠ Exécutez d'abord analyze_advanced_patterns()")
            return
        
        try:
            fig, axes = plt.subplots(2, 3, figsize=figsize)
            fig.suptitle(f'Analyse Avancée - {self.model_path.name}', fontsize=16, fontweight='bold')
            
            advanced = self.analysis_results['advanced']
            
            # 1. Qualité vs Compression
            if 'quality_assessment' in advanced:
                quality_data = advanced['quality_assessment'].get('quality_vs_compression', {})
                if quality_data:
                    qtypes = list(quality_data.keys())
                    quality_scores = [self._quality_name_to_score(self._estimate_quantization_quality(qt)) 
                                    for qt in qtypes]
                    compression_ratios = [quality_data[qt].get('compression_ratio', 1) for qt in qtypes]
                    
                    scatter = axes[0, 0].scatter(compression_ratios, quality_scores, 
                                               s=100, alpha=0.7, c=range(len(qtypes)), cmap='viridis')
                    
                    for i, qtype in enumerate(qtypes):
                        axes[0, 0].annotate(qtype, (compression_ratios[i], quality_scores[i]),
                                          xytext=(5, 5), textcoords='offset points')
                    
                    axes[0, 0].set_xlabel('Ratio de Compression')
                    axes[0, 0].set_ylabel('Score Qualité')
                    axes[0, 0].set_title('Qualité vs Compression')
                    axes[0, 0].grid(True, alpha=0.3)
            
            # 2. Distribution de la mémoire par composant
            if 'architecture' in self.analysis_results:
                param_dist = self.analysis_results['architecture']['parameter_distribution']['by_component']
                if param_dist:
                    components = list(param_dist.keys())
                    params = list(param_dist.values())
                    
                    # Calcul de la taille en MB pour chaque composant
                    sizes_mb = []
                    for component in components:
                        component_size = 0
                        for name, info in self.tensors_info.items():
                            if self._classify_tensor_role(name) == component:
                                component_size += info['size_mb']
                        sizes_mb.append(component_size)
                    
                    if sizes_mb:
                        wedges, texts, autotexts = axes[0, 1].pie(sizes_mb, labels=components, 
                                                                autopct='%1.1f%%', startangle=90)
                        axes[0, 1].set_title('Répartition Mémoire par Composant')
            
            # 3. Progression des couches
            if 'architecture' in self.analysis_results:
                layer_analysis = self.analysis_results['architecture'].get('layer_analysis', {})
                if 'parameter_progression' in layer_analysis:
                    progression = layer_analysis['parameter_progression']
                    layer_params = progression['layer_params']
                    
                    if layer_params:
                        layer_numbers = range(len(layer_params))
                        axes[0, 2].plot(layer_numbers, layer_params, 'bo-', linewidth=2, markersize=6)
                        axes[0, 2].axhline(y=progression['mean_params'], color='r', linestyle='--', 
                                         label=f"Moyenne: {progression['mean_params']:.0f}")
                        axes[0, 2].fill_between(layer_numbers, 
                                              progression['mean_params'] - progression['std_params'],
                                              progression['mean_params'] + progression['std_params'],
                                              alpha=0.2, color='red')
                        axes[0, 2].set_xlabel('Numéro de Couche')
                        axes[0, 2].set_ylabel('Nombre de Paramètres')
                        axes[0, 2].set_title('Progression des Paramètres par Couche')
                        axes[0, 2].legend()
                        axes[0, 2].grid(True, alpha=0.3)
            
            # 4. Efficacité mémoire
            memory_analysis = advanced.get('memory_access_patterns', {})
            if memory_analysis:
                cache_efficiency = memory_analysis.get('cache_efficiency', {})
                layout_info = memory_analysis.get('tensor_layout', {})
                
                metrics = ['Sequential Access', 'Fragmentation', 'Gap Ratio']
                values = [
                    cache_efficiency.get('sequential_access_ratio', 0) * 100,
                    (1 - cache_efficiency.get('sequential_access_ratio', 0)) * 100,
                    layout_info.get('fragmentation_ratio', 0) * 100
                ]
                
                bars = axes[1, 0].bar(metrics, values, color=['green', 'orange', 'red'], alpha=0.7)
                axes[1, 0].set_ylabel('Pourcentage')
                axes[1, 0].set_title('Métriques d\'Efficacité Mémoire')
                axes[1, 0].set_ylim(0, 100)
                
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 1,
                                   f'{value:.1f}%', ha='center', va='bottom')
            
            # 5. Estimation des performances
            performance = advanced.get('performance_estimation', {})
            if 'inference_metrics' in performance:
                metrics = performance['inference_metrics']
                tps_data = metrics.get('estimated_tokens_per_second', {})
                
                if tps_data:
                    hardware = list(tps_data.keys())
                    speeds = list(tps_data.values())
                    
                    bars = axes[1, 1].bar(hardware, speeds, color='skyblue', alpha=0.7)
                    axes[1, 1].set_ylabel('Tokens/seconde')
                    axes[1, 1].set_title('Performance Estimée par Hardware')
                    axes[1, 1].tick_params(axis='x', rotation=45)
                    
                    for bar, speed in zip(bars, speeds):
                        height = bar.get_height()
                        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + max(speeds)*0.01,
                                       f'{speed:.1f}', ha='center', va='bottom')
            
            # 6. Score de qualité global
            if 'quality_assessment' in advanced:
                quality = advanced['quality_assessment']
                categories = ['Quantification', 'Architecture', 'Optimisation', 'Global']
                scores = [
                    quality.get('quantization_quality', {}).get('weighted_score', 0),
                    quality.get('architecture_coherence', {}).get('score', 0),
                    quality.get('optimization_level', {}).get('score', 0),
                    quality.get('overall_score', {}).get('score', 0)
                ]
                
                colors = ['red' if s < 60 else 'orange' if s < 80 else 'green' for s in scores]
                bars = axes[1, 2].bar(categories, scores, color=colors, alpha=0.7)
                axes[1, 2].set_ylabel('Score (/100)')
                axes[1, 2].set_title('Scores de Qualité')
                axes[1, 2].set_ylim(0, 100)
                axes[1, 2].tick_params(axis='x', rotation=45)
                
                for bar, score in zip(bars, scores):
                    height = bar.get_height()
                    axes[1, 2].text(bar.get_x() + bar.get_width()/2., height + 2,
                                   f'{score:.0f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"⚠ Erreur lors de la visualisation avancée: {e}")
    
    def _quality_name_to_score(self, quality_name: str) -> float:
        """Convertit un nom de qualité en score numérique"""
        quality_scores = {
            'Excellent (pas de perte)': 100,
            'Très bon (perte minimale)': 95,
            'Bon (perte faible)': 85,
            'Bon (perte modérée)': 75,
            'Moyen (perte visible)': 65,
            'Moyen (perte notable)': 55,
            'Faible (perte importante)': 35,
            'Très faible (perte majeure)': 20
        }
        return quality_scores.get(quality_name, 50)

# Fonctions utilitaires pour l'analyse complète
def analyze_gguf_model(model_path: str, sample_size: int = 10) -> 'GGUFAnalyzer':
    """
    Fonction utilitaire pour analyser un modèle GGUF complet
    """
    analyzer = GGUFAnalyzer(model_path)
    
    # Analyse structurelle
    structure = analyzer.analyze_structure()
    print(f"\n✅ Structure analysée: {structure['tensor_count']} tenseurs")
    
    # Analyse des tenseurs
    tensor_analysis = analyzer.analyze_tensors(sample_size=sample_size)
    print(f"✅ Tenseurs analysés: compression {tensor_analysis['compression_analysis']['global_compression_ratio']:.2f}x")
    
    # Génération du rapport
    report = analyzer.generate_report()
    print("\n" + "="*50)
    print("RÉSULTATS DE L'ANALYSE")
    print("="*50)
    print(report)
    
    return analyzer

def comprehensive_gguf_analysis(model_path: str, sample_size: int = 10, visualize: bool = True) -> Tuple['GGUFAnalyzer', str]:
    """
    Analyse complète du modèle GGUF avec tous les niveaux d'analyse
    """
    print("🚀 Démarrage de l'analyse complète du modèle GGUF...")
    print("=" * 60)
    
    analyzer = GGUFAnalyzer(model_path)
    
    # Niveau 1: Analyse structurelle
    print("\n🔍 Niveau 1: Analyse structurelle...")
    structure = analyzer.analyze_structure()
    print(f"✅ Structure analysée: {structure['tensor_count']} tenseurs, {structure['file_size_mb']} MB")
    
    # Niveau 2: Analyse des tenseurs
    print("\n🔬 Niveau 2: Analyse des tenseurs...")
    tensors = analyzer.analyze_tensors(sample_size=sample_size)
    compression = tensors['compression_analysis']['global_compression_ratio']
    print(f"✅ Tenseurs analysés: compression globale {compression:.2f}x")
    
    # Niveau 3: Analyse architecturale
    print("\n🏗️ Niveau 3: Analyse architecturale...")
    try:
        architecture = analyzer.analyze_architecture()
        arch_info = architecture['reconstructed_architecture']
        print(f"✅ Architecture reconstruite: {arch_info.get('model_type', 'Unknown')}")
        print(f"   Couches: {arch_info.get('num_layers', 'N/A')}")
        print(f"   Paramètres: {architecture['parameter_distribution']['total_parameters']/1e9:.2f}B")
    except Exception as e:
        print(f"⚠️ Erreur dans l'analyse architecturale: {e}")
    
    # Niveau 4: Analyse avancée
    print("\n🔬 Niveau 4: Analyse avancée des patterns...")
    try:
        advanced = analyzer.analyze_advanced_patterns()
        
        # Qualité globale
        quality = advanced.get('quality_assessment', {})
        if quality:
            overall_score = quality.get('overall_score', {}).get('score', 0)
            grade = quality.get('overall_score', {}).get('grade', 'N/A')
            print(f"✅ Qualité globale: {overall_score:.1f}/100 ({grade})")
        
        # Opportunités d'optimisation
        opportunities = advanced.get('optimization_opportunities', [])
        high_priority = len([o for o in opportunities if o.get('priority') == 'high'])
        print(f"💡 {len(opportunities)} opportunités d'optimisation ({high_priority} haute priorité)")
        
        # Performance estimée
        performance = advanced.get('performance_estimation', {})
        if 'memory_requirements' in performance:
            memory_gb = performance['memory_requirements'].get('total_inference_memory_gb', 0)
            print(f"🎯 Mémoire requise: {memory_gb:.1f} GB")
        
    except Exception as e:
        print(f"⚠️ Erreur dans l'analyse avancée: {e}")
    
    # Génération du rapport complet
    print("\n📄 Génération du rapport complet...")
    report = analyzer.generate_comprehensive_report()
    
    # Visualisations
    if visualize:
        print("\n📊 Génération des visualisations...")
        try:
            analyzer.visualize_analysis()
            if 'advanced' in analyzer.analysis_results:
                analyzer.visualize_advanced_analysis()
        except Exception as e:
            print(f"⚠️ Erreur dans la visualisation: {e}")
    
    print("\n✅ Analyse complète terminée!")
    print("=" * 60)
    
    return analyzer, report

def compare_gguf_models(model_path1: str, model_path2: str) -> Dict[str, Any]:
    """
    Compare deux modèles GGUF
    """
    print("🔄 Comparaison de modèles GGUF en cours...")
    
    analyzer1 = GGUFAnalyzer(model_path1)
    analyzer2 = GGUFAnalyzer(model_path2)
    
    # Analyses de base
    struct1 = analyzer1.analyze_structure()
    struct2 = analyzer2.analyze_structure()
    
    comparison = {
        'model_1': model_path1,
        'model_2': model_path2,
        'size_comparison': {
            'model_1_mb': struct1['file_size_mb'],
            'model_2_mb': struct2['file_size_mb'],
            'size_ratio': struct2['file_size_mb'] / struct1['file_size_mb'] if struct1['file_size_mb'] > 0 else 0
        },
        'tensor_comparison': {
            'model_1_tensors': struct1['tensor_count'],
            'model_2_tensors': struct2['tensor_count'],
            'tensor_ratio': struct2['tensor_count'] / struct1['tensor_count'] if struct1['tensor_count'] > 0 else 0
        },
        'quantization_comparison': {},
        'architecture_comparison': {}
    }
    
    # Comparaison de quantification
    quant1 = struct1['quantization_summary']['types_distribution']
    quant2 = struct2['quantization_summary']['types_distribution']
    
    all_types = set(quant1.keys()) | set(quant2.keys())
    comparison['quantization_comparison'] = {
        qtype: {
            'model_1': quant1.get(qtype, 0),
            'model_2': quant2.get(qtype, 0)
        } for qtype in all_types
    }
    
    # Comparaison architecturale si possible
    try:
        arch1 = analyzer1.analyze_architecture()['reconstructed_architecture']
        arch2 = analyzer2.analyze_architecture()['reconstructed_architecture']
        
        comparison['architecture_comparison'] = {
            'model_type': (arch1.get('model_type', 'Unknown'), arch2.get('model_type', 'Unknown')),
            'num_layers': (arch1.get('num_layers', 0), arch2.get('num_layers', 0)),
            'hidden_size': (arch1.get('hidden_size', 0), arch2.get('hidden_size', 0)),
            'num_attention_heads': (arch1.get('num_attention_heads', 0), arch2.get('num_attention_heads', 0)),
            'vocab_size': (arch1.get('vocab_size', 0), arch2.get('vocab_size', 0))
        }
    except Exception as e:
        print(f"⚠️ Erreur dans la comparaison architecturale: {e}")
    
    # Rapport de comparaison
    print("\n📊 RAPPORT DE COMPARAISON GGUF")
    print("-" * 50)
    print(f"Modèle 1: {Path(model_path1).name}")
    print(f"Modèle 2: {Path(model_path2).name}")
    print("")
    print(f"Taille: {struct1['file_size_mb']:.1f} MB vs {struct2['file_size_mb']:.1f} MB")
    print(f"Ratio de taille: {comparison['size_comparison']['size_ratio']:.2f}x")
    print("")
    print(f"Tenseurs: {struct1['tensor_count']} vs {struct2['tensor_count']}")
    print(f"Ratio de tenseurs: {comparison['tensor_comparison']['tensor_ratio']:.2f}x")
    
    # Comparaison des types de quantification
    print("\nTypes de quantification:")
    for qtype in all_types:
        count1 = quant1.get(qtype, 0)
        count2 = quant2.get(qtype, 0)
        print(f"  {qtype}: {count1} vs {count2}")
    
    if comparison['architecture_comparison']:
        arch_comp = comparison['architecture_comparison']
        print("\nArchitecture:")
        print(f"  Type: {arch_comp['model_type'][0]} vs {arch_comp['model_type'][1]}")
        print(f"  Couches: {arch_comp['num_layers'][0]} vs {arch_comp['num_layers'][1]}")
        print(f"  Hidden size: {arch_comp['hidden_size'][0]} vs {arch_comp['hidden_size'][1]}")
    
    return comparison

def batch_gguf_analysis(model_paths: List[str], output_dir: str = "./gguf_analysis_results") -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Analyse un lot de modèles GGUF
    """
    print(f"📦 Analyse en lot de {len(model_paths)} modèles GGUF...")
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    results = {}
    summaries = []
    
    for i, model_path in enumerate(model_paths, 1):
        print(f"\n🔄 Analyse du modèle {i}/{len(model_paths)}: {Path(model_path).name}")
        
        try:
            analyzer = GGUFAnalyzer(model_path)
            
            # Analyse complète
            structure = analyzer.analyze_structure()
            tensors = analyzer.analyze_tensors(sample_size=5)
            
            # Stockage des résultats
            model_name = Path(model_path).stem
            results[model_name] = {
                'path': model_path,
                'structure': structure,
                'tensors': tensors,
                'analyzer': analyzer
            }
            
            # Génération du rapport individuel
            report_path = output_path / f"{model_name}_gguf_report.txt"
            report = analyzer.generate_comprehensive_report(str(report_path))
            
            # Résumé pour comparaison
            summaries.append({
                'name': model_name,
                'size_mb': structure['file_size_mb'],
                'tensor_count': structure['tensor_count'],
                'model_type': structure['model_info']['architecture'],
                'compression_ratio': tensors['compression_analysis']['global_compression_ratio'],
                'quantization_types': list(structure['quantization_summary']['types_distribution'].keys())
            })
            
            print(f"✅ Analyse terminée pour {model_name}")
            
        except Exception as e:
            print(f"❌ Erreur pour {model_path}: {e}")
    
    # Génération du rapport comparatif
    if len(summaries) > 1:
        comparison_report = []
        comparison_report.append("📊 RAPPORT COMPARATIF - ANALYSE GGUF EN LOT")
        comparison_report.append("=" * 70)
        comparison_report.append("")
        
        # Tableau de comparaison
        comparison_report.append("Résumé des modèles:")
        comparison_report.append("-" * 50)
        
        for summary in sorted(summaries, key=lambda x: x['size_mb'], reverse=True):
            comparison_report.append(f"• {summary['name']}")
            comparison_report.append(f"  Taille: {summary['size_mb']:.1f} MB")
            comparison_report.append(f"  Tenseurs: {summary['tensor_count']:,}")
            comparison_report.append(f"  Type: {summary['model_type']}")
            comparison_report.append(f"  Compression: {summary['compression_ratio']:.2f}x")
            comparison_report.append(f"  Quantification: {', '.join(summary['quantization_types'][:3])}")
            comparison_report.append("")
        
        # Statistiques du lot
        sizes = [s['size_mb'] for s in summaries]
        tensor_counts = [s['tensor_count'] for s in summaries]
        compression_ratios = [s['compression_ratio'] for s in summaries]
        
        comparison_report.append("Statistiques du lot:")
        comparison_report.append("-" * 50)
        comparison_report.append(f"• Nombre de modèles: {len(summaries)}")
        comparison_report.append(f"• Taille moyenne: {np.mean(sizes):.1f} MB")
        comparison_report.append(f"• Taille médiane: {np.median(sizes):.1f} MB")
        comparison_report.append(f"• Plus gros modèle: {max(sizes):.1f} MB")
        comparison_report.append(f"• Plus petit modèle: {min(sizes):.1f} MB")
        comparison_report.append(f"• Compression moyenne: {np.mean(compression_ratios):.2f}x")
        comparison_report.append(f"• Tenseurs moyens: {np.mean(tensor_counts):,.0f}")
        
        # Sauvegarde du rapport comparatif
        comparison_path = output_path / "gguf_batch_comparison_report.txt"
        with open(comparison_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(comparison_report))
        
        print(f"\n📄 Rapport comparatif sauvegardé: {comparison_path}")
    
    print(f"\n✅ Analyse en lot terminée! Résultats dans: {output_path}")
    return results, summaries

# Exemples d'utilisation
def demo_gguf_usage():
    """
    Démonstration des fonctionnalités de l'analyseur GGUF
    """
    print("🎯 DÉMONSTRATION - ANALYSEUR GGUF")
    print("=" * 60)
    
    print("\n1. Analyse simple:")
    print("analyzer = analyze_gguf_model('/path/to/model.gguf')")
    
    print("\n2. Analyse complète:")
    print("analyzer, report = comprehensive_gguf_analysis('/path/to/model.gguf')")
    
    print("\n3. Comparaison de modèles:")
    print("comparison = compare_gguf_models('/path/to/model1.gguf', '/path/to/model2.gguf')")
    
    print("\n4. Analyse en lot:")
    print("models = ['/path/to/model1.gguf', '/path/to/model2.gguf']")
    print("results, summaries = batch_gguf_analysis(models, './gguf_results')")
    
    print("\n5. Analyse détaillée:")
    print("analyzer = GGUFAnalyzer('/path/to/model.gguf')")
    print("structure = analyzer.analyze_structure()")
    print("tensors = analyzer.analyze_tensors(sample_size=15)")
    print("architecture = analyzer.analyze_architecture()")
    print("advanced = analyzer.analyze_advanced_patterns()")
    print("analyzer.visualize_advanced_analysis()")
    
    print("\n💡 SPÉCIFICITÉS GGUF:")
    print("- Support natif de la quantification K-quants")
    print("- Analyse de l'efficacité de compression")
    print("- Évaluation de la qualité vs compression")
    print("- Estimation des performances sur différents hardware")
    print("- Analyse de compatibilité avec les frameworks")
    print("- Détection des patterns de quantification optimaux")

if __name__ == "__main__":
    #demo_gguf_usage()
    #
    #analyzer = GGUFAnalyzer('gte-small.Q2_K.gguf')
    #structure = analyzer.analyze_structure()
    #tensors = analyzer.analyze_tensors(sample_size=15)
    #architecture = analyzer.analyze_architecture()
    #advanced = analyzer.analyze_advanced_patterns()
    #analyzer.visualize_advanced_analysis()
    #
    analyzer, report = comprehensive_gguf_analysis('gte-small.Q2_K.gguf')
    
    
