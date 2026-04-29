import struct
import numpy as np
import json
import os
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
import re
from enum import IntEnum

warnings.filterwarnings('ignore')

# =============================================================================
# GGUF FORMAT SPECIFICATION (v3)
# =============================================================================

GGUF_ALIGNMENT = 32

class GGMLType(IntEnum):
    F32 = 0; F16 = 1; Q4_0 = 2; Q4_1 = 3
    Q5_0 = 6; Q5_1 = 7; Q8_0 = 8; Q8_1 = 9
    Q2_K = 10; Q3_K = 11; Q4_K = 12; Q5_K = 13
    Q6_K = 14; Q8_K = 15; IQ2_XXS = 16; IQ2_XS = 17
    IQ3_XXS = 18; IQ1_S = 19; IQ4_NL = 20; IQ3_S = 21
    IQ2_S = 22; IQ4_XS = 23; I8 = 24; I16 = 25
    I32 = 26; I64 = 27; F64 = 28; IQ1_M = 29

GGML_TYPE_INFO = {
    GGMLType.F32:     {"name": "F32",     "block_size": 1,   "type_size": 4,   "bits": 32},
    GGMLType.F16:     {"name": "F16",     "block_size": 1,   "type_size": 2,   "bits": 16},
    GGMLType.Q4_0:    {"name": "Q4_0",    "block_size": 32,  "type_size": 18,  "bits": 4.5},
    GGMLType.Q4_1:    {"name": "Q4_1",    "block_size": 32,  "type_size": 20,  "bits": 5},
    GGMLType.Q5_0:    {"name": "Q5_0",    "block_size": 32,  "type_size": 22,  "bits": 5.5},
    GGMLType.Q5_1:    {"name": "Q5_1",    "block_size": 32,  "type_size": 24,  "bits": 6},
    GGMLType.Q8_0:    {"name": "Q8_0",    "block_size": 32,  "type_size": 34,  "bits": 8.5},
    GGMLType.Q8_1:    {"name": "Q8_1",    "block_size": 32,  "type_size": 36,  "bits": 9},
    GGMLType.Q2_K:    {"name": "Q2_K",    "block_size": 256, "type_size": 82,  "bits": 2.5625},
    GGMLType.Q3_K:    {"name": "Q3_K",    "block_size": 256, "type_size": 110, "bits": 3.4375},
    GGMLType.Q4_K:    {"name": "Q4_K",    "block_size": 256, "type_size": 144, "bits": 4.5},
    GGMLType.Q5_K:    {"name": "Q5_K",    "block_size": 256, "type_size": 176, "bits": 5.5},
    GGMLType.Q6_K:    {"name": "Q6_K",    "block_size": 256, "type_size": 210, "bits": 6.5625},
    GGMLType.Q8_K:    {"name": "Q8_K",    "block_size": 256, "type_size": 292, "bits": 9.125},
    GGMLType.IQ2_XXS: {"name": "IQ2_XXS", "block_size": 256, "type_size": 66,  "bits": 2.0625},
    GGMLType.IQ2_XS:  {"name": "IQ2_XS",  "block_size": 256, "type_size": 74,  "bits": 2.3125},
    GGMLType.IQ3_XXS: {"name": "IQ3_XXS", "block_size": 256, "type_size": 98,  "bits": 3.0625},
    GGMLType.IQ1_S:   {"name": "IQ1_S",   "block_size": 256, "type_size": 50,  "bits": 1.5625},
    GGMLType.IQ4_NL:  {"name": "IQ4_NL",  "block_size": 32,  "type_size": 18,  "bits": 4.5},
    GGMLType.IQ3_S:   {"name": "IQ3_S",   "block_size": 256, "type_size": 110, "bits": 3.4375},
    GGMLType.IQ2_S:   {"name": "IQ2_S",   "block_size": 256, "type_size": 82,  "bits": 2.5625},
    GGMLType.IQ4_XS:  {"name": "IQ4_XS",  "block_size": 256, "type_size": 136, "bits": 4.25},
    GGMLType.I8:      {"name": "I8",      "block_size": 1,   "type_size": 1,   "bits": 8},
    GGMLType.I16:     {"name": "I16",     "block_size": 1,   "type_size": 2,   "bits": 16},
    GGMLType.I32:     {"name": "I32",     "block_size": 1,   "type_size": 4,   "bits": 32},
    GGMLType.I64:     {"name": "I64",     "block_size": 1,   "type_size": 8,   "bits": 64},
    GGMLType.F64:     {"name": "F64",     "block_size": 1,   "type_size": 8,   "bits": 64},
    GGMLType.IQ1_M:   {"name": "IQ1_M",   "block_size": 256, "type_size": 56,  "bits": 1.75},
}

# =============================================================================
# SAFE MATH UTILITIES (Anti-ZeroDivision)
# =============================================================================

def safe_div(a, b, default=0.0):
    try:
        if b == 0 or b is None:
            return default
        return a / b
    except (TypeError, ZeroDivisionError):
        return default

def safe_mean(data, default=0.0):
    if not data:
        return default
    try:
        arr = np.array(data, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        return float(np.mean(arr)) if len(arr) > 0 else default
    except (TypeError, ValueError):
        return default

def safe_std(data, default=0.0):
    if not data or len(data) < 2:
        return default
    try:
        arr = np.array(data, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        return float(np.std(arr, ddof=0)) if len(arr) > 1 else default
    except (TypeError, ValueError):
        return default

def safe_min(data, default=0.0):
    if not data:
        return default
    try:
        arr = np.array(data, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        return float(np.min(arr)) if len(arr) > 0 else default
    except (TypeError, ValueError):
        return default

def safe_max(data, default=0.0):
    if not data:
        return default
    try:
        arr = np.array(data, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        return float(np.max(arr)) if len(arr) > 0 else default
    except (TypeError, ValueError):
        return default

def safe_median(data, default=0.0):
    if not data:
        return default
    try:
        arr = np.array(data, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        return float(np.median(arr)) if len(arr) > 0 else default
    except (TypeError, ValueError):
        return default

def safe_percentile(data, q, default=0.0):
    if not data:
        return default
    try:
        arr = np.array(data, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        return float(np.percentile(arr, q)) if len(arr) > 0 else default
    except (TypeError, ValueError):
        return default


# =============================================================================
# MAIN ANALYZER CLASS
# =============================================================================

class GGUFAnalyzer:
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Fichier GGUF non trouve: {model_path}")

        self.file_size = self.model_path.stat().st_size
        self.metadata = {}
        self.tensors_info = {}
        self.analysis_results = {}
        self.header_info = {}
        self.tensor_data_offset = 0
        self.errors = []
        self.warnings_list = []

        self._load_gguf_structure()

    def _load_gguf_structure(self):
        try:
            with open(self.model_path, 'rb') as f:
                magic = f.read(4)
                if magic != b'GGUF':
                    raise ValueError(f"Fichier GGUF invalide: magic={magic!r}")

                version = struct.unpack('<I', f.read(4))[0]
                tensor_count = struct.unpack('<Q', f.read(8))[0]
                metadata_kv_count = struct.unpack('<Q', f.read(8))[0]

                self.header_info = {
                    'magic': 'GGUF', 'version': version,
                    'tensor_count': tensor_count,
                    'metadata_kv_count': metadata_kv_count
                }

                self._read_metadata(f, metadata_kv_count)
                self._read_tensor_info(f, tensor_count)

                current_pos = f.tell()
                self.tensor_data_offset = ((current_pos + GGUF_ALIGNMENT - 1) // GGUF_ALIGNMENT) * GGUF_ALIGNMENT

                if self.tensor_data_offset != current_pos:
                    padding = self.tensor_data_offset - current_pos
                    self.warnings_list.append(f"Padding de {padding} octets detecte")

                print(f"✓ GGUF v{version}: {tensor_count} tenseurs, {metadata_kv_count} metadonnees")

        except Exception as e:
            self.errors.append(f"Erreur chargement: {e}")
            raise

    def _read_metadata(self, f, count):
        for i in range(count):
            try:
                key_length = struct.unpack('<Q', f.read(8))[0]
                key = f.read(key_length).decode('utf-8', errors='replace')
                value_type = struct.unpack('<I', f.read(4))[0]
                value = self._read_value_by_type(f, value_type)
                self.metadata[key] = {'value': value, 'type': value_type}
            except Exception as e:
                self.errors.append(f"Meta {i}: {e}")
                break

    def _read_value_by_type(self, f, value_type):
        try:
            if value_type == 0:    return struct.unpack('<B', f.read(1))[0]
            elif value_type == 1:  return struct.unpack('<b', f.read(1))[0]
            elif value_type == 2:  return struct.unpack('<H', f.read(2))[0]
            elif value_type == 3:  return struct.unpack('<h', f.read(2))[0]
            elif value_type == 4:  return struct.unpack('<I', f.read(4))[0]
            elif value_type == 5:  return struct.unpack('<i', f.read(4))[0]
            elif value_type == 6:  return struct.unpack('<f', f.read(4))[0]
            elif value_type == 7:  return struct.unpack('<B', f.read(1))[0] != 0
            elif value_type == 8:
                length = struct.unpack('<Q', f.read(8))[0]
                return f.read(length).decode('utf-8', errors='replace')
            elif value_type == 9:
                atype = struct.unpack('<I', f.read(4))[0]
                alen = struct.unpack('<Q', f.read(8))[0]
                return [self._read_value_by_type(f, atype) for _ in range(alen)]
            elif value_type == 10: return struct.unpack('<Q', f.read(8))[0]
            elif value_type == 11: return struct.unpack('<q', f.read(8))[0]
            elif value_type == 12: return struct.unpack('<d', f.read(8))[0]
            else: return f"<Unknown {value_type}>"
        except Exception as e:
            return f"<Error: {e}>"

    def _read_tensor_info(self, f, count):
        for i in range(count):
            try:
                nl = struct.unpack('<Q', f.read(8))[0]
                name = f.read(nl).decode('utf-8', errors='replace')
                ndim = struct.unpack('<I', f.read(4))[0]
                shape = [struct.unpack('<Q', f.read(8))[0] for _ in range(ndim)]
                ggml_type = struct.unpack('<I', f.read(4))[0]
                offset = struct.unpack('<Q', f.read(8))[0]

                ec = int(np.prod(shape)) if shape else 0
                ti = GGML_TYPE_INFO.get(ggml_type, {"name": f"U_{ggml_type}", "block_size": 1, "type_size": 1, "bits": 32})

                if ti["block_size"] > 1:
                    nb = (ec + ti["block_size"] - 1) // max(ti["block_size"], 1)
                    sb = nb * ti["type_size"]
                else:
                    sb = ec * ti["type_size"]

                self.tensors_info[name] = {
                    'shape': shape, 'ggml_type': ggml_type,
                    'type_name': ti["name"], 'offset': offset,
                    'absolute_offset': self.tensor_data_offset + offset,
                    'size_bytes': sb, 'size_mb': safe_div(sb, 1024*1024),
                    'element_count': ec, 'block_size': ti["block_size"],
                    'bits_per_element': ti.get("bits", 32),
                    'is_quantized': ti["block_size"] > 1
                }
            except Exception as e:
                self.errors.append(f"Tensor {i}: {e}")

    def analyze_structure(self):
        print("🔍 Analyse structurelle...")
        si = {
            'file_size_mb': round(safe_div(self.file_size, 1024*1024), 2),
            'gguf_version': self.header_info.get('version', 0),
            'tensor_count': self.header_info.get('tensor_count', 0),
            'metadata_count': self.header_info.get('metadata_kv_count', 0),
            'model_info': {}, 'quantization_summary': {},
            'architecture_hints': {}, 'errors': self.errors,
            'warnings': self.warnings_list
        }

        if self.metadata:
            arch = self._safe_get_meta('general.architecture', 'unknown')
            si['model_info'] = {
                'name': self._safe_get_meta('general.name', 'Unknown'),
                'architecture': arch,
                'file_type': self._safe_get_meta('general.file_type', 'Unknown'),
                'quantization_version': self._safe_get_meta('general.quantization_version', 'Unknown')
            }
            for k in self.metadata:
                if k.startswith(arch + '.'):
                    si['architecture_hints'][k.split('.', 1)[1]] = self.metadata[k]['value']

        qt = defaultdict(int)
        ts = 0
        for t in self.tensors_info.values():
            qt[t['type_name']] += 1
            ts += t['size_bytes']

        si['quantization_summary'] = {
            'types_distribution': dict(qt),
            'total_model_size_mb': round(safe_div(ts, 1024*1024), 2),
            'quantized_tensors': sum(1 for t in self.tensors_info.values() if t['is_quantized']),
            'unquantized_tensors': sum(1 for t in self.tensors_info.values() if not t['is_quantized'])
        }
        self.analysis_results['structure'] = si
        return si

    def _safe_get_meta(self, key, default=None):
        return self.metadata.get(key, {}).get('value', default) if key in self.metadata else default

    def analyze_tensors(self, sample_size=10):
        print("🔬 Analyse des tenseurs...")
        if not self.tensors_info:
            return {'error': 'Aucun tenseur'}

        ta = {
            'layer_classification': defaultdict(int),
            'quantization_analysis': {},
            'size_distribution': [],
            'largest_tensors': [],
            'shape_analysis': {},
            'compression_analysis': {},
            'detailed_sample': {}
        }

        for name, info in self.tensors_info.items():
            lt = self._classify_role(name)
            ta['layer_classification'][lt] += 1
            ta['size_distribution'].append({
                'name': name, 'size_mb': info['size_mb'],
                'shape': info['shape'], 'type': info['type_name'],
                'is_quantized': info['is_quantized']
            })

        qs = defaultdict(lambda: {'count': 0, 'total_size': 0, 'total_elements': 0})
        for info in self.tensors_info.values():
            q = info['type_name']
            qs[q]['count'] += 1
            qs[q]['total_size'] += info['size_bytes']
            qs[q]['total_elements'] += info['element_count']

        for q, s in qs.items():
            f32s = s['total_elements'] * 4
            qs[q]['compression_ratio'] = safe_div(f32s, s['total_size'], 1.0)
            qs[q]['bits_per_element'] = safe_div(s['total_size']*8, s['total_elements'], 32.0)

        ta['quantization_analysis'] = dict(qs)
        ta['largest_tensors'] = sorted(ta['size_distribution'], key=lambda x: x['size_mb'], reverse=True)[:15]

        ds = {'1d': 0, '2d': 0, '3d': 0, '4d+': 0}
        sp = defaultdict(int)
        for info in self.tensors_info.values():
            shape = info['shape']
            nd = len(shape)
            if nd == 1:
                ds['1d'] += 1; sp[f"1D({shape[0]})"] += 1
            elif nd == 2:
                ds['2d'] += 1; sp[f"2D({shape[0]}x{shape[1]})"] += 1
            elif nd == 3: ds['3d'] += 1
            else: ds['4d+'] += 1

        ta['shape_analysis'] = {
            'dimension_distribution': dict(ds),
            'common_shapes': dict(sorted(sp.items(), key=lambda x: x[1], reverse=True)[:10])
        }

        tu = sum(i['element_count']*4 for i in self.tensors_info.values())
        tc = sum(i['size_bytes'] for i in self.tensors_info.values())
        te = sum(i['element_count'] for i in self.tensors_info.values())

        ta['compression_analysis'] = {
            'global_compression_ratio': safe_div(tu, tc, 1.0),
            'space_saved_mb': safe_div(tu-tc, 1024*1024, 0.0),
            'average_bits_per_element': safe_div(tc*8, te, 32.0)
        }

        for tn in list(self.tensors_info.keys())[:sample_size]:
            ta['detailed_sample'][tn] = self._analyze_tensor_detail(tn)

        self.analysis_results['tensors'] = ta
        return ta

    def _classify_role(self, name):
        nl = name.lower()
        if any(p in nl for p in ['embed', 'token', 'wte']): return 'embedding'
        if any(p in nl for p in ['q_proj', 'query', 'attn_q']): return 'attention_q'
        if any(p in nl for p in ['k_proj', 'key', 'attn_k']): return 'attention_k'
        if any(p in nl for p in ['v_proj', 'value', 'attn_v']): return 'attention_v'
        if any(p in nl for p in ['o_proj', 'out_proj', 'attn_output']): return 'attention_output'
        if any(p in nl for p in ['gate_proj', 'w1', 'ffn_gate']): return 'mlp_gate'
        if any(p in nl for p in ['up_proj', 'w3', 'ffn_up']): return 'mlp_up'
        if any(p in nl for p in ['down_proj', 'w2', 'ffn_down']): return 'mlp_down'
        if any(p in nl for p in ['norm', 'ln', 'rms_norm']): return 'normalization'
        if any(p in nl for p in ['lm_head', 'output', 'classifier']): return 'output_head'
        if any(p in nl for p in ['bias']): return 'bias'
        return 'other'

    def _analyze_tensor_detail(self, name):
        if name not in self.tensors_info:
            return {}
        info = self.tensors_info[name]
        f32s = info['element_count'] * 4
        sb = info['size_bytes']
        ec = info['element_count']

        a = {
            'basic_info': {
                'shape': info['shape'], 'type': info['type_name'],
                'size_mb': info['size_mb'], 'element_count': ec,
                'is_quantized': info['is_quantized']
            },
            'quantization_details': {},
            'memory_efficiency': {},
            'layer_role': self._classify_role(name)
        }

        if info['is_quantized']:
            ti = GGML_TYPE_INFO.get(info['ggml_type'], {})
            a['quantization_details'] = {
                'type': info['type_name'],
                'block_size': ti.get('block_size', 1),
                'bytes_per_block': ti.get('type_size', 1),
                'compression_ratio': safe_div(f32s, sb, 1.0),
                'bits_per_element': safe_div(sb*8, ec, 32.0)
            }

        bs = self._get_base_size(info['ggml_type'])
        a['memory_efficiency'] = {
            'density': safe_div(ec, sb, 0.0),
            'overhead_ratio': 1.0 - safe_div(ec * bs, sb, 0.0)
        }
        return a

    def _get_base_size(self, gt):
        ti = GGML_TYPE_INFO.get(gt, {"type_size": 1, "block_size": 1})
        return safe_div(ti["type_size"], max(ti.get("block_size", 1), 1), 1.0)

    def analyze_architecture(self):
        print("🏗️ Analyse architecturale...")
        aa = {
            'reconstructed_architecture': self._reconstruct_arch(),
            'parameter_distribution': self._analyze_param_dist(),
            'layer_analysis': self._analyze_layers(),
            'attention_analysis': self._analyze_attention(),
            'quantization_strategy': self._analyze_quant_strategy(),
            'model_topology': self._create_topology()
        }
        self.analysis_results['architecture'] = aa
        return aa

    def _reconstruct_arch(self):
        arch = {
            'model_type': 'unknown', 'num_layers': 0, 'hidden_size': 0,
            'num_attention_heads': 0, 'num_key_value_heads': 0,
            'intermediate_size': 0, 'vocab_size': 0,
            'context_length': 0, 'rope_theta': 0, 'architecture_specific': {}
        }

        if self.metadata:
            an = self._safe_get_meta('general.architecture', 'unknown')
            arch['model_type'] = an
            ap = an + '.'
            for k, mi in self.metadata.items():
                if k.startswith(ap):
                    pn = k[len(ap):]
                    v = mi['value']
                    m = {
                        'block_count': 'num_layers',
                        'embedding_length': 'hidden_size',
                        'attention.head_count': 'num_attention_heads',
                        'attention.head_count_kv': 'num_key_value_heads',
                        'feed_forward_length': 'intermediate_size',
                        'context_length': 'context_length',
                        'rope.freq_base': 'rope_theta'
                    }
                    if pn in m:
                        arch[m[pn]] = v
                    else:
                        arch['architecture_specific'][pn] = v

            toks = self.metadata.get('tokenizer.ggml.tokens', {}).get('value', [])
            if isinstance(toks, list):
                arch['vocab_size'] = len(toks)

        self._validate_arch_from_tensors(arch)
        return arch

    def _validate_arch_from_tensors(self, arch):
        for name, info in self.tensors_info.items():
            if 'embed' in name.lower() or 'token' in name.lower():
                shape = info['shape']
                if len(shape) >= 2:
                    if arch['vocab_size'] == 0: arch['vocab_size'] = max(shape)
                    if arch['hidden_size'] == 0: arch['hidden_size'] = min(shape)

        lp = re.compile(r'(?:layers?|blocks?)\.(\d+)\.')
        lf = set()
        for name in self.tensors_info.keys():
            m = lp.search(name)
            if m: lf.add(int(m.group(1)))
        if lf and arch['num_layers'] == 0:
            arch['num_layers'] = max(lf) + 1

        if arch['num_attention_heads'] == 0 and arch['hidden_size'] > 0:
            for name, info in self.tensors_info.items():
                if 'attn' in name.lower() and 'weight' in name.lower():
                    shape = info['shape']
                    if len(shape) == 2:
                        hs = arch['hidden_size']
                        for h in [8, 12, 16, 32, 64, 128]:
                            if hs % max(h, 1) == 0:
                                arch['num_attention_heads'] = h
                                break
                        break

    def _analyze_param_dist(self):
        d = {
            'by_component': defaultdict(int),
            'by_layer': defaultdict(int),
            'by_quantization': defaultdict(int),
            'total_parameters': 0,
            'efficiency_metrics': {}
        }
        for name, info in self.tensors_info.items():
            ec = info['element_count']
            d['total_parameters'] += ec
            c = self._classify_role(name)
            d['by_component'][c] += ec
            m = re.search(r'(?:layers?|blocks?)\.(\d+)\.', name)
            if m:
                d['by_layer'][f'layer_{int(m.group(1))}'] += ec
            else:
                d['by_layer']['global'] += ec
            d['by_quantization'][info['type_name']] += ec

        t = d['total_parameters']
        lk = [k for k in d['by_layer'].keys() if k.startswith('layer_')]
        nl = max(len(lk), 1)

        d['efficiency_metrics'] = {
            'embedding_ratio': safe_div(d['by_component']['embedding'], t),
            'attention_ratio': safe_div(
                d['by_component']['attention_q'] + d['by_component']['attention_k'] +
                d['by_component']['attention_v'] + d['by_component']['attention_output'], t),
            'mlp_ratio': safe_div(
                d['by_component']['mlp_gate'] + d['by_component']['mlp_up'] + d['by_component']['mlp_down'], t),
            'normalization_ratio': safe_div(d['by_component']['normalization'], t),
            'params_per_layer': safe_div(t, nl)
        }
        return d

    def _analyze_layers(self):
        la = {
            'layer_structure': {},
            'parameter_progression': {
                'layer_params': [], 'mean_params': 0.0, 'std_params': 0.0,
                'min_params': 0, 'max_params': 0, 'layer_count': 0
            }
        }
        layers = defaultdict(list)
        lp = re.compile(r'(?:layers?|blocks?)\.(\d+)\.')
        for name, info in self.tensors_info.items():
            m = lp.search(name)
            if m: layers[int(m.group(1))].append((name, info))

        for ln, lt in layers.items():
            li = {
                'tensor_count': len(lt),
                'total_parameters': sum(i['element_count'] for _, i in lt),
                'total_size_mb': sum(i['size_mb'] for _, i in lt),
                'components': {}, 'quantization_mix': defaultdict(int)
            }
            for n, i in lt:
                c = self._classify_role(n)
                if c not in li['components']:
                    li['components'][c] = {'tensors': [], 'parameters': 0, 'size_mb': 0}
                li['components'][c]['tensors'].append(n)
                li['components'][c]['parameters'] += i['element_count']
                li['components'][c]['size_mb'] += i['size_mb']
                li['quantization_mix'][i['type_name']] += 1
            la['layer_structure'][f'layer_{ln}'] = li

        if layers:
            sl = sorted(layers.keys())
            pp = []
            for ln in sl:
                tp = sum(i['element_count'] for _, i in layers[ln])
                pp.append(tp)
            if pp:
                la['parameter_progression'] = {
                    'layer_params': pp,
                    'mean_params': safe_mean(pp),
                    'std_params': safe_std(pp),
                    'min_params': int(safe_min(pp)),
                    'max_params': int(safe_max(pp)),
                    'layer_count': len(pp)
                }
        return la

    def _analyze_attention(self):
        aa = {'attention_patterns': {}, 'head_analysis': {}, 'attention_efficiency': {}, 'kv_cache_analysis': {}}
        at = {}
        for name, info in self.tensors_info.items():
            if any(p in name.lower() for p in ['attn', 'attention']):
                at[name] = info

        atypes = {'query': [], 'key': [], 'value': [], 'output': [], 'other': []}
        for name, info in at.items():
            nl = name.lower()
            if any(p in nl for p in ['q_proj', 'query', 'attn_q']): atypes['query'].append((name, info))
            elif any(p in nl for p in ['k_proj', 'key', 'attn_k']): atypes['key'].append((name, info))
            elif any(p in nl for p in ['v_proj', 'value', 'attn_v']): atypes['value'].append((name, info))
            elif any(p in nl for p in ['o_proj', 'out_proj', 'output', 'attn_out']): atypes['output'].append((name, info))
            else: atypes['other'].append((name, info))

        if atypes['query']:
            sq = atypes['query'][0][1]
            shape = sq['shape']
            if len(shape) >= 2:
                td = shape[-1] if len(shape) == 2 else shape[-2]
                ai = self.analysis_results.get('architecture', {}).get('reconstructed_architecture', {})
                nh = ai.get('num_attention_heads', 0)
                if nh > 0:
                    aa['head_analysis'] = {
                        'num_heads': nh, 'head_dimension': td // max(nh, 1),
                        'total_attention_dim': td,
                        'kv_heads': ai.get('num_key_value_heads', nh)
                    }

        tap = sum(i['element_count'] for ts in atypes.values() for _, i in ts)
        tmp = sum(i['element_count'] for i in self.tensors_info.values())
        aa['attention_efficiency'] = {
            'attention_param_ratio': safe_div(tap, tmp),
            'total_attention_params': tap,
            'attention_size_mb': sum(i['size_mb'] for ts in atypes.values() for _, i in ts)
        }

        ai = self.analysis_results.get('architecture', {}).get('reconstructed_architecture', {})
        nh = ai.get('num_attention_heads', 0)
        kv = ai.get('num_key_value_heads', nh)
        if kv != nh and kv > 0:
            aa['kv_cache_analysis'] = {
                'is_grouped_query': True,
                'kv_reduction_ratio': safe_div(nh, kv, 1.0),
                'cache_efficiency': f"{kv}/{nh} heads for K/V"
            }
        aa['attention_patterns'] = {k: len(v) for k, v in atypes.items()}
        return aa

    def _analyze_quant_strategy(self):
        sa = {'quantization_scheme': {}, 'layer_specific_quantization': {},
              'efficiency_analysis': {}, 'quality_vs_compression': {}}

        qc = defaultdict(lambda: defaultdict(int))
        for name, info in self.tensors_info.items():
            qc[self._classify_role(name)][info['type_name']] += 1
        sa['quantization_scheme'] = dict(qc)

        lq = defaultdict(lambda: defaultdict(int))
        lp = re.compile(r'(?:layers?|blocks?)\.(\d+)\.')
        for name, info in self.tensors_info.items():
            m = lp.search(name)
            lk = f"layer_{m.group(1)}" if m else "global"
            lq[lk][info['type_name']] += 1
        sa['layer_specific_quantization'] = dict(lq)

        tcs = sum(i['size_bytes'] for i in self.tensors_info.values())
        tus = sum(i['element_count']*4 for i in self.tensors_info.values())
        te = sum(i['element_count'] for i in self.tensors_info.values())

        sa['efficiency_analysis'] = {
            'global_compression_ratio': safe_div(tus, tcs, 1.0),
            'space_saved_gb': safe_div(tus-tcs, 1024**3, 0.0),
            'average_bits_per_weight': safe_div(tcs*8, te, 32.0)
        }

        ta = {}
        for qt in set(i['type_name'] for i in self.tensors_info.values()):
            tt = [i for i in self.tensors_info.values() if i['type_name'] == qt]
            if tt:
                tec = sum(t['element_count'] for t in tt)
                ts = sum(t['size_bytes'] for t in tt)
                f32s = tec * 4
                ta[qt] = {
                    'tensor_count': len(tt),
                    'compression_ratio': safe_div(f32s, ts, 1.0),
                    'bits_per_element': safe_div(ts*8, tec, 32.0),
                    'quality_estimate': self._estimate_quality(qt)
                }
        sa['quality_vs_compression'] = ta
        return sa

    def _estimate_quality(self, qtype):
        qm = {
            'F32': 'Excellent', 'F16': 'Tres bon', 'Q8_0': 'Bon',
            'Q6_K': 'Bon', 'Q5_K': 'Moyen', 'Q4_K': 'Moyen',
            'Q3_K': 'Faible', 'Q2_K': 'Tres faible',
            'IQ2_XXS': 'Tres faible', 'IQ1_S': 'Critique', 'IQ1_M': 'Critique'
        }
        return qm.get(qtype, 'Inconnue')

    def _create_topology(self):
        topo = {'graph_structure': {'nodes': [], 'edges': [], 'num_layers': 0, 'is_sequential': True},
                'data_flow': {'input_nodes': [], 'output_nodes': [], 'intermediate_nodes': []}}
        try:
            import networkx as nx
            G = nx.DiGraph()
            lp = re.compile(r'(?:layers?|blocks?)\.(\d+)\.')
            layers = set()
            for name in self.tensors_info.keys():
                m = lp.search(name)
                if m:
                    layers.add(int(m.group(1)))
                    G.add_node(f"layer_{int(m.group(1))}")
            sl = sorted(layers)
            for i in range(len(sl)-1):
                G.add_edge(f"layer_{sl[i]}", f"layer_{sl[i+1]}")
            if any('embed' in n.lower() for n in self.tensors_info.keys()):
                G.add_node("embedding")
                if sl: G.add_edge("embedding", f"layer_{sl[0]}")
            if any(p in n.lower() for n in self.tensors_info.keys() for p in ['lm_head', 'output']):
                G.add_node("output_head")
                if sl: G.add_edge(f"layer_{sl[-1]}", "output_head")
            topo['graph_structure'] = {
                'nodes': list(G.nodes()), 'edges': list(G.edges()),
                'num_layers': len(layers),
                'is_sequential': len(G.edges()) == len(G.nodes()) - 1
            }
            topo['data_flow'] = {
                'input_nodes': [n for n in G.nodes() if G.in_degree(n) == 0],
                'output_nodes': [n for n in G.nodes() if G.out_degree(n) == 0],
                'intermediate_nodes': [n for n in G.nodes() if G.in_degree(n) > 0 and G.out_degree(n) > 0]
            }
        except ImportError:
            pass
        return topo

    def analyze_advanced_patterns(self):
        print("🔬 Analyse avancee...")
        aa = {
            'quantization_patterns': self._analyze_q_patterns(),
            'memory_access_patterns': self._analyze_mem_patterns(),
            'optimization_opportunities': self._find_opportunities(),
            'performance_estimation': self._estimate_perf(),
            'compatibility_analysis': self._analyze_compat(),
            'quality_assessment': self._assess_quality()
        }
        self.analysis_results['advanced'] = aa
        return aa

    def _analyze_q_patterns(self):
        p = {'mixed_precision_analysis': {}, 'quantization_transitions': {},
             'outlier_detection': {'outliers': [], 'outlier_count': 0,
                                   'size_statistics': {'median': 0.0, 'q25': 0.0, 'q75': 0.0, 'iqr': 0.0}}}
        pbl = defaultdict(lambda: defaultdict(int))
        lp = re.compile(r'(?:layers?|blocks?)\.(\d+)\.')
        for name, info in self.tensors_info.items():
            m = lp.search(name)
            lk = f"layer_{m.group(1)}" if m else "global"
            pbl[lk][info['type_name']] += 1
        p['mixed_precision_analysis'] = dict(pbl)

        sizes = [i['size_mb'] for i in self.tensors_info.values()]
        if sizes:
            q75 = safe_percentile(sizes, 75)
            q25 = safe_percentile(sizes, 25)
            iqr = q75 - q25
            lb = q25 - 1.5 * iqr
            ub = q75 + 1.5 * iqr
            med = safe_median(sizes)
            out = []
            for name, info in self.tensors_info.items():
                sz = info['size_mb']
                if sz < lb or sz > ub:
                    out.append({'name': name, 'size_mb': sz,
                                'type': 'oversized' if sz > ub else 'undersized',
                                'deviation_factor': safe_div(sz, med, 1.0)})
            p['outlier_detection'] = {
                'outliers': out, 'outlier_count': len(out),
                'size_statistics': {'median': med, 'q25': q25, 'q75': q75, 'iqr': iqr}
            }
        return p

    def _analyze_mem_patterns(self):
        mp = {'tensor_layout': {}, 'cache_efficiency': {}}
        to = [(n, i['offset'], i['size_bytes']) for n, i in self.tensors_info.items()]
        to.sort(key=lambda x: x[1])
        gaps = []
        for i in range(len(to)-1):
            ce = to[i][1] + to[i][2]
            ns = to[i+1][1]
            g = ns - ce
            if g > 0:
                gaps.append({'after': to[i][0], 'before': to[i+1][0], 'gap_bytes': g})
        tg = sum(g['gap_bytes'] for g in gaps)
        mp['tensor_layout'] = {
            'total_gaps': len(gaps), 'total_gap_bytes': tg,
            'largest_gap': safe_max([g['gap_bytes'] for g in gaps], 0),
            'fragmentation_ratio': safe_div(tg, self.file_size, 0.0)
        }
        sas = sum(1 for i in range(len(to)-1) if to[i][1]+to[i][2] == to[i+1][1])
        mp['cache_efficiency'] = {
            'sequential_access_ratio': safe_div(sas, max(len(to)-1, 1)),
            'fragmentation_level': 'low' if len(gaps) < 5 else 'medium' if len(gaps) < 20 else 'high'
        }
        return mp

    def _find_opportunities(self):
        opps = []
        tc = defaultdict(int)
        ts = defaultdict(float)
        for info in self.tensors_info.values():
            tc[info['type_name']] += 1
            ts[info['type_name']] += info['size_mb']

        uq = ts.get('F32', 0) + ts.get('F16', 0)
        if uq > 0:
            opps.append({
                'type': 'aggressive_quantization',
                'priority': 'high' if uq > 100 else 'medium',
                'description': f'Quantifier F32/F16 ({uq:.1f} MB)',
                'potential_reduction': f'{uq * 0.5:.1f} MB avec Q4_K',
                'risk': 'Perte moderee'
            })

        ut = len(set(i['type_name'] for i in self.tensors_info.values() if i['is_quantized']))
        if ut > 3:
            opps.append({
                'type': 'quantization_uniformization',
                'priority': 'medium',
                'description': f'Trop de types ({ut})',
                'recommendation': 'Standardiser sur Q4_K ou Q5_K'
            })

        es = sum(i['size_mb'] for n, i in self.tensors_info.items() if 'embed' in n.lower())
        tot = sum(i['size_mb'] for i in self.tensors_info.values())
        er = safe_div(es, tot)
        if er > 0.15:
            opps.append({
                'type': 'embedding_optimization',
                'priority': 'medium',
                'description': f'Embeddings = {er*100:.1f}% du modele',
                'potential_reduction': f'{es * 0.3:.1f} MB'
            })

        la = self.analysis_results.get('architecture', {}).get('layer_analysis', {})
        pp = la.get('parameter_progression', {})
        sp = pp.get('std_params', 0)
        mp = pp.get('mean_params', 1)
        if safe_div(sp, mp) < 0.1 and mp > 0:
            opps.append({
                'type': 'layer_sharing',
                'priority': 'low',
                'description': 'Couches tres similaires',
                'risk': 'Perte potentielle'
            })
        return opps

    def _estimate_perf(self):
        pe = {'inference_metrics': {}, 'memory_requirements': {}, 'hardware_compatibility': {}}
        tp = sum(i['element_count'] for i in self.tensors_info.values())
        tsm = sum(i['size_mb'] for i in self.tensors_info.values())
        ai = self.analysis_results.get('architecture', {}).get('reconstructed_architecture', {})
        nl = ai.get('num_layers', 0)
        hs = ai.get('hidden_size', 0)
        cl = ai.get('context_length', 2048)

        if nl > 0 and hs > 0:
            fpt = (cl * hs * hs + cl * hs * ai.get('intermediate_size', hs * 4)) * nl
            pe['inference_metrics'] = {
                'params_billion': safe_div(tp, 1e9),
                'model_size_gb': safe_div(tsm, 1024),
                'flops_per_token': fpt,
                'estimated_tokens_per_second': self._est_tps(fpt, tsm)
            }

        pe['memory_requirements'] = {
            'model_memory_gb': safe_div(tsm, 1024),
            'kv_cache_gb': self._est_kv(ai, cl),
            'total_inference_memory_gb': safe_div(tsm, 1024) * 1.2,
            'batch_size_1_memory_gb': safe_div(tsm, 1024) * 1.5
        }
        pe['hardware_compatibility'] = {
            'cpu_inference': 'Possible' if tsm < 16384 else 'Difficile',
            'mobile_inference': 'Possible' if tsm < 4096 else 'Non recommande',
            'edge_inference': 'Possible' if tsm < 1024 else 'Limite',
            'recommended_gpu_memory_gb': max(8, safe_div(tsm, 1024) * 1.5)
        }
        return pe

    def _est_tps(self, flops_per_token, model_size_mb):
        hw = {
            'cpu_high_end': {'flops': 1e12, 'bw': 100},
            'gpu_consumer': {'flops': 20e12, 'bw': 500},
            'gpu_datacenter': {'flops': 100e12, 'bw': 1000}
        }
        est = {}
        for h, s in hw.items():
            cl = safe_div(s['flops'], flops_per_token)
            ml = safe_div(s['bw'] * 1024, max(model_size_mb, 1))
            est[h] = round(min(cl, ml), 2)
        return est

    def _est_kv(self, ai, cl):
        nl = ai.get('num_layers', 0)
        hs = ai.get('hidden_size', 0)
        nh = ai.get('num_attention_heads', 0)
        kv = ai.get('num_key_value_heads', nh)
        if nl > 0 and hs > 0 and kv > 0:
            kcb = 2 * nl * cl * hs * safe_div(kv, nh, 1.0) * 2
            return safe_div(kcb, 1024**3, 0.0)
        return 0.0

    def _analyze_compat(self):
        c = {'framework_support': {}, 'quantization_support': {},
             'hardware_acceleration': {}, 'deployment_readiness': {}}
        gv = self.header_info.get('version', 0)
        qt = set(i['type_name'] for i in self.tensors_info.values())
        c['framework_support'] = {
            'llama_cpp': 'Full' if gv >= 2 else 'Limited',
            'ggml': 'Full', 'transformers': 'Via conversion',
            'onnx': 'Via conversion', 'ollama': 'Full' if gv >= 3 else 'Limited'
        }
        aq = {'Q2_K', 'Q3_K', 'Q4_K', 'Q5_K', 'Q6_K', 'Q8_K', 'IQ2_XXS', 'IQ2_XS', 'IQ3_XXS', 'IQ4_NL', 'IQ4_XS'}
        ha = bool(qt.intersection(aq))
        c['quantization_support'] = {
            'has_k_quants': ha, 'supported_types': sorted(list(qt)),
            'cpu_optimized': 'Q4_0' in qt or 'Q8_0' in qt,
            'gpu_optimized': ha
        }
        msg = safe_div(sum(i['size_mb'] for i in self.tensors_info.values()), 1024)
        c['hardware_acceleration'] = {
            'cpu_avx2': 'Recommended',
            'cpu_avx512': 'Optimal' if msg < 32 else 'Good',
            'gpu_cuda': 'Good' if ha else 'Basic',
            'gpu_metal': 'Good' if ha else 'Basic',
            'mobile_neon': 'Possible' if msg < 4 else 'Not recommended'
        }
        c['deployment_readiness'] = {
            'production_ready': msg < 64 and ha,
            'edge_deployment': msg < 2,
            'cloud_deployment': True,
            'containerization': 'Ready',
            'serving_frameworks': ['llama.cpp', 'Ollama', 'LM Studio', 'vLLM']
        }
        return c

    def _assess_quality(self):
        q = {'quantization_quality': {}, 'architecture_coherence': {},
             'optimization_level': {}, 'overall_score': {}}
        qt = [i['type_name'] for i in self.tensors_info.values()]
        tc = defaultdict(int)
        for t in qt: tc[t] += 1

        qs = {'F32': 100, 'F16': 95, 'Q8_0': 90, 'Q6_K': 85, 'Q5_K': 80,
              'Q4_K': 75, 'Q4_0': 70, 'Q3_K': 60, 'Q2_K': 45,
              'IQ2_XXS': 40, 'IQ2_XS': 42, 'IQ3_XXS': 55, 'IQ1_S': 30,
              'IQ4_NL': 75, 'IQ3_S': 60, 'IQ2_S': 45, 'IQ4_XS': 73, 'IQ1_M': 32}

        wq = sum(qs.get(t, 50) * c for t, c in tc.items()) / max(len(qt), 1)
        q['quantization_quality'] = {
            'weighted_score': round(wq, 1),
            'grade': self._to_grade(wq),
            'dominant_type': max(tc.items(), key=lambda x: x[1])[0] if tc else 'N/A',
            'uniformity': len(tc) <= 3
        }

        ai = self.analysis_results.get('architecture', {}).get('reconstructed_architecture', {})
        cs = 0
        if ai.get('num_layers', 0) > 0: cs += 25
        if ai.get('hidden_size', 0) > 0: cs += 25
        if ai.get('num_attention_heads', 0) > 0: cs += 25
        if ai.get('vocab_size', 0) > 0: cs += 25
        q['architecture_coherence'] = {
            'score': cs, 'grade': self._to_grade(cs),
            'complete_metadata': cs == 100
        }

        os = 0
        if any(i['is_quantized'] for i in self.tensors_info.values()): os += 30
        aq = {'Q2_K', 'Q3_K', 'Q4_K', 'Q5_K', 'Q6_K', 'Q8_K', 'IQ2_XXS', 'IQ2_XS', 'IQ3_XXS', 'IQ4_NL', 'IQ4_XS', 'IQ2_S', 'IQ3_S'}
        if any(i['type_name'] in aq for i in self.tensors_info.values()): os += 20
        tu = sum(i['element_count']*4 for i in self.tensors_info.values())
        tc_ = sum(i['size_bytes'] for i in self.tensors_info.values())
        cr = safe_div(tu, tc_, 1.0)
        if cr > 2: os += 20
        ma = self.analysis_results.get('advanced', {}).get('memory_access_patterns', {})
        if ma.get('cache_efficiency', {}).get('sequential_access_ratio', 0) > 0.8: os += 15
        if len(set(i['type_name'] for i in self.tensors_info.values())) <= 3: os += 15
        os = min(os, 100)
        q['optimization_level'] = {
            'score': os, 'grade': self._to_grade(os),
            'compression_ratio': round(cr, 2)
        }

        ov = wq * 0.4 + cs * 0.3 + os * 0.3
        q['overall_score'] = {
            'score': round(ov, 1), 'grade': self._to_grade(ov),
            'recommendation': self._rec(ov)
        }
        return q

    def _to_grade(self, s):
        if s >= 90: return 'A+'
        elif s >= 80: return 'A'
        elif s >= 70: return 'B'
        elif s >= 60: return 'C'
        elif s >= 50: return 'D'
        else: return 'F'

    def _rec(self, s):
        if s >= 85: return "Excellent, pret pour production"
        elif s >= 75: return "Bon, optimisations possibles"
        elif s >= 65: return "Correct, optimisations recommandees"
        elif s >= 50: return "Ameliorations significatives necessaires"
        else: return "Faible qualite, reconstruction recommandee"

    def generate_comprehensive_report(self, save_path=None):
        r = []
        r.append("=" * 80)
        r.append("RAPPORT COMPLET D'ANALYSE - MODELE GGUF")
        r.append("=" * 80)
        r.append(f"Fichier: {self.model_path.name}")
        r.append(f"Taille: {safe_div(self.file_size, 1024**2):.1f} MB")
        r.append(f"GGUF v{self.header_info.get('version', 'N/A')}")
        r.append("")

        if self.errors:
            r.append("ERREURS:")
            for e in self.errors: r.append(f"  ! {e}")
            r.append("")

        if 'structure' in self.analysis_results:
            s = self.analysis_results['structure']
            r.append("📁 STRUCTURE")
            r.append("-" * 40)
            mi = s['model_info']
            r.append(f"Nom: {mi['name']} | Arch: {mi['architecture']} | Type: {mi['file_type']}")
            r.append(f"Tenseurs: {s['tensor_count']} | Metadonnees: {s['metadata_count']}")
            q = s['quantization_summary']
            r.append(f"Taille modele: {q['total_model_size_mb']:.1f} MB")
            r.append("Types:")
            for t, c in sorted(q['types_distribution'].items()):
                r.append(f"  {t}: {c}")
            r.append("")

        if 'tensors' in self.analysis_results:
            t = self.analysis_results['tensors']
            r.append("🔬 TENSEURS")
            r.append("-" * 40)
            ca = t['compression_analysis']
            r.append(f"Compression: {ca['global_compression_ratio']:.2f}x")
            r.append(f"Espace sauve: {ca['space_saved_mb']:.1f} MB")
            r.append(f"Bits/element: {ca['average_bits_per_element']:.2f}")
            r.append("Top 5 tenseurs:")
            for i, ten in enumerate(t['largest_tensors'][:5], 1):
                r.append(f"  {i}. {ten['name']}: {ten['size_mb']:.2f} MB ({ten['type']})")
            r.append("")

        if 'architecture' in self.analysis_results:
            a = self.analysis_results['architecture']
            ar = a['reconstructed_architecture']
            r.append("🏗️ ARCHITECTURE")
            r.append("-" * 40)
            r.append(f"Type: {ar['model_type']} | Couches: {ar['num_layers']} | Hidden: {ar['hidden_size']}")
            r.append(f"Heads: {ar['num_attention_heads']} | KV Heads: {ar['num_key_value_heads']}")
            r.append(f"Vocab: {ar['vocab_size']} | Context: {ar['context_length']}")
            pd = a['parameter_distribution']
            r.append(f"Parametres totaux: {pd['total_parameters']:,} ({safe_div(pd['total_parameters'], 1e9):.2f}B)")
            r.append("")

        if 'advanced' in self.analysis_results:
            ad = self.analysis_results['advanced']
            qa = ad['quality_assessment']
            r.append("📊 QUALITE")
            r.append("-" * 40)
            ov = qa['overall_score']
            r.append(f"Score global: {ov['score']:.1f}/100 ({ov['grade']})")
            r.append(f"Recommandation: {ov['recommendation']}")
            r.append("")

            opps = ad['optimization_opportunities']
            if opps:
                r.append("💡 OPTIMISATIONS:")
                for o in opps:
                    r.append(f"  [{o['priority'].upper()}] {o['description']}")
                r.append("")

            pe = ad['performance_estimation']
            if 'memory_requirements' in pe:
                mr = pe['memory_requirements']
                r.append(f"Memoire inference: {mr['total_inference_memory_gb']:.2f} GB")
                r.append("")

        r.append("=" * 80)
        txt = "\n".join(r)
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(txt)
        return txt


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def analyze_gguf_model(model_path: str, sample_size: int = 10) -> 'GGUFAnalyzer':
    analyzer = GGUFAnalyzer(model_path)
    analyzer.analyze_structure()
    analyzer.analyze_tensors(sample_size=sample_size)
    return analyzer


def comprehensive_gguf_analysis(model_path: str, sample_size: int = 10) -> Tuple['GGUFAnalyzer', str]:
    print("🚀 Analyse complete GGUF...")
    analyzer = GGUFAnalyzer(model_path)
    analyzer.analyze_structure()
    analyzer.analyze_tensors(sample_size=sample_size)
    try:
        analyzer.analyze_architecture()
    except Exception as e:
        print(f"⚠ Architecture: {e}")
    try:
        analyzer.analyze_advanced_patterns()
    except Exception as e:
        print(f"⚠ Advanced: {e}")
    report = analyzer.generate_comprehensive_report()
    print("\n✅ Analyse terminee!")
    return analyzer, report


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        analyzer, report = comprehensive_gguf_analysis(sys.argv[1])
        print(report)
    else:
        print("Usage: python GGUF_analyzer_v3.py <model.gguf>")
