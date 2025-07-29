import json
import hashlib
import zlib

# Exemple de JSON
"""model_json = {
    "layers": [
        {"type": "Conv", "params": {"features": 32, "kernel_size": [3, 3], "padding": "SAME"}},
        {"type": "relu", "params": {}},
        {"type": "Conv", "params": {"features": 32, "kernel_size": [3, 3], "padding": "SAME"}}
    ]
}"""
with open('./Genomes/tmp.json') as f:
  model_json = json.load(f)
# Sérialisation en chaîne de caractères
model_str = json.dumps(model_json, sort_keys=True)

# Hachage SHA-256
# Nombre de Combinaisons : 2^128
# Valeur Numérique : Environ 3.4×10^38
hash_object = hashlib.sha256(model_str.encode())
unique_id = hash_object.hexdigest()
print("Identifiant unique (SHA-256):", unique_id)

# Hachage MD5 pour générer un identifiant unique plus court
# Nombre de Combinaisons : 2^128
# Valeur Numérique : Environ 3.4×10^38
hash_object = hashlib.md5(model_str.encode())
unique_id = hash_object.hexdigest()
print("Identifiant unique (MD5) :", unique_id)

# Hachage CRC32 pour générer un identifiant unique court
# Taille de Sortie : 32 bits (4 octets)
# Nombre de Combinaisons : 2^32 (environ 4,3 milliards)
crc32_hash = zlib.crc32(model_str.encode()) & 0xffffffff
unique_id = f"{crc32_hash:08x}"  # Formatage en chaîne hexadécimale
print("Identifiant unique (CRC32) :", unique_id)

def fletcher32(data):
    sum1 = 0xffff
    sum2 = 0xffff
    for byte in data:
        sum1 = (sum1 + byte) % 0xffff
        sum2 = (sum2 + sum1) % 0xffff
    return (sum2 << 16) | sum1
# Hachage Fletcher-32 pour générer un identifiant unique court
# Taille de Sortie : 32 bits (4 octets)
# Nombre de Combinaisons : 2^32 (environ 4,3 milliards)
fletcher32_hash = fletcher32(model_str.encode())
unique_id_fletcher32 = f"{fletcher32_hash:08x}"  # Formatage en chaîne hexadécimale
print("Identifiant unique (Fletcher-32) :", unique_id_fletcher32)

# Hachage Adler-32 pour générer un identifiant unique court
# Taille de Sortie : 32 bits (4 octets)
# Nombre de Combinaisons : 2^32 (environ 4,3 milliards)
adler32_hash = zlib.adler32(model_str.encode()) & 0xffffffff
unique_id_adler32 = f"{adler32_hash:08x}"  # Formatage en chaîne hexadécimale
print("Identifiant unique (Adler-32) :", unique_id_adler32)

