"""Caching functionality for conformers and parsed polymers."""

import hashlib
import pickle
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from rdkit import Chem
from rdkit.Chem import AllChem

if TYPE_CHECKING:
    from boltz.data.parse.schema import ParsedChain

LRU_CACHE_SIZE = 64


@lru_cache(maxsize=LRU_CACHE_SIZE)
def _load_conformer_from_cache(
    smiles: str,
    affinity: bool,
    cache_dir_str: Optional[str],
) -> Chem.Mol:
    """Load conformer from disk cache. LRU cached in-memory.

    Parameters
    ----------
    smiles : str
        The SMILES string of the molecule
    affinity : bool
        Whether this molecule is used for affinity prediction
    cache_dir_str : str, optional
        String path to cache directory. If None, no caching is performed.

    Returns
    -------
    Chem.Mol
        The molecule with 3D conformer

    Raises
    ------
    ValueError
        If conformer generation fails
    """
    # Late import to avoid circular dependency
    from boltz.data.parse.schema import compute_3d_conformer

    # Create mol with hydrogens
    mol = AllChem.MolFromSmiles(smiles)
    mol = AllChem.AddHs(mol)

    # If no cache directory, just compute conformer
    if cache_dir_str is None:
        success = compute_3d_conformer(mol)
        if not success:
            msg = f"Failed to compute 3D conformer for {smiles}"
            raise ValueError(msg)
        return mol

    cache_dir = Path(cache_dir_str)

    # Generate cache hash from SMILES + affinity flag
    try:
        cache_key_data = (smiles, affinity)
        cache_key_str = str(cache_key_data).encode("utf-8")
        cache_hash = hashlib.sha256(cache_key_str).hexdigest()[:16]
    except Exception as e:
        print(f"WARNING: Failed to generate cache key for {smiles}: {e}. Computing conformer without caching.")
        success = compute_3d_conformer(mol)
        if not success:
            msg = f"Failed to compute 3D conformer for {smiles}"
            raise ValueError(msg)
        return mol

    # Use single conformers directory (affinity flag already in hash)
    conformer_cache_dir = cache_dir / "conformers"
    conformer_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = conformer_cache_dir / f"{cache_hash}.pkl"

    # Try to load from cache
    if cache_path.exists():
        try:
            with cache_path.open("rb") as f:
                cached_data = pickle.load(f)

            cached_mol = cached_data["mol"]
            cached_smiles = cached_data["smiles"]
            cached_affinity = cached_data["affinity"]

            # Check for hash collision by comparing SMILES and affinity
            if smiles != cached_smiles or affinity != cached_affinity:
                print(
                    f"WARNING: Hash collision detected for {cache_hash}!\n"
                    f"  Query: smiles={smiles}, affinity={affinity}\n"
                    f"  Cached: smiles={cached_smiles}, affinity={cached_affinity}\n"
                    f"  Overwriting cache with new molecule."
                )
            else:
                # Cache hit with matching molecule
                return cached_mol

        except Exception as e:
            print(f"WARNING: Failed to load cached conformer from {cache_path}: {e}. Regenerating.")

    # Cache miss or collision - compute conformer
    success = compute_3d_conformer(mol)
    if not success:
        msg = f"Failed to compute 3D conformer for {smiles}"
        raise ValueError(msg)

    # Save to cache
    try:
        cache_data = {
            "mol": mol,
            "smiles": smiles,
            "affinity": affinity,
        }
        with cache_path.open("wb") as f:
            pickle.dump(cache_data, f)
    except Exception as e:
        print(f"WARNING: Failed to cache conformer to {cache_path}: {e}")

    return mol


def get_mol_with_conformer(
    smiles: str,
    affinity: bool,
    cache_dir: Optional[Path] = None,
) -> Chem.Mol:
    """Get a molecule with 3D conformer, using cache if available.

    This function caches conformers based on a hash of (SMILES, affinity).

    Parameters
    ----------
    smiles : str
        The SMILES string of the molecule
    affinity : bool
        Whether this molecule is used for affinity prediction
    cache_dir : Path, optional
        The cache directory. If None, no caching is performed.

    Returns
    -------
    Chem.Mol
        The molecule with 3D conformer

    Raises
    ------
    ValueError
        If conformer generation fails
    """
    cache_dir_str = str(cache_dir) if cache_dir is not None else None
    return _load_conformer_from_cache(smiles, affinity, cache_dir_str)


@lru_cache(maxsize=LRU_CACHE_SIZE)
def _load_polymer_from_cache(
    sequence_tuple: tuple[str, ...],
    chain_type: int,
    cyclic: bool,
    cache_dir_str: Optional[str],
) -> Optional["ParsedChain"]:
    """Load polymer from disk cache. LRU cached in-memory.

    Parameters
    ----------
    sequence_tuple : tuple[str, ...]
        The sequence of residue names as a tuple
    chain_type : int
        The chain type ID (protein/DNA/RNA)
    cyclic : bool
        Whether the chain is cyclic
    cache_dir_str : str, optional
        String path to cache directory. If None, returns None.

    Returns
    -------
    ParsedChain, optional
        The cached parsed chain object, or None if not cached

    """
    # If no cache directory, return None (no cache available)
    if cache_dir_str is None:
        return None

    cache_dir = Path(cache_dir_str)

    # Generate SHA256 hash for cache lookup
    try:
        cache_key_data = (sequence_tuple, chain_type, cyclic)
        cache_key_str = str(cache_key_data).encode("utf-8")
        # Use first 16 characters for shorter filenames
        cache_hash = hashlib.sha256(cache_key_str).hexdigest()[:16]
    except Exception as e:
        print(f"WARNING: Failed to generate cache key for polymer: {e}.")
        return None

    # Determine cache path
    polymer_cache_dir = cache_dir / "polymers"
    polymer_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = polymer_cache_dir / f"{cache_hash}.pkl"

    # Try to load from cache
    if cache_path.exists():
        try:
            with cache_path.open("rb") as f:
                cached_data = pickle.load(f)

            cached_chain = cached_data["chain"]
            cached_sequence = cached_data["sequence"]
            cached_chain_type = cached_data["chain_type"]
            cached_cyclic = cached_data["cyclic"]

            # Check for hash collision by comparing inputs
            if (
                sequence_tuple == tuple(cached_sequence)
                and chain_type == cached_chain_type
                and cyclic == cached_cyclic
            ):
                # Cache hit with matching polymer
                return cached_chain
            else:
                print(
                    f"WARNING: Cache collision detected for hash {cache_hash}!\n"
                    f"  Query: sequence_len={len(sequence_tuple)}, chain_type={chain_type}, cyclic={cyclic}\n"
                    f"  Cached: sequence_len={len(cached_sequence)}, chain_type={cached_chain_type}, cyclic={cached_cyclic}\n"
                    f"  Will overwrite cache with new polymer."
                )
                return None

        except Exception as e:
            print(f"WARNING: Failed to load cached polymer from {cache_path}: {e}.")
            return None

    # Cache miss
    return None


def get_polymer_with_cache(
    sequence: list[str],
    raw_sequence: str,
    entity: str,
    chain_type: int,
    components: dict[str, Chem.Mol],
    cyclic: bool,
    mol_dir: Path,
    cache_dir: Optional[Path] = None,
) -> Optional["ParsedChain"]:
    """Get a parsed polymer chain, using cache if available.

    This function caches parsed polymer chains based on a SHA256 hash of
    (sequence, chain_type, cyclic).

    Parameters
    ----------
    sequence : list[str]
        The sequence of residue names
    raw_sequence : str
        The raw sequence string
    entity : str
        The entity ID
    chain_type : int
        The chain type ID (protein/DNA/RNA)
    components : dict[str, Mol]
        The CCD components dictionary
    cyclic : bool
        Whether the chain is cyclic
    mol_dir : Path
        Path to the molecule directory
    cache_dir : Path, optional
        The cache directory. If None, no caching is performed.

    Returns
    -------
    ParsedChain, optional
        The parsed chain object

    """
    # Late import to avoid circular dependency
    from boltz.data.parse.schema import parse_polymer

    # Convert to hashable types for cache lookup
    sequence_tuple = tuple(sequence)
    cache_dir_str = str(cache_dir) if cache_dir is not None else None

    # Try to load from cache
    cached_chain = _load_polymer_from_cache(
        sequence_tuple, chain_type, cyclic, cache_dir_str
    )
    if cached_chain is not None:
        return cached_chain

    # Cache miss - parse the polymer
    parsed_chain = parse_polymer(
        sequence=sequence,
        raw_sequence=raw_sequence,
        entity=entity,
        chain_type=chain_type,
        components=components,
        cyclic=cyclic,
        mol_dir=mol_dir,
    )

    # Save to disk cache
    if parsed_chain is not None and cache_dir is not None:
        try:
            # Recreate cache path (same logic as in _load_polymer_from_cache)
            cache_key_data = (sequence_tuple, chain_type, cyclic)
            cache_key_str = str(cache_key_data).encode("utf-8")
            # Use first 16 characters for shorter filenames
            cache_hash = hashlib.sha256(cache_key_str).hexdigest()[:16]

            polymer_cache_dir = cache_dir / "polymers"
            polymer_cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = polymer_cache_dir / f"{cache_hash}.pkl"

            cache_data = {
                "chain": parsed_chain,
                "sequence": sequence,
                "chain_type": chain_type,
                "cyclic": cyclic,
            }
            with cache_path.open("wb") as f:
                pickle.dump(cache_data, f)
        except Exception as e:
            print(f"WARNING: Failed to cache polymer: {e}")

    return parsed_chain
